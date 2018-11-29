
#include <sys/stat.h>
#include <cstdlib>
#include <cassert>

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

#include <cuda_runtime_api.h>

#include "NvInfer.h"
#include "NvUffParser.h"
#include "NvUtils.h"

#include "common.h"

using namespace nvuffparser;
using namespace nvinfer1;


static Logger gLogger;
static int gUseDLACore{ -1 };
static std::vector<std::string> classes{ "Caterpillar",
										"Dirty",
										"Flat flower",
										"Hole",
										"Mold",
										"pressing",
										"Scratch",
										"Spot",
										"Warped",
										"Zinc ash",
										"Zinc residue" };

#define MAX_WORKSPACE (1 << 30)

#define RETURN_AND_LOG(ret, severity, message)                                              \
    do {                                                                                    \
        std::string error_message = "sample_uff_detection: " + std::string(message);            \
        gLogger.log(ILogger::Severity::k ## severity, error_message.c_str());               \
        return (ret);                                                                       \
    } while(0)

inline int64_t volume(const Dims& d)
{
	int64_t v = 1;
	for (int64_t i = 0; i < d.nbDims; i++)
		v *= d.d[i];
	return v;
}


inline unsigned int elementSize(DataType t)
{
	switch (t)
	{
	case DataType::kINT32:
		// Fallthrough, same as kFLOAT
	case DataType::kFLOAT: return 4;
	case DataType::kHALF: return 2;
	case DataType::kINT8: return 1;
	}
	assert(0);
	return 0;
}


static const int INPUT_H = 224;
static const int INPUT_W = 224;
static const int OUTPUT_SIZE = 11;


std::string locateFile(const std::string& input)
{
	std::vector<std::string> dirs{ "data/defet_detection/", "data/samples/defet_detection/" };
	return locateFile(input, dirs);
}


// simple PGM (portable greyscale map) reader
void readPGMFile(const std::string& filename, uint8_t buffer[INPUT_H*INPUT_W])
{
	readPGMFile(locateFile(filename), buffer, INPUT_H, INPUT_W);
}


void* safeCudaMalloc(size_t memSize)
{
	void* deviceMem;
	CHECK(cudaMalloc(&deviceMem, memSize));
	if (deviceMem == nullptr)
	{
		std::cerr << "Out of memory" << std::endl;
		exit(1);
	}
	return deviceMem;
}


std::vector<std::pair<int64_t, DataType>>
calculateBindingBufferSizes(const ICudaEngine& engine, int nbBindings, int batchSize)
{
	std::vector<std::pair<int64_t, DataType>> sizes;
	for (int i = 0; i < nbBindings; ++i)
	{
		Dims dims = engine.getBindingDimensions(i);
		DataType dtype = engine.getBindingDataType(i);

		int64_t eltCount = volume(dims) * batchSize;
		sizes.push_back(std::make_pair(eltCount, dtype));
	}

	return sizes;
}


void* createDetectionCudaBuffer(int64_t eltCount, DataType dtype, std::string run)
{
	/* in that specific case, eltCount == INPUT_H * INPUT_W */
	assert(eltCount == INPUT_H * INPUT_W);
	assert(elementSize(dtype) == sizeof(float));

	size_t memSize = eltCount * elementSize(dtype);
	float* inputs = new float[eltCount];

	/* read PGM file */
	uint8_t fileData[INPUT_H * INPUT_W];
	readPGMFile(run + ".pgm", fileData);

	/* initialize the inputs buffer */
	for (int i = 0; i < eltCount; i++)
		inputs[i] = float(fileData[i]);

	void* deviceMem = safeCudaMalloc(memSize);
	CHECK(cudaMemcpy(deviceMem, inputs, memSize, cudaMemcpyHostToDevice));

	delete[] inputs;
	return deviceMem;
}


void printOutput(int64_t eltCount, DataType dtype, void* buffer)
{
	std::cout << eltCount << " eltCount" << std::endl;
	assert(elementSize(dtype) == sizeof(float));
	std::cout << "--- OUTPUT ---" << std::endl;

	size_t memSize = eltCount * elementSize(dtype);
	float* outputs = new float[eltCount];
	CHECK(cudaMemcpy(outputs, buffer, memSize, cudaMemcpyDeviceToHost));

	int maxIdx = 0;
	for (int i = 0; i < eltCount; ++i)
		if (outputs[i] > outputs[maxIdx])
			maxIdx = i;

	std::ios::fmtflags prevSettings = std::cout.flags();
	std::cout.setf(std::ios::fixed, std::ios::floatfield);
	std::cout.precision(6);
	for (int64_t eltIdx = 0; eltIdx < eltCount; ++eltIdx)
	{
		std::cout << eltIdx << " => " << setw(10) << outputs[eltIdx] << "\t : ";
		if (eltIdx == maxIdx)
			std::cout << "***---" << "---Class: " << classes[maxIdx];
		std::cout << "\n";
	}
	std::cout.flags(prevSettings);

	std::cout << std::endl;
	delete[] outputs;
}


void* loadModelAndCreateEngine(const char* uffFile, int maxBatchSize,
	IUffParser* parser, IHostMemory*& trtEngineStream)
{
	IBuilder* builder = createInferBuilder(gLogger);
	INetworkDefinition* network = builder->createNetwork();

#if 1
	if (!parser->parse(uffFile, *network, nvinfer1::DataType::kFLOAT))
		RETURN_AND_LOG(nullptr, ERROR, "Fail to parse");
#else
	if (!parser->parse(uffFile, *network, nvinfer1::DataType::kHALF))
		RETURN_AND_LOG(nullptr, ERROR, "Fail to parse");
	builder->setFp16Mode(true);
#endif

	/* we create the engine */
	builder->setMaxBatchSize(maxBatchSize);
	builder->setMaxWorkspaceSize(MAX_WORKSPACE);
	samplesCommon::enableDLA(builder, gUseDLACore);

	ICudaEngine* engine = builder->buildCudaEngine(*network);
	if (!engine)
		RETURN_AND_LOG(nullptr, ERROR, "Unable to create engine");

	/* we can clean the network and the parser */
	network->destroy();
	builder->destroy();

	trtEngineStream = engine->serialize();

	engine->destroy();
	shutdownProtobufLibrary();
}


void execute(ICudaEngine& engine)
{
	IExecutionContext* context = engine.createExecutionContext();

	int batchSize = 1;

	int nbBindings = engine.getNbBindings();
	assert(nbBindings == 2);

	std::vector<void*> buffers(nbBindings);
	auto buffersSizes = calculateBindingBufferSizes(engine, nbBindings, batchSize);

	int bindingIdxInput = 0;
	for (int i = 0; i < nbBindings; ++i)
	{
		if (engine.bindingIsInput(i))
			bindingIdxInput = i;
		else
		{
			auto bufferSizesOutput = buffersSizes[i];
			buffers[i] = safeCudaMalloc(bufferSizesOutput.first *
				elementSize(bufferSizesOutput.second));
		}
	}

	auto bufferSizesInput = buffersSizes[bindingIdxInput];

	int iterations = 1;
	int numberRun = 11;
	for (int i = 0; i < iterations; i++)
	{
		float total = 0, ms;
		for (int run = 0; run < numberRun; run++)
		{
			buffers[bindingIdxInput] = createDetectionCudaBuffer(bufferSizesInput.first,
				bufferSizesInput.second, classes[run]);

			auto t_start = std::chrono::high_resolution_clock::now();
			context->execute(batchSize, &buffers[0]);
			auto t_end = std::chrono::high_resolution_clock::now();
			ms = std::chrono::duration<float, std::milli>(t_end - t_start).count();
			total += ms;

			for (int bindingIdx = 0; bindingIdx < nbBindings; ++bindingIdx)
			{
				if (engine.bindingIsInput(bindingIdx))
					continue;

				auto bufferSizesOutput = buffersSizes[bindingIdx];
				printOutput(bufferSizesOutput.first, bufferSizesOutput.second,
					buffers[bindingIdx]);
			}
			CHECK(cudaFree(buffers[bindingIdxInput]));
		}

		total /= numberRun;
		std::cout << "Average over " << numberRun << " runs is " << total << " ms." << std::endl;
	}

	for (int bindingIdx = 0; bindingIdx < nbBindings; ++bindingIdx)
		if (!engine.bindingIsInput(bindingIdx))
			CHECK(cudaFree(buffers[bindingIdx]));
	context->destroy();
}


int main(int argc, char** argv)
{
	int status = cudaSetDevice(0);
	gUseDLACore = samplesCommon::parseDLA(argc, argv);
	auto fileName = locateFile("defect.uff");
	std::cout << fileName << std::endl;

	int maxBatchSize = 1;
	auto parser = createUffParser();

	/* Register tensorflow input */
	parser->registerInput("in", Dims3(1, 224, 224), UffInputOrder::kNCHW);
	parser->registerOutput("Softmax");

	IHostMemory *trtEngineStream{ nullptr };

	loadModelAndCreateEngine(fileName.c_str(), maxBatchSize, parser, trtEngineStream);

	if (!trtEngineStream)
		RETURN_AND_LOG(EXIT_FAILURE, ERROR, "Model load failed");

	cout << trtEngineStream->size() << std::endl;


	//serialize the engine and save to 'engineStream.bin' file in data/mnist/ folder
	fstream os("../../data/defet_detection/engineStream.bin", std::ios::out | std::ios::binary);
	os.write((const char*)trtEngineStream->data(), trtEngineStream->size());
	os.close();

	/* we need to keep the memory created by the parser */
	parser->destroy();

	IRuntime* runtime = createInferRuntime(gLogger);
	nvinfer1::IPluginFactory* factory{ nullptr };
	ICudaEngine* engine = runtime->deserializeCudaEngine(trtEngineStream->data(), trtEngineStream->size(), factory);

	runtime->destroy();
	trtEngineStream->destroy();

	auto t_start = std::chrono::high_resolution_clock::now();
	execute(*engine);
	auto t_end = std::chrono::high_resolution_clock::now();

	std::cout << std::chrono::duration<float, std::milli>(t_end - t_start).count() << " / ms" << std::endl;

	engine->destroy();
	shutdownProtobufLibrary();

	system("pause");

	return EXIT_SUCCESS;
}
