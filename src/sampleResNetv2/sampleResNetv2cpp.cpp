

#include<chrono>
#include<vector>

#include<NvInfer.h>
#include<NvUffParser.h>
#include"common.h"



static Logger gLogger;
static int gUseDLACore{ -1 };


static const string input_filename = "bird.ppm";
static const int INPUT_H = 224;
static const int INPUT_W = 224;
static const int INPUT_C = 3;

struct PPM
{
	std::string magic, fileName;
	int h, w, max;
	uint8_t buffer[INPUT_C * INPUT_H * INPUT_W];
};

#define MAX_WORKSPACE (1 << 30)

#define RETURN_AND_LOG(ret, severity, message)                                              \
    do {                                                                                    \
        std::string error_message = "resnet_uff_imagenet: " + std::string(message);            \
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


std::string locateFile(const std::string& input)
{
	std::vector<std::string> dirs{ "data/resnet/", "data/samples/resnet/" };
	return locateFile(input, dirs);
}


// simple PGM (portable greyscale map) reader
void readPPMFile(const std::string& filename, PPM &ppm)
{
	ppm.fileName = filename;
	std::ifstream infile(locateFile(filename), std::ios::binary);
	infile >> ppm.magic >> ppm.w >> ppm.h >> ppm.max;
	infile.seekg(1, infile.cur);
	infile.read(reinterpret_cast<char*>(ppm.buffer), ppm.w * ppm.h * ppm.max);
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


void* loadModelAndCreateEngine(const char* uffFile, int maxBatchSize,
	nvuffparser::IUffParser* parser, IHostMemory*& trtEngineStream)
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
	nvuffparser::shutdownProtobufLibrary();
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


void* createResNetCudaBuffer(int64_t eltCount, DataType dtype, const string run)
{
	/* in that specific case, eltCount == INPUT_H * INPUT_W */
	assert(eltCount == INPUT_H * INPUT_W * INPUT_C);
	assert(elementSize(dtype) == sizeof(float));

	size_t memSize = eltCount * elementSize(dtype);
	float* inputs = new float[eltCount];

	/* read PPM file */
	PPM fileData;
	readPPMFile(run, fileData);

	/* initialize the inputs buffer */
	for (int i = 0; i < eltCount; i++)
		inputs[i] = fileData.buffer[i];

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
			std::cout << "***";
		std::cout << "\n";
	}
	std::cout.flags(prevSettings);

	std::cout << std::endl;
	delete[] outputs;
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
	buffers[bindingIdxInput] = createResNetCudaBuffer(bufferSizesInput.first,
		bufferSizesInput.second, input_filename);

	auto t_start = std::chrono::high_resolution_clock::now();
	context->execute(batchSize, &buffers[0]);
	auto t_end = std::chrono::high_resolution_clock::now();
	long long ms = std::chrono::duration<float, std::milli>(t_end - t_start).count();
	std::cout << "total consume: " << ms << " ms" << std::endl;

	for (int bindingIdx = 0; bindingIdx < nbBindings; ++bindingIdx)
	{
		if (engine.bindingIsInput(bindingIdx))
			continue;

		auto bufferSizesOutput = buffersSizes[bindingIdx];
		printOutput(bufferSizesOutput.first, bufferSizesOutput.second,
			buffers[bindingIdx]);
	}
	CHECK(cudaFree(buffers[bindingIdxInput]));

	for (int bindingIdx = 0; bindingIdx < nbBindings; ++bindingIdx)
		if (!engine.bindingIsInput(bindingIdx))
			CHECK(cudaFree(buffers[bindingIdx]));
	context->destroy();
}


int main(int argc, char **argv)
{
	gUseDLACore = samplesCommon::parseDLA(argc, argv);
	auto fileName = locateFile("resnetv2.uff");
	std::cout << fileName << std::endl;

	int maxBatchSize = 1;

	auto parser = nvuffparser::createUffParser();
	parser->registerInput("input_tensor", Dims3(3, INPUT_H, INPUT_W), nvuffparser::UffInputOrder::kNCHW);
	parser->registerOutput("softmax_tensor");

	IHostMemory *trtEngineStream{ nullptr };

	auto t_start = std::chrono::high_resolution_clock::now();
	loadModelAndCreateEngine(fileName.c_str(), maxBatchSize, parser, trtEngineStream);
	auto t_end = std::chrono::high_resolution_clock::now();
	long long ms = std::chrono::duration<float, std::milli>(t_end - t_start).count();
	std::cout << "Load model and create Engine consume: " << ms << " ms" << std::endl;

	if (!trtEngineStream)
		RETURN_AND_LOG(EXIT_FAILURE, ERROR, "Model load failed");

	//serialize the engine and save to 'engineStream.bin' file in data/resnet/ folder
	fstream os("../../data/resnet/engineStream.bin", std::ios::out | std::ios::binary);
	os.write((const char*)trtEngineStream->data(), trtEngineStream->size());
	os.close();

	parser->destroy();

	IRuntime *runtime = nvinfer1::createInferRuntime(gLogger);
	nvinfer1::IPluginFactory *factory{ nullptr };
	ICudaEngine *engine = runtime->deserializeCudaEngine(trtEngineStream->data(), trtEngineStream->size(), factory);

	execute(*engine);

	engine->destroy();
	runtime->destroy();
	trtEngineStream->destroy();

	system("pause");

	return 0;

}