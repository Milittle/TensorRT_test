
#include "NvInfer.h"
#include "common.h"
#include <fstream>

static Logger gLogger{nvinfer1::ILogger::Severity::kINFO};
static const int INPUT_H = 28;
static const int INPUT_W = 28;
static const int OUTPUT_SIZE = 10;


#define RETURN_AND_LOG(ret, severity, message)                                              \
    do {                                                                                    \
        std::string error_message = "sample_uff_mnist: " + std::string(message);            \
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
	std::vector<std::string> dirs{ "data/mnist/", "data/samples/mnist/" };
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


void* createMnistCudaBuffer(int64_t eltCount, DataType dtype, int run)
{
	/* in that specific case, eltCount == INPUT_H * INPUT_W */
	assert(eltCount == INPUT_H * INPUT_W);
	assert(elementSize(dtype) == sizeof(float));

	size_t memSize = eltCount * elementSize(dtype);
	float* inputs = new float[eltCount];

	/* read PGM file */
	uint8_t fileData[INPUT_H * INPUT_W];
	readPGMFile(std::to_string(run) + ".pgm", fileData);

	/* display the number in an ascii representation */
	std::cout << "\n\n\n---------------------------" << "\n\n\n" << std::endl;
	for (int i = 0; i < eltCount; i++)
		std::cout << (" .:-=+*#%@"[fileData[i] / 26]) << (((i + 1) % INPUT_W) ? "" : "\n");

	/* initialize the inputs buffer */
	for (int i = 0; i < eltCount; i++)
		inputs[i] = 1.0f - float(fileData[i]) / 255.0f;

	void* deviceMem = safeCudaMalloc(memSize);
	CHECK(cudaMemcpy(deviceMem, inputs, memSize, cudaMemcpyHostToDevice));

	delete[] inputs;
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

	int iterations = 1;
	int numberRun = 10;
	for (int i = 0; i < iterations; i++)
	{
		float total = 0, ms;
		for (int run = 0; run < numberRun; run++)
		{
			buffers[bindingIdxInput] = createMnistCudaBuffer(bufferSizesInput.first,
				bufferSizesInput.second, run);

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
	nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(gLogger);
	nvinfer1::IPluginFactory* factory{ nullptr };
	std::ifstream is("../../data/mnist/engineStream.bin", std::ios::binary);
	is.seekg(0, is.end);
	int length = is.tellg();
	is.seekg(0, is.beg);

	try {
		char *buffer = new char[length];
		is.read(buffer, length);
		is.close();

		nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine(buffer, length, factory);
		if (!engine)
			RETURN_AND_LOG(-1, ERROR, "load the engine failed");
		execute(*engine);

		delete[] buffer;
		runtime->destroy();
		engine->destroy();

		system("pause");
		return 0;
	}
	catch (...) {
		is.close();
	}
	return 0;
	
}