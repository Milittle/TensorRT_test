# TensorRT5.0 Test Integration

Project **TensorRT_test** is a TensorRT Library Example integrated based on Windows Visual Studio 2017, which make our machine learning can run fastly at inference stage.

>you can look more information about **TensorRT** in [TensorRT Dev Guide](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html) 

## Not NVIDIA TensorRT official Sample(BY Myself)

* **sampleLoadEngineStream:** deserializing the engine stream by `engineStream.bin` locating in the `{SolutionDir}/data/mnist/` folder.
* **sampleResNetv2**: using the Resnetv2 pb file transform to uff file and executing the inference.
* **sampleDetection**: (Defect Detection Demo)Solving the TensorFlow BatchNormalization operator. TensorRT do not support the BN's Switch and Merge. I use pb graph and remove some nodes about Switching and Merging then merging related node to pb's graph, which  convert to uff file using for TensorRT uff parser parsing the model file.
I use ten defect images to inference the results. So the fllowing time performance is 10 images inferencing time.

**sampleDetection Time consume:**

|      tensorflow（python）- Titan-12G       |     tensorrt（c++）- Qudra 4G      | Conclusion |
| :----------------------------------------: | :--------------------------------: | :--------: |
|        pure run time（1344.3049ms）        |   pure execution time（44.5ms）    |  30 times  |
| load data and related Tensor nodes（3473ms） | load data and execute（171.373ms） |  20 times  |
|                GPU mem-2GB                 |                ---                 |            |

## Table of Content

- [TensorRT5.0 Test Integration](#tensorrt50-test-integration)
  - [Not NVIDIA TensorRT official Sample(BY Myself)](#not-nvidia-tensorrt-official-sampleby-myself)
  - [Table of Content](#table-of-content)
  - [Prerequisites](#prerequisites)
  - [Getting the code](#getting-the-code)
  - [Project Structure](#project-structure)
  - [Run the Example using VS](#run-the-example-using-vs)
    - [sampleUffMNIST](#sampleuffmnist)
    - [sampleUffSSD](#sampleuffssd)
    - [sampleMNIST](#samplemnist)
    - [sampleMNISTAPI](#samplemnistapi)
    - [sampleSSD](#samplessd)
    - [samplePlugin](#sampleplugin)
    - [sampleCharRNN](#samplecharrnn)
    - [sampleFasterRCNN](#samplefasterrcnn)
    - [sampleGoogleNet](#samplegooglenet)
    - [sampleINT8](#sampleint8)
    - [sampleMLP](#samplemlp)
    - [sampleMovieLens](#samplemovielens)
    - [sampleNMT](#samplenmt)
  - [Contact Getting Help](#contact-getting-help)

## Prerequisites

* CUDA 10.0  [DOWNLOAD LINK](https://developer.nvidia.com/cuda-downloads)
* cudnn 7.3 [DOWNLOAD LINK](https://developer.nvidia.com/cudnn)
* You need the Visual Stdio 2017
* The DataSets in my Google Driver

## Getting the code

You can use the git tool to clone the Project, through:

```shell
git clone git@github.com:Milittle/TensorRT_test.git
```

## Project Structure

The Following is my Integrated Project's Structure, and you can download **data** and **3rdparty** by:

**Google Driver** : [data and 3rdparty download link](https://drive.google.com/drive/folders/1mDKSmK5n2n7KnZhW5mQbUSJTSzZteN8c?usp=sharing)

Once you download the data and 3rdparty, you can open the TenosrRT_test.sln file and exec the samples by Visual Studio 2017.

Good luck to you.

```shell
TensorRT_test:
|	3rdparty
└---|	TensorRT-5.0.1.3
|	└-------------------
|	common
└---|	windows
|	|	argsParser.h
|	|	BatchStream.h
|	|	buffers.h
|	|	common.h
|	|	dumpTFWts.py
|	|	half.h
|	|	sampleConfig.h
|	└-------------------
|	data
└---|	char-rnn
|	|	example_gif
|	|	faster-rcnn
|	|	googlenet
|	|	mlp
|	|	mnist
|	|	movielens
|	|	nmt
|	|	ssd
|	└-------------------
|	src
└---|	sampleCharRNN
|	|	sampleFasterRCNN
|	|	sampleGoogleNet
|	|	sampleINT8
|	|	sampleMLP
|	|	sampleMNIST
|	|	sampleMNISTAPI
|	|	sampleMovieLens
|	|	sampleNMT
|	|	samplePlugin
|	|	sampleUffMNIST
|	|	sampleUffSSD
|	└--------------------
|	.gitignore
└------------------------
|	README.md
└------------------------
|	TensorRT_test.sln
└------------------------
```

## Run the Example using VS

### sampleUffMNIST

![Demo](https://s1.ax1x.com/2018/10/28/ig9UTe.gif)

### sampleUffSSD

This example load the model and build the engine taking a long time, you need more patience.

step1: Begin parsing model...

​	    End parsing model...

step2: Begin building engine...

​	    End building engine...

step3: Begin inference.

![](https://s1.ax1x.com/2018/10/29/igNDaT.gif)

### sampleMNIST

![](https://s1.ax1x.com/2018/10/29/igNcRJ.gif)

### sampleMNISTAPI

![](https://s1.ax1x.com/2018/10/29/igNgz9.gif)

### sampleSSD

This example has some error, I cannot through the model prototxt parser the model.

![]()

### samplePlugin

![](https://s1.ax1x.com/2018/10/29/igNWs1.gif)

### sampleCharRNN

![](https://s1.ax1x.com/2018/10/29/igN5dK.gif)

### sampleFasterRCNN

![](https://s1.ax1x.com/2018/10/29/igN7Je.gif)

### sampleGoogleNet

![](https://s1.ax1x.com/2018/10/29/igNHRH.gif)

### sampleINT8

**Note**: my computer isn't support the FP16 and INT8. so:

![](https://s1.ax1x.com/2018/10/29/igNxdf.gif)

### sampleMLP

![](https://s1.ax1x.com/2018/10/29/igNzo8.gif)

### sampleMovieLens

![](https://s1.ax1x.com/2018/10/29/igUpFS.gif)

### sampleNMT

![](https://s1.ax1x.com/2018/10/29/igUNFO.gif)

## Contact Getting Help

**Email:** mizeshuang@gmail.com

**QQ**: 329804334

**Author:**  Milittle
