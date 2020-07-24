# TFLite Model Benchmarking
This tool is meant to analyze and Benchmark a given TensorFlow Lite (TFLite) model with respect to its timing and space requirements.

## Tools Used
To analyze a TFLite model, we will benchmark it by the following attributes:
*   Aggregate latency statistics
*   Number of Weights
*   Weights memory size
*   Tensors Used

To measure the above attributes I am using the following open-source tools:
*   [TF Benchmark Tool](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/benchmark): This tool is found in the tensorflow repo and is used to estimate the model's latency by measuring the initialization time, 1st inference time, average warmup time, average inference time.
*   [tflite_analyzer](https://github.com/PeteBlackerThe3rd/tflite_analyser): This tool estimates the memory requirements of the model, i.e. the Number of weights, their memory size, the name of the tensors used, and their dynamic memory allocation.