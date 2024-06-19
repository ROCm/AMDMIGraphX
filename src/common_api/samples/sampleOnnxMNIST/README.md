# "Hello World" For TensoRT From ONNX

## Description
This sample, sampleOnnxMNIST, converts a model trained on the MNIST dataset in Open Neural Network Exchange (ONNX) format to a TensorRT network and runs inference on the network.

ONNX is a standard for representing deep learning models that enables models to be transferred between frameworks.

## Running the sample

```bash
# Within the build directory of the MIGraphX project
make -j($nproc) MNISTSample
./bin/MNISTSample
```

**Note:** by default the sample will look for data in `code/AMDMIGraphX/src/common_api/sample_data/mnist`, to change the data directory the `-d` option can be used, e.g. `./binMNISTSample -d /path/to/data`.

If the sample ran successfully, the output should look similar to this:
```
	Input:
	@@@@@@@@@@@@@@@@@@@@@@@@@@@@
	@@@@@@@@@@@@@@@@@@@@@@@@@@@@
	@@@@@@@@@@@@@@@@@@@@@@@@@@@@
	@@@@@@@@@@@@@@@@@@@@@@@@@@@@
	@@@@@@@@@@@*.  .*@@@@@@@@@@@
	@@@@@@@@@@*.     +@@@@@@@@@@
	@@@@@@@@@@. :#+   %@@@@@@@@@
	@@@@@@@@@@.:@@@+  +@@@@@@@@@
	@@@@@@@@@@.:@@@@:  +@@@@@@@@
	@@@@@@@@@@=%@@@@:  +@@@@@@@@
	@@@@@@@@@@@@@@@@#  +@@@@@@@@
	@@@@@@@@@@@@@@@@*  +@@@@@@@@
	@@@@@@@@@@@@@@@@:  +@@@@@@@@
	@@@@@@@@@@@@@@@@:  +@@@@@@@@
	@@@@@@@@@@@@@@@*  .@@@@@@@@@
	@@@@@@@@@@%**%@.  *@@@@@@@@@
	@@@@@@@@%+.  .:  .@@@@@@@@@@
	@@@@@@@@=  ..    :@@@@@@@@@@
	@@@@@@@@:  *@@:  :@@@@@@@@@@
	@@@@@@@%   %@*    *@@@@@@@@@
	@@@@@@@%   ++ ++  .%@@@@@@@@
	@@@@@@@@-    +@@-  +@@@@@@@@
	@@@@@@@@=  :*@@@#  .%@@@@@@@
	@@@@@@@@@+*@@@@@%.   %@@@@@@
	@@@@@@@@@@@@@@@@@@@@@@@@@@@@
	@@@@@@@@@@@@@@@@@@@@@@@@@@@@
	@@@@@@@@@@@@@@@@@@@@@@@@@@@@
	@@@@@@@@@@@@@@@@@@@@@@@@@@@@

	[I] Output:
	Prob 0 0.0000 Class 0:
	Prob 1 0.0000 Class 1:
	Prob 2 1.0000 Class 2: **********
	Prob 3 0.0000 Class 3:
	Prob 4 0.0000 Class 4:
	Prob 5 0.0000 Class 5:
	Prob 6 0.0000 Class 6:
	Prob 7 0.0000 Class 7:
	Prob 8 0.0000 Class 8:
	Prob 9 0.0000 Class 9:
	```