# ONNX Runtime driver
The onnx runtime driver is a quick way to test onnx files via onnx runtime.
## Installation Instructions
Use the command below to install dependencies:
```
pip install -r requirements.txt
```
## Usage
Example usage is below:
```
python ort_driver.py --onnx [path to onnx_file]
```

The output of the driver will be `"onnx runtime driver completed successfully."` if there are no errors. For detailed information,
pass the `--verbose` flag to the command line. It shows information such as input parameters and output of onnx runtime.

Random values are assigned to the model's inputs. However, they can be set to only contain 1s if the `--fill1` flag is passed in,
or only 0s if the `--fill0` flag is passed in.
This is potentially useful for verifying models such as bert which use integer datatypes.

Onnx models may use dim values on their shapes to make their values dynamic. Use `--default-dim-value` them to a value other than 1.

If the model has multiple dim values in the inputs, use `--input-dim` to specify the shape. Example syntax:
```
--input-dim input_ids:1,384
```
Multiple inputs can be set by adding additional `--input-dim` flags.

By default, the CPU Execution Provider is used when running onnx runtime. If building onnx runtime with a different version, specify the provider using `--provider`.

For verbose onnx runtime logging, pass in `--ort-logging`. 
