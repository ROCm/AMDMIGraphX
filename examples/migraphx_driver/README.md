# MIGraphX Driver
There is a new driver as /opt/rocm/bin/migraphx-driver.  The --help command shows options available
```
prompt$ /opt/rocm/bin/migraphx-driver --help

    -h, --help
        Show help

Commands:
    perf
    params
    read
    run
    verify
    compile
```
For example, the read command will read a file and print the internal graph from MIGraphx
```
/opt/rocm/bin/migraphx-driver read --onnx /home/mev/source/migraphx_onnx/torchvision/resnet50i64.onnx 
```
Another example, the following command measures performance running an ONNX file using the driver
```
/opt/rocm/bin/migraphx-driver perf --onnx /home/mev/source/migraphx_onnx/torchvision/resnet50i64.onnx 
```
The verify command checks internal consistency once read into MIGraphX
```
/opt/rocm/bin/migraphx-driver verify --onnx /home/mev/source/migraphx_onnx/torchvision/resnet50i64.onnx 
```