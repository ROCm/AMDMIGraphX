# MIGraphX Driver
The MIGraphX driver is located under /opt/rocm/bin/migraphx-driver.  The --help command shows options available
```
$ /opt/rocm/bin/migraphx-driver --help

    -h, --help
        Show help

Commands:
    op
    params
    run
    read
    compile
    verify
    perf

```
For example, the read command will read a file and print the internal graph from MIGraphx
```
$ /opt/rocm/bin/migraphx-driver read --model resnet50
@0 = @literal{ ... } -> float_type, {64, 3, 7, 7}, {147, 49, 7, 1}
@1 = @literal{ ... } -> float_type, {64}, {1}
@2 = @literal{ ... } -> float_type, {64}, {1}
...
...
@441 = transpose[dims={1, 0}](@265) -> float_type, {2048, 1000}, {1, 2048}
@442 = multibroadcast[output_lens={1, 1000}](@266) -> float_type, {1, 1000}, {0, 1}
@443 = dot[alpha=1,beta=1](@440,@441,@442) -> float_type, {1, 1000}, {1000, 1}

```
Another example, the following command measures performance running an ONNX file using the driver
```
/opt/rocm/bin/migraphx-driver perf --onnx /home/mev/source/migraphx_onnx/torchvision/resnet50i64.onnx 
```
The verify command is a correctness checker and runs both the reference and GPU implementation to ensure the model outputs are consistent 
```
/opt/rocm/bin/migraphx-driver verify --onnx /home/mev/source/migraphx_onnx/torchvision/resnet50i64.onnx 
```