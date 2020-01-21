#include <migraphx/migraphx.h>
#include <migraphx/migraphx.hpp>
#include "test.hpp"

TEST_CASE(load_and_run) 
{ 
    auto p = migraphx::parse_onnx("conv_relu_maxpool_test.onnx");
    migraphx_compile_options options;
    options.offload_copy = true;
    p.compile(migraphx::target("gpu"), options);
    migraphx::program_parameters pp;
    auto param_shapes = p.get_parameter_shapes();
    for(auto&& name:param_shapes.names())
    {
        pp.add(name, migraphx::argument::generate(param_shapes[name]));
    }
    p.eval(pp);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
