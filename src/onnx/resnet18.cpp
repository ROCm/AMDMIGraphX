#include <cstdio>
#include <string>
#include <fstream>
#include <numeric>
#include <stdexcept>

#include <migraph/onnx.hpp>

#include <migraph/gpu/target.hpp>
#include <migraph/gpu/hip.hpp>
#include <migraph/generate.hpp>

int main(int argc, char const* argv[])
{
    std::string file = argv[1];
    auto prog        = migraph::parse_onnx(file);

    // GPU target
    prog.compile(migraph::gpu::target{});
    migraph::program::parameter_map m;
    auto s = migraph::shape{migraph::shape::float_type, {1, 3, 32, 32}};
    m["output"] =
        migraph::gpu::to_gpu(migraph::generate_argument(prog.get_parameter_shape("output")));
    m["0"]      = migraph::gpu::to_gpu(migraph::generate_argument(s, 12345));
    auto result = migraph::gpu::from_gpu(prog.eval(m));

    // // CPU target
    // prog.compile(migraph::cpu::cpu_target{});
    // auto s = migraph::shape{migraph::shape::float_type, {1, 3, 32, 32}};
    // auto input3 = migraph::generate_argument(s, 12345);
    // auto result = prog.eval({{"0", input3}});
}
