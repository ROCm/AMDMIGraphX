
#include <migraph/onnx.hpp>

#include <migraph/cpu/cpu_target.hpp>
#include <migraph/gpu/target.hpp>
#include <migraph/gpu/hip.hpp>
#include <migraph/generate.hpp>
#include <miopen/miopen.h>
#include <migraph/gpu/miopen.hpp>

migraph::argument run_cpu(std::string file)
{
    auto p = migraph::parse_onnx(file);
    p.compile(migraph::cpu::cpu_target{});
    auto s      = p.get_parameter_shape("Input3");
    auto input3 = migraph::generate_argument(s);
    auto out    = p.eval({{"Input3", input3}});
    std::cout << p << std::endl;
    return out;
}

migraph::argument run_gpu(std::string file)
{
    auto p = migraph::parse_onnx(file);
    p.compile(migraph::cpu::cpu_target{});
    auto s      = p.get_parameter_shape("Input3");
    auto input3 = migraph::gpu::to_gpu(migraph::generate_argument(s));

    auto output = migraph::gpu::to_gpu(migraph::generate_argument(p.get_parameter_shape("output")));
    auto handle = migraph::gpu::make_obj<migraph::gpu::miopen_handle>(&miopenCreate);

    auto out = p.eval({{"Input3", input3}, {"output", output}});
    std::cout << p << std::endl;
    return migraph::gpu::from_gpu(out);
}

int main(int argc, char const* argv[])
{
    if(argc > 1)
    {
        std::string file = argv[1];
        auto x           = run_cpu(file);
        auto y           = run_gpu(file);
        if(x == y)
        {
            std::cout << "Passed" << std::endl;
        }
        else
        {
            std::cout << "Not equal" << std::endl;
            std::cout << x << std::endl;
            std::cout << y << std::endl;
        }
    }
}
