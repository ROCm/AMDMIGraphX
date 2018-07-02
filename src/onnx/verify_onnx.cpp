
#include <migraph/onnx.hpp>

#include <migraph/cpu/cpu_target.hpp>
#include <migraph/miopen/miopen_target.hpp>
#include <migraph/miopen/hip.hpp>
#include <migraph/generate.hpp>
#include <miopen/miopen.h>
#include <migraph/miopen/miopen.hpp>

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
    auto input3 = migraph::miopen::to_gpu(migraph::generate_argument(s));

    auto output = migraph::miopen::to_gpu(migraph::generate_argument(p.get_parameter_shape("output")));
    auto handle = migraph::miopen::make_obj<migraph::miopen::miopen_handle>(&miopenCreate);

    auto out = p.eval({{"Input3", input3}, {"output", output}});
    std::cout << p << std::endl;
    return migraph::miopen::from_gpu(out);
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
