
#include <rtg/onnx.hpp>

#include <rtg/cpu/cpu_target.hpp>
#include <rtg/miopen/miopen_target.hpp>
#include <rtg/miopen/hip.hpp>
#include <rtg/generate.hpp>
#include <miopen/miopen.h>
#include <rtg/miopen/miopen.hpp>

rtg::argument run_cpu(std::string file)
{
    auto p = rtg::parse_onnx(file);
    p.compile(rtg::cpu::cpu_target{});
    auto s      = p.get_parameter_shape("Input3");
    auto input3 = rtg::generate_argument(s);
    auto out    = p.eval({{"Input3", input3}});
    std::cout << p << std::endl;
    return out;
}

rtg::argument run_gpu(std::string file)
{
    auto p = rtg::parse_onnx(file);
    p.compile(rtg::cpu::cpu_target{});
    auto s      = p.get_parameter_shape("Input3");
    auto input3 = rtg::miopen::to_gpu(rtg::generate_argument(s));

    auto output = rtg::miopen::to_gpu(rtg::generate_argument(p.get_parameter_shape("output")));
    auto handle = rtg::miopen::make_obj<rtg::miopen::miopen_handle>(&miopenCreate);

    auto out = p.eval({{"Input3", input3}, {"output", output}});
    std::cout << p << std::endl;
    return rtg::miopen::from_gpu(out);
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
