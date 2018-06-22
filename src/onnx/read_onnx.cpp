
#include <rtg/onnx.hpp>

#include <rtg/cpu/cpu_target.hpp>
#include <rtg/generate.hpp>

int main(int argc, char const* argv[])
{
    if(argc > 1)
    {
        std::string file = argv[1];
        auto prog        = rtg::parse_onnx(file);
        prog.compile(rtg::cpu::cpu_target{});
        auto s      = prog.get_parameter_shape("Input3");
        auto input3 = generate_argument(s);
        auto out    = prog.eval({{"Input3", input3}});
        (void)out;
        std::cout << prog << std::endl;
    }
}
