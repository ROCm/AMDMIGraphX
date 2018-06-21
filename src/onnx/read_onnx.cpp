
#include <rtg/onnx.hpp>

#include <rtg/cpu/cpu_target.hpp>
#include <random>

// TODO: Move this to a seperate header
std::vector<float> get_tensor_data(rtg::shape s)
{
    std::vector<float> result(s.elements());
    std::mt19937 engine{0};
    std::uniform_real_distribution<> dist;
    std::generate(result.begin(), result.end(), [&] { return dist(engine); });
    return result;
}

rtg::argument get_tensor_argument(rtg::shape s)
{
    auto v = get_tensor_data(s);
    return {s, [v]() mutable { return reinterpret_cast<char*>(v.data()); }};
}

int main(int argc, char const* argv[])
{
    if(argc > 1)
    {
        std::string file = argv[1];
        auto prog        = rtg::parse_onnx(file);
        prog.compile(rtg::cpu::cpu_target{});
        auto s      = prog.get_parameter_shape("Input3");
        auto input3 = get_tensor_argument(s);
        auto out    = prog.eval({{"Input3", input3}});
        (void)out;
        std::cout << prog << std::endl;
    }
}
