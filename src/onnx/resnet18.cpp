#include <cstdio>
#include <string>
#include <fstream>
#include <numeric>
#include <stdexcept>

#include <migraph/onnx.hpp>

#include <migraph/cpu/cpu_target.hpp>
#include <migraph/generate.hpp>

int main(int argc, char const* argv[])
{
    std::string file = argv[1];
    auto prog        = migraph::parse_onnx(file);
    prog.compile(migraph::cpu::cpu_target{});
    auto s = migraph::shape{migraph::shape::float_type, {1, 3, 32, 32}};
    auto input3 = migraph::generate_argument(s, 12345);
    auto result = prog.eval({{"0", input3}});
}