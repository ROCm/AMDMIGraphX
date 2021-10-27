#include <migraphx/gpu/driver/parser.hpp>
#include <migraphx/json.hpp>
#include <migraphx/convert_to_json.hpp>
#include <migraphx/file_buffer.hpp>
#include <iostream>

using namespace migraphx;              // NOLINT
using namespace migraphx::gpu;         // NOLINT
using namespace migraphx::gpu::driver; // NOLINT

int main(int argc, char const* argv[])
{
    std::vector<std::string> args(argv, argv + argc);
    if(args.size() < 2)
    {
        std::cout << "Usage: gpu-driver <input-file>" << std::endl;
        std::abort();
    }
    auto v = from_json_string(convert_to_json(read_string(args[1])));
    parser::process(v);
}
