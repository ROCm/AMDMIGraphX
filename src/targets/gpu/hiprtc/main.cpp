#include <migraphx/gpu/compile_hip.hpp>
#include <migraphx/serialize.hpp>
#include <migraphx/value.hpp>
#include <migraphx/msgpack.hpp>
#include <migraphx/file_buffer.hpp>
#include <iostream>
#include <cstring>

std::vector<char> read_stdin()
{
    std::vector<char> result;

    std::array<char, 1024> buffer;
    std::size_t len = 0;
    while((len = std::fread(buffer.data(), 1, buffer.size(), stdin)) > 0)
    {
        if(std::ferror(stdin) && !std::feof(stdin))
            throw std::runtime_error(std::strerror(errno));

        result.insert(result.end(), buffer.data(), buffer.data() + len);
    }
    return result;
}

int main(int argc, char const* argv[])
{
    if(argc < 2)
        std::abort();
    std::string output_name = argv[1];

    auto v = migraphx::from_msgpack(read_stdin());
    std::vector<migraphx::src_file> srcs;
    migraphx::from_value(v.at("srcs"), srcs);
    auto out = migraphx::gpu::compile_hip_src_with_hiprtc(
        srcs, v.at("params").to<std::string>(), v.at("arch").to<std::string>());
    if(not out.empty())
        migraphx::write_buffer(output_name, out.front());
}
