#ifndef MIGRAPHX_GUARD_INCLUDE_READ_ONNX_HPP
#define MIGRAPHX_GUARD_INCLUDE_READ_ONNX_HPP

#include <onnx_files.hpp>
#include <migraphx/migraphx.hpp>

inline migraphx::program read_onnx(const std::string& name,
                                   migraphx::onnx_options options = migraphx::onnx_options{})
{
    static auto onnx_files{::onnx_files()};
    if(onnx_files.find(name) == onnx_files.end())
    {
        std::cerr << "Can not find onnx file file by name: " << name
                  << " , aborting the program\n"
                  << std::endl;
        std::abort();
    }
    auto prog = migraphx::parse_onnx_buffer(std::string{onnx_files.at(name)}, options);
    return prog;
}

#endif // MIGRAPHX_GUARD_INCLUDE_READ_ONNX_HPP
