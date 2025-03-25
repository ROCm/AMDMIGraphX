#ifndef MIGRAPHX_GUARD_INCLUDE_READ_TF_HPP
#define MIGRAPHX_GUARD_INCLUDE_READ_TF_HPP

#include <pb_files.hpp>

inline migraphx::program read_tf(const std::string& name, migraphx::tf_options options = migraphx::tf_options{})
{
    static auto pb_files{::pb_files()};
    if(pb_files.find(name) == pb_files.end())
    {
        std::cerr << "Can not find TensorFlow Protobuf file by name: " << name
                  << " , aborting the program\n"
                  << std::endl;
        std::abort();
    }
    return migraphx::parse_tf_buffer(std::string{pb_files.at(name)}, options);
}

#endif // MIGRAPHX_GUARD_INCLUDE_READ_TF_HPP
