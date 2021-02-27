#include <migraphx/tf/tf_parser.hpp>
#include <iostream>
#include <fstream>
#include <unordered_map>
#include <functional>
#include <array>
#include <utility>
#include <vector>

#include <migraphx/program.hpp>
#include <migraphx/tf.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

program parse_tf(const std::string& name, const tf_options& options)
{
    std::fstream input(name.c_str(), std::ios::in | std::ios::binary);
    tf::tf_parser parser;
    parser.is_nhwc           = options.is_nhwc;
    parser.batch_size        = options.batch_size;
    parser.map_input_dims    = options.map_input_dims;
    parser.output_node_names = options.output_node_names;

#ifndef NDEBUG
    // Log the program when it can't be parsed
    try
    {
        parser.parse_from(input);
    }
    catch(...)
    {
        std::cerr << parser.prog << std::endl;
        throw;
    }
#else
    parser.parse_from(input);
#endif
    return std::move(parser.prog);
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
