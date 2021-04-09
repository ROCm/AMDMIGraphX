#ifndef MIGRAPHX_GUARD_MIGRAPHLIB_TF_HPP
#define MIGRAPHX_GUARD_MIGRAPHLIB_TF_HPP

#include <migraphx/program.hpp>
#include <migraphx/config.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

/// struct to pass in tf options to parser
struct tf_options
{
    bool is_nhwc            = false;
    unsigned int batch_size = 1;
    /// Explicitly specify the dims of an input
    std::unordered_map<std::string, std::vector<std::size_t>> map_input_dims = {};
    std::vector<std::string> output_node_names                               = {};
};

/// Create a program from a tf pb file (default is nhwc format)
program parse_tf(const std::string& name, const tf_options& options = tf_options{});

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
