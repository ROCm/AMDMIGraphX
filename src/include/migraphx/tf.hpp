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
};

/// Create a program from a tf pb file (default is nhwc format)
program parse_tf(const std::string& name, tf_options = tf_options{});

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
