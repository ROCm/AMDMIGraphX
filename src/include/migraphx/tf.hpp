#ifndef MIGRAPHX_GUARD_MIGRAPHLIB_TF_HPP
#define MIGRAPHX_GUARD_MIGRAPHLIB_TF_HPP

#include <migraphx/program.hpp>
#include <migraphx/config.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct unknown
{
    std::string op;
    std::string name() const { return "unknown:" + op; }
    shape compute_shape(std::vector<shape> input) const
    {
        if(input.empty())
            return {};
        else
            return input.front();
    }
    friend std::ostream& operator<<(std::ostream& os, const unknown& x)
    {
        os << x.name();
        return os;
    }
};

/// Create a program from an onnx file
program parse_tf(const std::string& name, bool is_nhwc);

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
