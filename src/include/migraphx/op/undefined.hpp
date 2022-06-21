#ifndef MIGRAPHX_GUARD_RTGLIB_UNDEFINED_HPP
#define MIGRAPHX_GUARD_RTGLIB_UNDEFINED_HPP

#include <migraphx/config.hpp>
#include <migraphx/argument.hpp>
#include <migraphx/check_shapes.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

struct undefined
{
    std::string name() const { return "undefined"; }
    shape compute_shape(const std::vector<shape>& inputs) const
    {
        check_shapes{inputs, *this}.has(0);
        return {};
    }

    argument compute(const shape&, const std::vector<argument>&) const { return {{}, nullptr}; }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
