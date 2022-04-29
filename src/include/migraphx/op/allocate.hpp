#ifndef MIGRAPHX_GUARD_OPERATORS_ALLOCATE_HPP
#define MIGRAPHX_GUARD_OPERATORS_ALLOCATE_HPP

#include <array>
#include <migraphx/check_shapes.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/streamutils.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/shape_for_each.hpp>
#include <migraphx/config.hpp>
#include <cmath>
#include <utility>
#include <migraphx/shape.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

struct allocate
{
    shape s{};
    std::string tag = "";
    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.s, "shape"), f(self.tag, "tag"));
    }
    std::string name() const { return "allocate"; }
    shape compute_shape(const std::vector<shape>& inputs) const {
         migraphx::check_shapes{inputs, *this}.has(0);
         return s; 
    }
    argument compute(const shape& output_shape, const std::vector<argument>&) const
    {
        return {output_shape};
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
