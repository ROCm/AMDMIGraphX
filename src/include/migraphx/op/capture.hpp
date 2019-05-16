#ifndef MIGRAPHX_GUARD_OPERATORS_CAPTURE_HPP
#define MIGRAPHX_GUARD_OPERATORS_CAPTURE_HPP

#include <array>
#include <migraphx/operation.hpp>
#include <migraphx/check_shapes.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/streamutils.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/shape_for_each.hpp>
#include <migraphx/config.hpp>
#include <cmath>
#include <utility>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

struct capture
{
    std::function<void(std::vector<argument>)> f;
    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.f, "func"));
    }

    std::string name() const { return "capputure"; }

    shape compute_shape(std::vector<shape> inputs) const { return inputs.front(); }

    argument compute(const shape& output_shape, std::vector<argument> args) const
    {
        argument result{output_shape};
        args.push_back(result);
        f(args);

        return result;
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
