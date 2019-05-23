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
    std::size_t ins_index;
    std::function<void(std::size_t ins_index, std::vector<argument>)> f;
    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.ins_index, "instruction_index"));
    }

    std::string name() const { return "capture"; }

    shape compute_shape(std::vector<shape> inputs) const { return inputs.front(); }

    argument compute(const shape&, std::vector<argument> args) const
    {
        f(ins_index, args);
        return args.front();
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
