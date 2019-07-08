#ifndef MIGRAPHX_GUARD_OPERATORS_LOGSOFTMAX_HPP
#define MIGRAPHX_GUARD_OPERATORS_LOGSOFTMAX_HPP

#include <migraphx/operation.hpp>
#include <migraphx/check_shapes.hpp>
#include <migraphx/config.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

struct logsoftmax
{
    int axis = 1;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.axis, "axis"));
    }

    std::string name() const { return "logsoftmax"; }
    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs}.has(1).standard();
        if(axis < 0 || axis >= inputs[0].lens().size())
        {
            MIGRAPHX_THROW("LogSoftMax: input axis value " + std::to_string(axis) +
                           " is out of range");
        }
        return inputs.at(0);
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
