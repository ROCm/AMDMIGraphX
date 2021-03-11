#ifndef MIGRAPHX_GUARD_OPERATORS_IF_OP_HPP
#define MIGRAPHX_GUARD_OPERATORS_IF_OP_HPP

#include <array>
#include <migraphx/check_shapes.hpp>
#include <migraphx/argument.hpp>
#include <migraphx/functional.hpp>
#include <migraphx/config.hpp>
#include <migraphx/module.hpp>
#include <cmath>
#include <utility>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

struct if_op
{
    std::string name() const { return "if"; }

    shape compute_shape(const std::vector<shape>& inputs, std::vector<module_ref> mods) const
    {
        check_shapes{inputs, *this}.has(1).standard();
        if(mods.size() != 2)
        {
            MIGRAPHX_THROW("IF: operator should have two submodules.");
        }

        auto out_shapes0 = mods[0]->get_output_shapes();
        auto out_shapes1 = mods[1]->get_output_shapes();
        if(not std::equal(
               out_shapes1.begin(), out_shapes1.end(), out_shapes0.begin(), out_shapes0.end()))
        {
            MIGRAPHX_THROW("IF: output shapes of submodules must be the same.");
        }

        return out_shapes0.front();
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
