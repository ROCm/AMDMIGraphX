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
    std::string name() const { return "if_op"; }

    shape compute_shape(std::vector<shape> inputs) const 
    { 
        return inputs[0];
    }

    shape compute_shape(std::vector<shape>, std::vector<module_ref> mods) const 
    { 
        auto out_shapes = mods[0]->get_output_shapes();
        return out_shapes[0]; 
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
