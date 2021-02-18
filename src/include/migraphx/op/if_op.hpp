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

    shape compute_shape(std::vector<shape> inputs, std::vector<module_ref> mods) const 
    { 
        auto mod_shapes = mods[0]->get_output_shapes();
        return mod_shapes[0]; 
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
