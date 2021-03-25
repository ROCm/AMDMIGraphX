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
        check_shapes{inputs, *this}.standard();
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

    argument compute(
        const std::vector<argument>& args,
        const std::vector<module_ref>& mods,
        const std::function<std::vector<argument>(
            module_ref& mdl, const std::unordered_map<std::string, argument>& inputs)>& run) const
    {
        auto cond      = args.front().at<bool>();
        module_ref mod = cond ? mods[0] : mods[1];
        std::unordered_map<std::string, argument> params;
        // const auto& out_shapes = mod->get_output_shapes();
        assert(args.size() == 1 or args.size() == mod->get_output_shapes().size() + 1);
        for(std::size_t i = 1; i < args.size(); ++i)
        {
            std::string name = mod->name() + ":#output_" + std::to_string(i - 1);
            auto&& ps        = mod->get_parameter_shape(name);
            if(ps != shape{})
            {
                params[name] = args.at(i);
            }
        }

        auto results = run(mod, params);
        return results[0];
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
