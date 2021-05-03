#ifndef MIGRAPHX_GUARD_OPERATORS_LOOP_HPP
#define MIGRAPHX_GUARD_OPERATORS_LOOP_HPP

#include "migraphx/errors.hpp"
#include <array>
#include <migraphx/check_shapes.hpp>
#include <migraphx/argument.hpp>
#include <migraphx/functional.hpp>
#include <migraphx/config.hpp>
#include <migraphx/module.hpp>
#include <cmath>
#include <utility>
#include <set>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

struct loop
{
    int64_t max_iters = 0;

    std::string name() const { return "loop"; }

    shape compute_shape(const std::vector<shape>& inputs, std::vector<module_ref> mods) const
    {
        check_shapes{inputs, *this}.standard();
        if (mods.size() != 1)
        {
            MIGRAPHX_THROW("LOOP: operator should have one submodule.");
        }

        const auto& mod = mods.front();
        auto mod_out_shapes = mod->get_output_shapes();
        auto param_names = mod->get_parameter_names();
        // remove the first two names -- iter_num and cond_var
        param_names.erase(param_names.begin(), param_names.begin() + 2);
        std::vector<shape> ins_out_shapes;
        for (const auto& name : param_names)
        {
            const auto& s = mod->get_parameter_shape(name);
            if (s == shape{})
            {
                MIGRAPHX_THROW("LOOP: mode shape does not exist for parameter: " + name);
            }
            ins_out_shapes.push_back(s);
        }

        mod_out_shapes.erase(mod_out_shapes.begin(), mod_out_shapes.begin() + ins_out_shapes.size() + 1);
        for (const auto& out_s : mod_out_shapes)
        {
            auto lens = out_s.lens();
            lens.insert(lens.begin(), max_iters);
            ins_out_shapes.push_back({out_s.type(), lens});
        }

        return shape(ins_out_shapes);
    }

    argument compute(const shape&,
                     const std::vector<argument>& args,
                     const std::vector<module_ref>& mods,
                     const std::function<std::vector<argument>(
                         module_ref&, const std::unordered_map<std::string, argument>&)>& run) const
    {
        auto cond      = args.front().at<bool>();
        module_ref mod = cond ? mods[0] : mods[1];
        std::unordered_map<std::string, argument> params;

        std::set<std::string> pnames;
        for(const auto& smod : mods)
        {
            auto names = smod->get_parameter_names();
            pnames.insert(names.begin(), names.end());
        }

        assert(pnames.size() < args.size());
        std::transform(pnames.begin(),
                       pnames.end(),
                       args.begin() + 1,
                       std::inserter(params, params.end()),
                       [](auto&& name, auto&& arg) { return std::make_pair(name, arg); });

        auto results = run(mod, params);
        return argument{results};
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
