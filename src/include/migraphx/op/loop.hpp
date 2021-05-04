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
    int64_t max_iter_num = 0;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.max_iter_num, "max_iter_num"));
    }

    std::string name() const { return "loop"; }

    shape compute_shape(const std::vector<shape>& inputs, std::vector<module_ref> mods) const
    {
        check_shapes{inputs, *this}.standard();
        if(mods.size() != 1)
        {
            MIGRAPHX_THROW("LOOP: operator should have one submodule.");
        }

        const auto& mod     = mods.front();
        auto mod_out_shapes = mod->get_output_shapes();
        auto dep_param_num  = mod->get_parameter_names().size() - 2;
        // first two names -- iter_num and cond_var -- are not counted
        mod_out_shapes.erase(mod_out_shapes.begin());
        std::vector<shape> ins_out_shapes(mod_out_shapes.begin(),
                                          mod_out_shapes.begin() + dep_param_num);
        mod_out_shapes.erase(mod_out_shapes.begin(), mod_out_shapes.begin() + dep_param_num);
        for(const auto& out_s : mod_out_shapes)
        {
            auto lens = out_s.lens();
            lens.insert(lens.begin(), max_iter_num);
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
        auto iter_num  = args.at(0).at<int64_t>();
        auto cond      = args.at(1).at<bool>();
        module_ref mod = mods.at(0);
        std::vector<std::string> pnames = mod->get_parameter_names();
        std::size_t dep_var_num = pnames.size() - 1;

        std::vector<argument> scan_outputs(dep_var_num);
        std::vector<argument> mod_args(args.begin() + 1, args.end());
        shape s_iter{shape::int64_type};
        for(int64_t iter = 0; (iter < iter_num) and cond; ++iter)
        {
            std::unordered_map<std::string, argument> params;
            // iter index
            params[pnames.at(0)] = argument(s_iter, &iter);
            // cond variable
            params[pnames.at(1)] = mod_args.at(0);

            // carry dependencies
            std::transform(pnames.begin() + 2,
                           pnames.end(),
                           mod_args.begin() + 1,
                           std::inserter(params, params.end()),
                           [](auto&& name, auto&& arg) { return std::make_pair(name, arg); });

            mod_args = run(mod, params);
        }
        // remove the cond variable
        mod_args.erase(mod_args.begin());

        return argument{mod_args};
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
