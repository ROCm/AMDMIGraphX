#ifndef MIGRAPHX_GUARD_OPERATORS_LOOP_HPP
#define MIGRAPHX_GUARD_OPERATORS_LOOP_HPP

#include "migraphx/errors.hpp"
#include "migraphx/raw_data.hpp"
#include <array>
#include <iterator>
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
        auto dep_param_num  = inputs.size() - 2;
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

    argument compute(const shape& out_shape,
                     const std::vector<argument>& args,
                     const std::vector<module_ref>& mods,
                     const std::function<std::vector<argument>(
                         module_ref&, const std::unordered_map<std::string, argument>&)>& run) const
    {
        auto iter_num                   = args.at(0).at<int64_t>();
        auto cond                       = args.at(1).at<bool>();
        std::size_t dep_var_num         = args.size() - 2;
        module_ref mod                  = mods.at(0);
        auto mod_name                   = mod->name();
        std::vector<std::string> pnames = mod->get_parameter_names();

        std::string prefix = "@mgx_" + mod->name();
        std::vector<std::pair<std::string, bool>> fixed_input_pair;
        auto it = std::find_if(pnames.begin(), pnames.end(), [&](auto name) {
            return contains(name, prefix + "_iter_");
        });
        if(it != pnames.end())
        {
            fixed_input_pair.push_back({*it, true});
            pnames.erase(it);
        }
        else
        {
            fixed_input_pair.push_back({{}, false});
        }

        it = std::find_if(pnames.begin(), pnames.end(), [&](auto name) {
            return contains(name, prefix + "_cond_");
        });
        if(it != pnames.end())
        {
            fixed_input_pair.push_back({*it, true});
            pnames.erase(it);
        }
        else
        {
            fixed_input_pair.push_back({{}, false});
        }

        std::vector<shape> vec_out_shapes = out_shape.sub_shapes();
        std::vector<argument> scan_outputs;
        std::transform(vec_out_shapes.begin() + 1 + dep_var_num,
                       vec_out_shapes.end(),
                       std::back_inserter(scan_outputs),
                       [&](auto& s) { return argument{s}; });

        // sub graph inputs for each iteration
        std::vector<argument> mod_args(args.begin() + 1, args.end());
        shape s_iter{shape::int64_type};
        for(int64_t iter = 0; (iter < iter_num) and cond; ++iter)
        {
            std::unordered_map<std::string, argument> params;

            // iter index
            if(fixed_input_pair.at(0).second)
            {
                params[fixed_input_pair.at(0).first] = argument(s_iter, &iter);
            }

            // cond variable
            if(fixed_input_pair.at(1).second)
            {
                params[fixed_input_pair.at(1).first] = mod_args.at(0);
            }

            // carry dependencies
            std::transform(pnames.begin(),
                           pnames.end(),
                           mod_args.begin() + 1,
                           std::inserter(params, params.end()),
                           [](auto&& name, auto&& arg) { return std::make_pair(name, arg); });

            mod_args = run(mod, params);
            std::cout << "mod_output:" << std::endl;
            for(const auto& arg : mod_args)
            {
                std::cout << "\targ = " << arg << std::endl;
            }

            cond = mod_args.at(0).at<bool>();
            // concat scan outputs
            std::vector<argument> mod_scan_outputs(mod_args.begin() + 1 + dep_var_num,
                                                   mod_args.end());
            for(std::size_t i = 0; i < mod_scan_outputs.size(); ++i)
            {
                auto& mod_out  = mod_scan_outputs.at(i);
                auto& scan_out = scan_outputs.at(i);

                auto in_data         = mod_out.data();
                auto out_data        = scan_out.data();
                std::size_t out_size = mod_out.get_shape().bytes();
                memcpy(out_data + iter * out_size, in_data, out_size);
            }
        }

        // copy dependency carry output to final output
        std::vector<argument> outputs(mod_args.begin() + 1, mod_args.begin() + 1 + dep_var_num);
        outputs.insert(outputs.end(), scan_outputs.begin(), scan_outputs.end());

        std::cout << "loop output = " << std::endl;
        for(auto& out : outputs)
        {
            std::cout << "\tins_out = " << out << std::endl;
        }

        return argument{outputs};
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
