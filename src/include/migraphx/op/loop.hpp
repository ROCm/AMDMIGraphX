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
        std::vector<std::string> pnames = mod->get_parameter_names();

        std::vector<shape> vec_out_shapes = out_shape.sub_shapes();
        std::vector<argument> outputs;
        std::transform(vec_out_shapes.begin(),
                       vec_out_shapes.end(),
                       std::back_inserter(outputs),
                       [&](auto& s) { return argument{s}; });

        // dependency carry outputs
        std::vector<argument> dep_outputs(outputs.begin(), outputs.begin() + dep_var_num);
        // scan outputs
        std::vector<argument> scan_outputs(outputs.begin() + dep_var_num, outputs.end());

        // sub graph inputs for each iteration
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
            cond     = mod_args.at(0).at<bool>();

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
            // std::transform(mod_scan_outputs.begin(),
            //                mod_scan_outputs.end(),
            //                scan_outputs.begin(),
            //                scan_outputs.begin(),
            //                [&](auto arg_in, auto arg_out) {
            //                    visit_all(arg_in, arg_out)([&](auto in, auto out) {
            //                        std::copy(in.begin(), in.end(),
            //                                  out.begin());
            //                                 //  out.begin() + static_cast<int>(iter *
            //                                 arg_in.get_shape().elements()));
            //                    });
            //                });
        }
        // remove the cond variable
        mod_args.erase(mod_args.begin());
        outputs = mod_args;
        outputs.insert(outputs.end(), scan_outputs.begin(), scan_outputs.end());

        return argument{outputs};
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
