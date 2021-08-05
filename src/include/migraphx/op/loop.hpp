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
#include <migraphx/run_loop.hpp>
#include <migraphx/ranges.hpp>
#include <cmath>
#include <string>
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

    struct ref_loop
    {
        int64_t max_iter_num = 0;

        template <class T>
        void copy_arg(context&, const argument& arg, T& var, bool from_to_var) const
        {
            argument arg_var{arg.get_shape(), &var};
            if(from_to_var)
            {
                memcpy(arg.data(), &var, arg.get_shape().bytes());
            }
            else
            {
                memcpy(&var, arg.data(), arg.get_shape().bytes());
            }
        }

        void concat_scan_outputs(const std::vector<argument>& mod_scan_outs,
                                 const std::vector<argument>& scan_outputs,
                                 const int iter) const
        {
            assert(mod_scan_outs.size() == scan_outputs.size());
            for(auto i : range(mod_scan_outs.size()))
            {
                const auto& mod_out  = mod_scan_outs.at(i);
                const auto& scan_out = scan_outputs.at(i);

                auto* in_data        = mod_out.data();
                auto* out_data       = scan_out.data();
                std::size_t out_size = mod_out.get_shape().bytes();
                memcpy(out_data + iter * out_size, in_data, out_size);
            }
        }

        void set_zero(const std::vector<argument>& scan_outputs, const int iter) const
        {
            if(iter >= max_iter_num)
                return;

            auto elem_num = max_iter_num - iter;
            for(const auto& out : scan_outputs)
            {
                auto s    = out.get_shape();
                auto size = s.bytes() / max_iter_num;
                memset(out.data() + iter * size, 0, size * elem_num);
            }
        }
    };

    argument compute(context& ctx,
                     const shape& out_shape,
                     const std::vector<argument>& args,
                     const std::vector<module_ref>& mods,
                     const std::function<std::vector<argument>(
                         module_ref&, const std::unordered_map<std::string, argument>&)>& run) const
    {
        // wrap up the arguments vector, so ref and gpu impl are the same
        auto cpy_args = args;
        bool in_cond  = args.at(1).at<bool>();
        bool cond     = in_cond;
        int64_t iter  = 0;
        // insert iter and cond used in the loop
        cpy_args.insert(cpy_args.begin() + 1, {{shape::int64_type}, &iter});
        cpy_args.insert(cpy_args.begin() + 3, {{shape::bool_type}, &cond});
        // add cond and mod outputs to the argument list
        cpy_args.push_back(argument(shape{shape::bool_type}));
        cpy_args.push_back(argument(out_shape));

        // run loop
        return run_loop(ref_loop{max_iter_num}, ctx, cpy_args, mods, run);
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
