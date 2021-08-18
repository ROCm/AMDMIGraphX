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
        void copy(context&, const argument& src, T& dst) const
        {
            dst = *src.cast<T>();
        }

        template <class T>
        void copy(context&, T src, const argument& dst) const
        {
            *dst.cast<T>() = src;
        }

        template <class InputIt, class OutputIt>
        void copy_carry_dependencies(context&, InputIt first, InputIt last, OutputIt d_first) const
        {
            std::copy(first, last, d_first);
        }

        void append(const std::vector<argument>& iter_state,
                    const std::vector<argument>& concatenated_outputs,
                    const int iter) const
        {
            assert(iter_state.size() == concatenated_outputs.size());
            for(auto i : range(iter_state.size()))
            {
                const auto& iter_stat = iter_state.at(i);
                const auto& scan_out  = concatenated_outputs.at(i);

                auto* in_data        = iter_stat.data();
                auto* out_data       = scan_out.data();
                std::size_t out_size = iter_stat.get_shape().bytes();
                assert((iter + 1) * out_size <= scan_out.get_shape().bytes());
                std::copy(in_data, in_data + out_size, out_data + iter * out_size);
            }
        }

        void
        set_zero(context&, const std::vector<argument>& concatenated_outputs, const int iter) const
        {
            if(iter >= max_iter_num)
                return;

            for(const auto& out : concatenated_outputs)
            {
                auto s    = out.get_shape();
                auto size = s.bytes() / max_iter_num;
                std::fill(out.data() + iter * size, out.data() + max_iter_num * size, 0);
            }
        }

        template <class T>
        void print_params(const T& params) const
        {
            for(auto na : params)
            {
                std::cout << "ref, name = " << na.first << ", val = " << na.second << std::endl;
            }
        }

        template <class T>
        void print_outputs(const T& outputs) const
        {
            for(auto na : outputs)
            {
                std::cout << "ref, output = " << na << std::endl;
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
        auto s_cond = args.at(1).get_shape();
        auto s_iter = args.at(0).get_shape();
        cpy_args.push_back({s_iter, &iter});
        cpy_args.push_back({s_cond, &cond});
        cpy_args.insert(cpy_args.end(), args.begin() + 2, args.end());
        // add cond and mod outputs to the argument list
        cpy_args.push_back(argument(s_cond));
        cpy_args.push_back(argument(out_shape));
        // run loop
        return run_loop(ref_loop{max_iter_num}, ctx, cpy_args, mods, run);
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
