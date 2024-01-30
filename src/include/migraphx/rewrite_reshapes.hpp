#ifndef MIGRAPHX_GUARD_MIGRAPHX_REWRITE_RESHAPES_HPP
#define MIGRAPHX_GUARD_MIGRAPHX_REWRITE_RESHAPES_HPP

#include <migraphx/config.hpp>
#include <migraphx/matcher.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/common_dims.hpp>
#include <migraphx/simplify_reshapes.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct rewrite_reshapes_base
{
    template <class AxesMap>
    static instruction_ref insert(module_pass_manager& mpm,
                                  instruction_ref ins,
                                  const std::vector<instruction_ref>& inputs,
                                  const AxesMap&)
    {
        return mpm.get_module().insert_instruction(
            ins, ins->get_operator(), inputs, ins->module_inputs());
    }

    template <class AxesMap>
    static bool supports(instruction_ref, std::vector<std::size_t>&, const AxesMap&)
    {
        return true;
    }

    static std::vector<std::size_t> base_dims(instruction_ref ins)
    {
        return ins->get_shape().lens();
    }
};

template <class T>
struct rewrite_reshapes
{
    std::string name() const { return "rewrite_reshapes"; }
    struct find_op_reshape_op
    {
        std::string op1;
        std::string op2;

        auto matcher() const
        {
            auto reshape =
                match::name("reshape", "squeeze", "unsqueeze", "flatten")(match::used_once());
            auto skip_contiguous = [](auto... ms) {
                return match::arg(0)(match::skip(
                    match::name("contiguous", "multibroadcast")(match::used_once()))(ms...));
            };
            auto pointwise         = match::name(op1)(match::used_once());
            auto reshape_pointwise = reshape(skip_contiguous(pointwise.bind("x"))).bind("reshape");
            return match::name(op2)(match::any_of[match::inputs()](reshape_pointwise));
        }

        template <class F>
        static instruction_ref find_input_if(instruction_ref start, instruction_ref last, F f)
        {
            while(start != last)
            {
                if(f(start))
                    return start;
                if(start->inputs().size() != 1)
                    return last;
                start = start->inputs().front();
            }
            return last;
        }

        static bool match_input(instruction_ref ins, instruction_ref x_ins)
        {
            if(ins->inputs().empty())
                return false;
            auto input = ins->inputs().front();
            if(input->name() == "contiguous")
                return match_input(input, x_ins);
            return x_ins == input;
        }

        void apply(module_pass_manager& mpm, const match::matcher_result& r) const
        {
            auto ins         = r.result;
            auto x_ins       = r.instructions["x"];
            auto reshape_ins = r.instructions["reshape"];

            auto broadcast_ins = find_input_if(
                reshape_ins, x_ins, [&](auto i) { return i->name() == "multibroadcast"; });
            const bool has_broadcast = broadcast_ins != x_ins;
            if(has_broadcast and not match_input(broadcast_ins, x_ins))
                return;

            auto dims1 = T::base_dims(ins);
            auto dims2 = T::base_dims(x_ins);

            if(elements(dims1) != elements(dims2))
                return;

            auto cd = common_dims::compute(T::base_dims(ins), T::base_dims(x_ins));
            if(cd.dims.empty())
                return;

            if(ins->name() != "pointwise" and not T::supports(ins, cd.dims, cd.axes_map1))
                return;
            if(x_ins->name() != "pointwise" and not T::supports(x_ins, cd.dims, cd.axes_map2))
                return;

            auto reshape_input = [&](const auto& ins_to_insert) {
                return [&](auto input) {
                    auto dims = cd.get_dimensions_for(ins->get_shape().lens());
                    return mpm.get_module().insert_instruction(
                        ins_to_insert, make_op("reshape", {{"dims", dims}}), input);
                };
            };
            auto x_inputs = x_ins->inputs();
            std::transform(
                x_inputs.begin(), x_inputs.end(), x_inputs.begin(), reshape_input(x_ins));
            auto new_x_ins = insert(mpm, x_ins, x_inputs, cd.axes_map1);
            if(has_broadcast)
            {
                new_x_ins = mpm.get_module().insert_instruction(
                    x_ins, make_op("multibroadcast", {{"out_lens", cd.dims}}), new_x_ins);
            }

            auto inputs = ins->inputs();
            std::transform(inputs.begin(), inputs.end(), inputs.begin(), [&](auto input) {
                if(input == reshape_ins)
                    return new_x_ins;
                return reshape_input(ins)(input);
            });
            auto pw = insert(mpm, ins, inputs, cd.axes_map2);
            mpm.get_module().replace_instruction(
                ins, make_op("reshape", {{"dims", ins->get_shape().lens()}}), pw);
        }

        static bool same_dims(instruction_ref ins)
        {
            return all_of(ins->inputs(), [&](auto input) {
                return input->get_shape().lens() == ins->get_shape().lens();
            });
        }

        template <class AxesMap>
        static instruction_ref insert(module_pass_manager& mpm,
                                      instruction_ref ins,
                                      const std::vector<instruction_ref>& inputs,
                                      const AxesMap& am)
        {
            if(ins->name() == "pointwise")
                return mpm.get_module().insert_instruction(
                    ins, ins->get_operator(), inputs, ins->module_inputs());
            return T::insert(mpm, ins, inputs, am);
        }
    };

    void apply(module_pass_manager& mpm) const
    {
        if(T::name() == "pointwise")
        {
            match::find_matches(mpm, find_op_reshape_op{"pointwise", T::name()});
        }
        else
        {
            match::find_matches(mpm,
                                find_op_reshape_op{"pointwise", T::name()},
                                find_op_reshape_op{T::name(), "pointwise"},
                                find_op_reshape_op{T::name(), T::name()});
        }
        mpm.run_pass(simplify_reshapes{1});
    }
};

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
#endif // MIGRAPHX_GUARD_MIGRAPHX_REWRITE_RESHAPES_HPP
