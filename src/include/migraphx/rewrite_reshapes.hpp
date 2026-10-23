/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 */
#ifndef MIGRAPHX_GUARD_MIGRAPHX_REWRITE_RESHAPES_HPP
#define MIGRAPHX_GUARD_MIGRAPHX_REWRITE_RESHAPES_HPP

#include <migraphx/config.hpp>
#include <migraphx/matcher.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/common_dims.hpp>
#include <migraphx/simplify_reshapes.hpp>
#include <migraphx/eliminate_common_subexpression.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/instruction_traversal.hpp>
#include <migraphx/shape_transform_descriptor.hpp>
#include <migraphx/algorithm.hpp>
#include <migraphx/ranges.hpp>
#include <algorithm>

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

    /// Whether the op can be remapped to the common dims with the axes map
    template <class AxesMap>
    static bool supports(instruction_ref, const std::vector<std::size_t>&, const AxesMap&)
    {
        return true;
    }

    static bool matches(instruction_ref) { return true; }

    static std::vector<std::size_t> base_dims(instruction_ref ins)
    {
        return ins->get_shape().lens();
    }
};

template <class T>
struct rewrite_reshapes
{
    std::string name() const { return "rewrite_reshapes"; }

    void apply(module_pass_manager& mpm) const
    {
        // cppcheck-suppress knownConditionTrueFalse
        if(T::name() == "pointwise")
        {
            match::find_matches(mpm, find_op_reshape_op{"pointwise", T::name()});
        }
        else
        {
            match::find_matches(mpm,
                                find_op_reshape_op{"pointwise", T::name()},
                                find_op_reshape_op{T::name(), "pointwise"},
                                find_op_reshape_op{T::name(), T::name()},
                                find_entry_reshape_op{});
        }
        mpm.run_pass(simplify_reshapes{1});
        mpm.run_pass(eliminate_common_subexpression{});
        mpm.run_pass(dead_code_elimination{});
    }

    private:
    static auto shape_transforms()
    {
        return match::name("reshape",
                           "squeeze",
                           "unsqueeze",
                           "flatten",
                           "transpose",
                           "contiguous",
                           "multibroadcast",
                           "broadcast");
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

    template <class F>
    static bool any_input_of(instruction_ref start, instruction_ref last, F f)
    {
        return find_input_if(start, last, f) != last;
    }

    /// The shape transforms from x_ins to input_ins as a descriptor
    static shape_transform_descriptor chain_descriptor(instruction_ref x_ins,
                                                       instruction_ref input_ins)
    {
        auto path = get_input_path(input_ins);
        auto last = std::find(path.begin(), path.end(), x_ins);
        assert(last != path.end());
        std::vector<operation> ops;
        std::transform(path.begin(), last, std::back_inserter(ops), [](instruction_ref ins) {
            return ins->get_operator();
        });
        std::reverse(ops.begin(), ops.end());
        return shape_transform_descriptor::create(x_ins->get_shape().lens(), ops);
    }

    /// Whether the chain from x_ins to a full-size input of ins merges a
    /// broadcast axis with another, which needs a contiguous copy unless ins
    /// is rebuilt in the common dims of the chain
    static bool
    collapses_into(instruction_ref ins, instruction_ref input_ins, instruction_ref x_ins)
    {
        // Only a reshape merges axes
        if(not any_input_of(input_ins, x_ins, [](instruction_ref i) {
               return contains({"reshape", "flatten"}, i->name());
           }))
            return false;
        // The common dims must apply to the other inputs of ins as well
        if(input_ins->get_shape().lens() != T::base_dims(ins))
            return false;
        auto desc = chain_descriptor(x_ins, input_ins);
        return not desc.empty() and desc.is_collapsing();
    }

    /// Insert the shape transforms mapping input through gdesc before pos
    static instruction_ref insert_transforms(module& m,
                                             instruction_ref pos,
                                             const shape_transform_descriptor& gdesc,
                                             instruction_ref input,
                                             bool no_broadcast = false)
    {
        auto gops = gdesc.generate(input->get_shape().lens(), no_broadcast);
        return std::accumulate(gops.begin(), gops.end(), input, [&](auto start, const auto& op) {
            return m.insert_instruction(pos, op, start);
        });
    }

    template <class AxesMap>
    static bool
    supports(instruction_ref ins, const std::vector<std::size_t>& dims, const AxesMap& am)
    {
        if(ins->name() == "pointwise")
            return true;
        return T::supports(ins, dims, am);
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

    /// Whether ins can be rebuilt in other dims: T allows it, and a tuple
    /// output is only read through get_tuple_elem, where the reshapes go
    static bool rebuildable(instruction_ref ins)
    {
        // cppcheck-suppress knownConditionTrueFalse
        if(not T::matches(ins))
            return false;
        return ins->get_shape().type() != shape::tuple_type or
               all_of(ins->outputs(),
                      [](instruction_ref out) { return out->name() == "get_tuple_elem"; });
    }

    /// Match an op with an input chain of shape transforms from x that
    /// is_viable accepts, binding the chain as "input". A chain that matches
    /// but cant be rewritten would shadow a rewritable chain on another
    /// input, so the inputs are scanned
    template <class M, class F>
    static auto viable_input(M x, F is_viable)
    {
        auto reshapes = shape_transforms()(match::used_once());
        auto chain    = reshapes(match::arg(0)(match::skip(reshapes())(x.bind("x"))));
        return match::make_basic_fun_matcher(
            [=](match::matcher_context& ctx, instruction_ref ins) -> optional<instruction_ref> {
                if(not rebuildable(ins))
                    return nullopt;
                const auto& inputs = ins->inputs();
                auto it = std::find_if(inputs.begin(), inputs.end(), [&](instruction_ref input) {
                    if(not chain.match(ctx, input).has_value())
                        return false;
                    return is_viable(ins, input, ctx.instructions.at("x"));
                });
                if(it == inputs.end())
                    return nullopt;
                ctx.instructions["input"] = *it;
                return ins;
            });
    }

    /// The rebuild of ins in the common dims of the chain ending at input_ins
    struct common_rewrite
    {
        common_rewrite(instruction_ref pins,
                       instruction_ref pinput_ins,
                       shape_transform_descriptor pdesc)
            : ins(pins),
              input_ins(pinput_ins),
              desc(std::move(pdesc)),
              cdims(desc.common_dims()),
              axes_map(desc.common_axes_map_from_dst())
        {
            assert(not desc.empty());
        }

        instruction_ref ins;
        instruction_ref input_ins;
        shape_transform_descriptor desc;
        std::vector<std::size_t> cdims;
        std::vector<std::vector<std::size_t>> axes_map;

        bool supported() const
        {
            // cppcheck-suppress knownConditionTrueFalse
            return supports(ins, cdims, axes_map);
        }

        /// Replace ins by its rebuild reading new_x_ins in place of the chain.
        /// The other inputs are reshaped to the common dims, with the
        /// broadcasts of the chain unless no_broadcast
        void apply(module_pass_manager& mpm, instruction_ref new_x_ins, bool no_broadcast) const
        {
            auto& m = mpm.get_module();
            if(new_x_ins->get_shape().lens() != cdims)
            {
                new_x_ins = m.insert_instruction(
                    ins, make_op("multibroadcast", {{"out_lens", cdims}}), new_x_ins);
            }
            auto to_common = desc.to_common_from_dst();
            auto inputs    = ins->inputs();
            std::transform(inputs.begin(), inputs.end(), inputs.begin(), [&](auto input) {
                if(input == input_ins)
                    return new_x_ins;
                return insert_transforms(m, ins, to_common, input, no_broadcast);
            });
            auto pw     = insert(mpm, ins, inputs, axes_map);
            auto to_dst = desc.to_dst_from_common();
            if(pw->get_shape().type() != shape::tuple_type)
            {
                auto rins = insert_transforms(m, ins, to_dst, pw);
                assert(ins->get_shape().lens() == rins->get_shape().lens());
                m.replace_instruction(ins, rins);
                return;
            }
            // A tuple cant be reshaped, so each element is reshaped where it is read
            auto outputs = ins->outputs();
            std::vector<instruction_ref> elems;
            std::transform(outputs.begin(),
                           outputs.end(),
                           std::back_inserter(elems),
                           [&](instruction_ref out) {
                               auto elem = m.insert_instruction(out, out->get_operator(), pw);
                               return insert_transforms(m, out, to_dst, elem);
                           });
            for_each(outputs.begin(),
                     outputs.end(),
                     elems.begin(),
                     [&](instruction_ref out, instruction_ref elem) {
                         m.replace_instruction(out, elem);
                     });
        }
    };

    /// Rewrite the op T into the common dims of a shape-transform chain whose
    /// source is not a pointwise, T or shape transform (a parameter, literal
    /// or tuple element), so a reshape that merges a broadcast axis of that
    /// source doesnt need a contiguous copy
    struct find_entry_reshape_op
    {
        auto matcher() const
        {
            // A pointwise or T source is rewritten through its inputs by
            // find_op_reshape_op
            auto entry = match::none_of(shape_transforms(), match::name("pointwise", T::name()));
            return match::name(T::name())(viable_input(entry, &find_entry_reshape_op::is_viable));
        }

        static bool is_viable(instruction_ref ins, instruction_ref input_ins, instruction_ref x_ins)
        {
            if(x_ins->get_shape().scalar() or x_ins->get_shape().dynamic())
                return false;
            return collapses_into(ins, input_ins, x_ins);
        }

        void apply(module_pass_manager& mpm, const match::matcher_result& r) const
        {
            auto ins       = r.result;
            auto x_ins     = r.instructions["x"];
            auto input_ins = r.instructions["input"];

            common_rewrite cr{ins, input_ins, chain_descriptor(x_ins, input_ins)};
            if(not cr.supported())
                return;
            // The source is read through views that keep the broadcasts of
            // the chain; the other inputs already hold those axes as data
            auto new_x_ins =
                insert_transforms(mpm.get_module(), ins, cr.desc.to_common_from_src(), x_ins);
            cr.apply(mpm, new_x_ins, true);
        }
    };

    struct find_op_reshape_op
    {
        std::string op1;
        std::string op2;

        auto matcher() const
        {
            return match::name(op2)(
                viable_input(match::name(op1)(match::used_once()), &find_op_reshape_op::is_viable));
        }

        static bool is_viable(instruction_ref ins, instruction_ref input_ins, instruction_ref x_ins)
        {
            // cppcheck-suppress knownConditionTrueFalse
            if(not T::matches(x_ins))
                return false;
            // A fully reduced output has no dimension info left to rebase the
            // reshapes onto, and there is nothing to gain from the rewrite
            if(x_ins->get_shape().elements() == 1)
                return false;
            // If its just a broadcast then skip
            if(not any_input_of(input_ins, x_ins, [](instruction_ref x) {
                   return not contains({"multibroadcast", "broadcast", "contiguous"}, x->name());
               }))
                return false;
            // The chain either broadcasts the output of x back to the size of
            // its inputs, keeps the size of the output, or expands a pointwise
            // into ins through a merged axis. Anything else cant be rebased
            return broadcasts_to_inputs(input_ins, x_ins) or
                   input_ins->get_shape().elements() == x_ins->get_shape().elements() or
                   (expands_into_op(ins, input_ins, x_ins) and
                    collapses_into(ins, input_ins, x_ins));
        }

        /// Whether the chain broadcasts a pointwise x into the op ins. When
        /// it then merges the broadcast axis with another, x is moved into
        /// the common dims and fused into ins there instead of copying the
        /// broadcast with a contiguous
        static bool
        expands_into_op(instruction_ref ins, instruction_ref input_ins, instruction_ref x_ins)
        {
            if(x_ins->name() != "pointwise" or ins->name() != T::name() or
               ins->name() == "pointwise")
                return false;
            return input_ins->get_shape().elements() > x_ins->get_shape().elements();
        }

        /// Whether the chain ending at input_ins broadcasts the output of x
        /// back to the size of its inputs, so the reshapes can be rebased
        /// onto those inputs
        static bool broadcasts_to_inputs(instruction_ref input_ins, instruction_ref x_ins)
        {
            return input_ins->get_shape().elements() != x_ins->get_shape().elements() and
                   input_ins->get_shape().elements() == elements(T::base_dims(x_ins));
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

        static std::optional<bool> is_broadcasted(instruction_ref start, instruction_ref last)
        {
            auto broadcast_ins =
                find_input_if(start, last, [&](auto i) { return i->name() == "multibroadcast"; });
            bool result = broadcast_ins != last;
            if(result and not match_input(broadcast_ins, last))
                return nullopt;
            return result;
        }

        static bool is_broadcast(instruction_ref ins) { return ins->name() == "multibroadcast"; }

        void apply(module_pass_manager& mpm, const match::matcher_result& r) const
        {
            auto ins       = r.result;
            auto x_ins     = r.instructions["x"];
            auto input_ins = r.instructions["input"];

            // A chain that broadcasts the output of x back to the size of its
            // inputs is rebased onto those inputs. Otherwise the descriptor
            // stays on the output of x, and generating the transform for each
            // input rebases it onto that input, so the axes a reduce reduces
            // keep their own lens
            auto dims2  = T::base_dims(x_ins);
            bool rebase = broadcasts_to_inputs(input_ins, x_ins);
            auto desc   = chain_descriptor(x_ins, input_ins);
            if(rebase)
                desc = desc.rebase(dims2);
            if(desc.empty())
                return;

            // A pointwise expanding into ins is moved into the common dims,
            // where the next fusion absorbs it
            const bool expand =
                not rebase and expands_into_op(ins, input_ins, x_ins) and desc.is_collapsing();
            if(not expand and
               desc.elements() != (rebase ? elements(dims2) : x_ins->get_shape().elements()))
                return;

            // cppcheck-suppress knownConditionTrueFalse
            if(not supports(x_ins, desc.common_dims(dims2), desc.common_axes_map_from_src()))
                return;
            common_rewrite cr{ins, input_ins, std::move(desc)};
            if(not cr.supported())
                return;

            auto& m            = mpm.get_module();
            auto to_common_src = cr.desc.to_common_from_src();
            auto x_inputs      = x_ins->inputs();
            std::transform(x_inputs.begin(), x_inputs.end(), x_inputs.begin(), [&](auto input) {
                return insert_transforms(m, x_ins, to_common_src, input);
            });
            auto new_x_ins = insert(mpm, x_ins, x_inputs, cr.desc.common_axes_map_from_src());
            // In the expand case the other inputs already hold the broadcast
            // axes as data, so they are reshaped rather than broadcast
            cr.apply(mpm, new_x_ins, expand);
        }

        static bool same_dims(instruction_ref ins)
        {
            return all_of(ins->inputs(), [&](auto input) {
                return input->get_shape().lens() == ins->get_shape().lens();
            });
        }
    };
};

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
#endif // MIGRAPHX_GUARD_MIGRAPHX_REWRITE_RESHAPES_HPP
