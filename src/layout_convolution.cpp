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
 */
#include <migraphx/layout_convolution.hpp>
#include <migraphx/module.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/permutation.hpp>
#include <migraphx/functional.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/eliminate_contiguous.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/reshape_dims.hpp>
#include <migraphx/stringutils.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

namespace {
std::vector<int64_t> get_permutation(instruction_ref ins,
                                     const layout_convolution::layout_order& order)
{
    std::vector<int64_t> perm(ins->get_shape().ndim());
    if(order == layout_convolution::channels_last)
    {
        std::iota(perm.begin() + 1, perm.end() - 1, 2);
        perm.back() = 1;
    }
    else
    {
        std::iota(perm.begin(), perm.end(), 0);
    }
    return perm;
}

std::vector<int64_t> get_default_permutation(instruction_ref ins)
{
    std::vector<int64_t> perm(ins->get_shape().ndim());
    std::iota(perm.begin(), perm.end(), 0);
    return perm;
}

bool skip_layout(const shape& s)
{
    return s.ndim() == 1 or s.dynamic() or s.type() == shape::tuple_type;
}

void preserve_output_layout(module& m)
{
    auto last = std::prev(m.end());
    if(last->name() == "@return")
    {
        std::vector<instruction_ref> outputs;
        std::transform(last->inputs().begin(),
                       last->inputs().end(),
                       std::back_inserter(outputs),
                       [&](instruction_ref ins) {
                           if(skip_layout(ins->get_shape()))
                               return ins;
                           auto permutation = find_permutation(ins->get_shape());
                           return m.insert_instruction(
                               last, make_op("layout", {{"permutation", permutation}}), ins);
                       });
        m.replace_return(outputs);
    }
    else if(not skip_layout(last->get_shape()))
    {
        auto permutation = find_permutation(last->get_shape());
        m.add_instruction(make_op("layout", {{"permutation", permutation}}), last);
    }
}

void transform_convolutions(module& m,
                            const layout_convolution::layout_order& order,
                            std::size_t output_channels_last_threshold)
{
    for(auto ins : iterator_for(m))
    {
        if(not contains({"convolution", "quant_convolution"}, ins->name()))
            continue;
        if(ins->get_shape().dynamic())
            continue;
        if(ins->get_shape().lens().size() != 4)
            continue;
        auto v = ins->get_operator().to_value();
        bool is_group_conv = v.at("group").to<int>() > 1;
        auto perm  = is_group_conv ? get_default_permutation(ins) : get_permutation(ins, order);
        auto wperm = perm;
        // With only a few output channels there is nothing to vectorize along K,
        // so keep kyxc where its dense C loads win (e.g. 3-channel RGB heads).
        if(output_channels_last_threshold > 0 and order == layout_convolution::channels_last and
           not is_group_conv and ins->name() == "convolution" and
           ins->inputs().front()->get_shape().type() == shape::float_type and
           ins->inputs().back()->get_shape().lens().front() >= output_channels_last_threshold)
        {
            // Weights [K, C, spatial...] stored spatial-major with the output
            // channel dim K innermost (yxck for 2-D convolutions)
            std::iota(wperm.begin(), wperm.end() - 2, 2);
            *(wperm.end() - 2) = 1;
            wperm.back()       = 0;
        }
        auto args = ins->inputs();
        args.front() =
            m.insert_instruction(ins, make_op("layout", {{"permutation", perm}}), args.front());
        args.back() =
            m.insert_instruction(ins, make_op("layout", {{"permutation", wperm}}), args.back());
        auto conv = m.insert_instruction(ins, ins->get_operator(), args);
        auto c    = m.insert_instruction(ins, make_op("contiguous"), conv);
        m.replace_instruction(ins, c);
    }
}

void remove_layout(module& m)
{
    for(auto ins : iterator_for(m))
    {
        if(ins->name() != "layout")
            continue;
        auto perm  = ins->get_operator().to_value()["permutation"].to_vector<std::int64_t>();
        auto iperm = find_permutation(ins->inputs().front()->get_shape());
        if(perm != iperm)
            continue;
        m.replace_instruction(ins, ins->inputs().front());
    }
}

void apply_layout(module& m,
                  layout_convolution::layout_order order,
                  std::size_t output_channels_last_threshold)
{
    preserve_output_layout(m);
    transform_convolutions(m, order, output_channels_last_threshold);
    run_passes(
        m, {dead_code_elimination{}, eliminate_contiguous{"contiguous"}, dead_code_elimination{}});
    remove_layout(m);
    run_passes(m, {dead_code_elimination{}});
}

std::size_t score(const module& m)
{
    return std::count_if(m.begin(), m.end(), [](const instruction& ins) {
        if(ins.can_eval())
            return false;
        if(contains({"layout", "contiguous"}, ins.name()))
            return true;
        // A reshape whose collapsed dims are not contiguous cannot alias its input as a view
        // (reshape_lazy) and needs a copy, so count it the same as a contiguous.
        if(ins.name() == "reshape")
            return not reshape_dims(ins.inputs().front()->get_shape(),
                                    ins.get_shape().lens(),
                                    {.lazy = true})
                           .has_value();
        return false;
    });
}
} // namespace

void layout_convolution::apply(module_pass_manager& mpm) const
{
    if(order == layout_order::channels_auto)
    {
        // Score each candidate layout on a copy, then transform the live module in
        // place with the cheaper one. A copy is not swapped in because its parameters
        // have fresh identities, which would orphan submodules capturing the originals.
        module m_first = mpm.get_module();
        apply_layout(m_first, channels_first, output_channels_last_threshold);
        module m_last = mpm.get_module();
        apply_layout(m_last, channels_last, output_channels_last_threshold);
        // channels_last converts each parameter to NHWC and back, so allow up to two extra
        // layouts per parameter before preferring channels_first.
        auto allowance = 2 * mpm.get_module().get_parameters().size();
        auto chosen = (score(m_first) + allowance < score(m_last)) ? channels_first : channels_last;
        apply_layout(mpm.get_module(), chosen, output_channels_last_threshold);
    }
    else
    {
        apply_layout(mpm.get_module(), order, output_channels_last_threshold);
    }
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
