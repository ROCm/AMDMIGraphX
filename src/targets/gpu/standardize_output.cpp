/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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
#include <migraphx/gpu/standardize_output.hpp>
#include <migraphx/gpu/insert_contiguous.hpp>
#include <migraphx/argument.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/module.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/register_op.hpp>
#include <algorithm>
#include <numeric>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

// Groups already-packed tensor leaves back into the tuple shape of a returned op
// such as topk. The leaves are the outputs; this op only rebuilds the tuple.
struct group_tuple
{
    std::string name() const { return "gpu::group_tuple"; }

    shape compute_shape(const std::vector<shape>& inputs) const { return shape{inputs}; }

    argument compute(const shape&, const std::vector<argument>& args) const
    {
        return argument{args};
    }
};
MIGRAPHX_REGISTER_OP(group_tuple);

namespace {

bool needs_contiguous_output(const shape& s)
{
    if(s.dynamic() or s.elements() <= 1)
        return false;
    return not(s.standard() and s.normalize_standard() == s);
}

bool leaf_needs_packing(const shape& s)
{
    if(s.type() == shape::tuple_type)
        return std::any_of(s.sub_shapes().begin(), s.sub_shapes().end(), leaf_needs_packing);
    return needs_contiguous_output(s);
}

instruction_ref pack_value(module& m, instruction_ref pos, instruction_ref input)
{
    const auto& s = input->get_shape();
    if(s.type() != shape::tuple_type)
        return needs_contiguous_output(s) ? insert_contiguous(m, pos, input) : input;
    if(not leaf_needs_packing(s))
        return input;

    std::vector<std::size_t> indexes(s.tuple_size());
    std::iota(indexes.begin(), indexes.end(), 0);
    std::vector<instruction_ref> elems;
    std::transform(indexes.begin(), indexes.end(), std::back_inserter(elems), [&](std::size_t i) {
        auto elem = m.insert_instruction(pos, make_op("get_tuple_elem", {{"index", i}}), input);
        return pack_value(m, pos, elem);
    });
    return m.insert_instruction(pos, make_op("gpu::group_tuple"), elems);
}

} // namespace

void standardize_output::apply(module_pass_manager& mpm) const
{
    auto& m = mpm.get_module();
    // Callers of control-flow submodules already allocated output buffers from the
    // submodule return shapes. Packing those returns changes the strides and the
    // parent then rejects the buffer.
    if(&m != mpm.get_root_module() or m.begin() == m.end())
        return;

    const bool explicit_return = std::prev(m.end())->name() == "@return";
    auto pos                   = m.insert_end();
    auto outputs               = m.get_returns();
    std::vector<instruction_ref> packed;
    std::transform(
        outputs.begin(), outputs.end(), std::back_inserter(packed), [&](instruction_ref output) {
            // hip::copy_from_gpu reports its input shape. Pack the GPU value so the
            // host copy, inserted for offload_copy, receives a packed buffer.
            const bool offload =
                output->name() == "hip::copy_from_gpu" and output->inputs().size() == 1;
            auto produced = offload ? output->inputs().front() : output;
            auto result   = pack_value(m, offload ? output : pos, produced);
            if(offload and result != produced)
                instruction::replace_argument(output, produced, result);
            return offload ? output : result;
        });
    // An implicit return is the last instruction. The packed value was inserted at
    // the end, so adding @return would change the form lowering kept.
    if(explicit_return and packed != outputs)
        m.replace_return(packed);
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
