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
#include <migraphx/rewrite_broadcast.hpp>
#include <migraphx/module.hpp>
#include <migraphx/matcher.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/dead_code_elimination.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

namespace {

/**
 * Finds broadcast axes by comparing input shape to output shape.
 * A broadcast axis is where input dimension is 1 and output dimension is > 1.
 */
std::vector<std::size_t> get_broadcast_axes(const shape& input_shape, const shape& output_shape)
{
    std::vector<std::size_t> axes;
    const auto& in_lens  = input_shape.lens();
    const auto& out_lens = output_shape.lens();

    // case where input has fewer dimensions (implicit leading 1s)
    std::size_t offset = out_lens.size() - in_lens.size();
    for(std::size_t i = 0; i < offset; ++i)
    {
        if(out_lens[i] > 1)
            axes.push_back(i);
    }

    // Check remaining dimensions
    for(std::size_t i = 0; i < in_lens.size(); ++i)
    {
        if(in_lens[i] == 1 and out_lens[i + offset] > 1)
            axes.push_back(i + offset);
    }
    return axes;
}

/**
 * Check if two sets of axes are disjoint
 */
bool axes_disjoint(const std::vector<std::size_t>& axes1,
                   const std::vector<int64_t>& axes2)
{
    for(auto a1 : axes1)
    {
        for(auto a2 : axes2)
        {
            if(static_cast<int64_t>(a1) == a2)
                return false;
        }
    }
    return true;
}

/*
 * Compute output shape after reduction on the pre-broadcast tensor
 */
std::vector<std::size_t> compute_reduced_shape(const std::vector<std::size_t>& lens,
                                                const std::vector<int64_t>& axes)
{
    auto result = lens;
    for(auto axis : axes)
    {
        result[axis] = 1;
    }
    return result;
}

/*
 * matches multibroadcast -> reduce
 * rewrites to reduce -> multibroadcast
 * so that the smaller tensor is reduced instead of the broadcasted tensor
*/
struct find_broadcast_reduce
{
    auto matcher() const
    {
        auto broadcast = match::name("multibroadcast")().bind("broadcast");
        return match::name("reduce_sum", "reduce_max", "reduce_min", "reduce_prod")(
            match::any_of[match::inputs()](broadcast))
            .bind("reduce");
    }

    void apply(module& m, const match::matcher_result& r) const
    {
        auto broadcast_ins = r.instructions["broadcast"];
        auto reduce_ins    = r.instructions["reduce"];

        // get the input to the broadcast (the smaller tensor)
        auto input       = broadcast_ins->inputs().front();
        auto input_shape = input->get_shape();

        // get broadcast and reduce axes
        auto broadcast_axes =
            get_broadcast_axes(input_shape, broadcast_ins->get_shape());
        auto reduce_axes =
            reduce_ins->get_operator().to_value()["axes"].to_vector<int64_t>();

        // only optimize if axes are disjoint
        if(not axes_disjoint(broadcast_axes, reduce_axes))
            return;

        // compute the shape after reducing the original input
        auto reduced_lens = compute_reduced_shape(input_shape.lens(), reduce_axes);

        // insert reduce on the original small input
        auto new_reduce =
            m.insert_instruction(reduce_ins, reduce_ins->get_operator(), input);

        // broadcast the reduced result to match expected output shape
        auto new_broadcast = make_op("multibroadcast",
                                     {{"out_lens", reduce_ins->get_shape().lens()}});

        m.replace_instruction(reduce_ins, new_broadcast, new_reduce);
    }
};

/*
 * matches multibroadcast -> convert
 * rewrites to convert -> multibroadcast
 * so that the smaller tensor is converted instead of the broadcasted tensor
*/
struct find_broadcast_convert
{
    auto matcher() const
    {
        auto broadcast =
            match::name("multibroadcast")(match::used_once()).bind("broadcast");
        return match::name("convert")(match::arg(0)(broadcast)).bind("convert");
    }

    void apply(module& m, const match::matcher_result& r) const
    {
        auto broadcast_ins = r.instructions["broadcast"];
        auto convert_ins   = r.instructions["convert"];

        auto input = broadcast_ins->inputs().front();
        auto new_convert =
            m.insert_instruction(convert_ins, convert_ins->get_operator(), input);
        auto new_broadcast = make_op("multibroadcast",
                                     {{"out_lens", convert_ins->get_shape().lens()}});
        m.replace_instruction(convert_ins, new_broadcast, new_convert);
    }
};

} // namespace

void rewrite_broadcast::apply(module& m) const
{
    std::cout << "=== rewrite_broadcast::apply ===" << std::endl;
    std::cout << "module before rewrite_broadcast:\n";
    m.debug_print();

    match::find_matches(m, find_broadcast_convert{}, find_broadcast_reduce{});
    dead_code_elimination{}.apply(m);

    std::cout << "module after rewrite_broadcast:\n";
    m.debug_print();
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

