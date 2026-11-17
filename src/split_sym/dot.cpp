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
#include <migraphx/split_sym/analyzer.hpp>
#include <migraphx/make_op.hpp>
#include <string>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace split_sym {
namespace {

bool is_dot(const operation& op)
{
    return op.name() == "dot" or op.attributes().get("general_data_type", std::string{}) == "dot";
}

struct analyze_dot : analyzer<analyze_dot>
{
    bool matches(const operation& op) const { return is_dot(op); }

    symbolic_op_info analyze(instruction_ref ins) const
    {
        auto input_shapes = to_shapes(ins->inputs());
        auto result       = analyze_axes(ins, [&](std::size_t input, std::size_t axis) {
            std::size_t rank = input_shapes.at(input).ndim();
            assert(rank >= 2);
            std::size_t contraction_axis = (input == 0) ? rank - 1 : rank - 2;
            return axis == contraction_axis ? masked_axis(mask_role::contracted, fill_kind::zero)
                                            : parallel_axis();
        });
        if(result.supported)
            result.freezer = freeze;
        return result;
    }

    // A body may absorb the dot without its plan, as for a dot of fixed shape whose operands come
    // from another block; zeros padded on the contracted axis leave the products unchanged
    static instruction_ref freeze(module& m,
                                  instruction_ref source,
                                  const std::vector<instruction_ref>& args,
                                  const std::unordered_map<sym::expr, std::size_t>&)
    {
        auto padded_args = args;
        for(auto& arg : padded_args)
            if(arg->get_shape().dynamic())
                arg = m.add_instruction(make_op("fixed_pad", {{"value", 0.0f}}), arg);
        return m.add_instruction(source->get_operator(), padded_args);
    }
};

} // namespace
} // namespace split_sym
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
