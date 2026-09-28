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

#include <migraphx/common.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/op/builder/op_builder.hpp>
#include <migraphx/op/builder/insert.hpp>

#include <numeric>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {
namespace builder {

struct rotary_embedding : op_builder<rotary_embedding>
{
    bool interleaved = false;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.interleaved, "interleaved"));
    }

    // {input, pos_ids, cos_cache, sin_cache} — raw caches, builder gathers internally
    std::vector<instruction_ref>
    insert(module& m, instruction_ref ins, const std::vector<instruction_ref>& args) const
    {
        auto in        = args[0];
        auto pos_ids   = args[1];
        auto cos_cache = args[2];
        auto sin_cache = args[3];

        auto [cos, sin] =
            in->get_shape().symbolic()
                ? gather_symbolic_cache(m, ins, in, pos_ids, cos_cache, sin_cache)
                : gather_cache(m, ins, in, pos_ids, cos_cache, sin_cache);
        return apply_rotation(m, ins, in, cos, sin);
    }

    static std::size_t fixed_dimension(const shape& s, std::size_t axis, const std::string& name)
    {
        if(not s.dynamic())
            return s.lens().at(axis);
        auto value = sym::fixed_value(s.dyn_dims().at(axis).sym_expr);
        if(not value.has_value())
            MIGRAPHX_THROW("rotary_embedding: " + name + " must be fixed");
        return sym::to<std::size_t>(*value);
    }

    static instruction_ref
    insert_runtime_reshape(module& m,
                           instruction_ref ins,
                           instruction_ref input,
                           const std::vector<shape::dynamic_dimension>& dims,
                           const std::vector<instruction_ref>& shape_sources)
    {
        std::vector<sym::expr> expressions(dims.size());
        transform(dims, expressions.begin(), [](const auto& dim) { return dim.sym_expr; });
        auto resolved_dims = m.insert_instruction(
            ins,
            make_op("eval_expr_from_shape", {{"expressions", to_value(expressions)}}),
            shape_sources);
        auto allocation = m.insert_instruction(
            ins,
            make_op("allocate", {{"shape", to_value(shape{input->get_shape().type(), dims})}}),
            resolved_dims);
        return m.insert_instruction(ins, make_op("reshape"), input, allocation);
    }

    std::pair<instruction_ref, instruction_ref>
    gather_symbolic_cache(module& m,
                          instruction_ref ins,
                          instruction_ref in,
                          instruction_ref pos_ids,
                          instruction_ref cos_cache,
                          instruction_ref sin_cache) const
    {
        const auto& input_shape = in->get_shape();
        if(input_shape.ndim() != 4)
            MIGRAPHX_THROW("rotary_embedding: expected input of rank 4 with layout "
                           "[batch, heads, seq, head_size] in 4-arg mode");
        if(interleaved)
            MIGRAPHX_THROW("rotary_embedding: interleaved symbolic input is not supported");

        const auto& input_dims = input_shape.dyn_dims();
        auto head_size         = fixed_dimension(input_shape, 3, "head_size");
        if(head_size % 2 != 0)
            MIGRAPHX_THROW("rotary_embedding: head_size must be even");
        auto half_head = head_size / 2;
        if(cos_cache->get_shape().ndim() != 2 or
           fixed_dimension(cos_cache->get_shape(), 1, "cos_cache width") != half_head)
            MIGRAPHX_THROW(
                "rotary_embedding: cos_cache last dimension must equal head_size/2");
        if(sin_cache->get_shape().ndim() != 2 or
           fixed_dimension(sin_cache->get_shape(), 1, "sin_cache width") != half_head)
            MIGRAPHX_THROW(
                "rotary_embedding: sin_cache last dimension must equal head_size/2");

        auto position_dims = pos_ids->get_shape().to_symbolic().dyn_dims();
        if(position_dims.size() != 2 or
           not sym::same_symbol(position_dims.at(0).sym_expr, input_dims.at(0).sym_expr) or
           not sym::same_symbol(position_dims.at(1).sym_expr, input_dims.at(2).sym_expr))
            MIGRAPHX_THROW(
                "rotary_embedding: position_ids must match input batch and sequence");

        std::vector<shape::dynamic_dimension> index_dims{
            input_dims.at(0), input_dims.at(2), shape::dynamic_dimension{sym::lit(1)}};
        auto indices = insert_runtime_reshape(m, ins, pos_ids, index_dims, {pos_ids});
        auto cos     = m.insert_instruction(ins, make_op("gathernd"), cos_cache, indices);
        auto sin     = m.insert_instruction(ins, make_op("gathernd"), sin_cache, indices);
        cos = m.insert_instruction(ins, make_op("concat", {{"axis", -1}}), cos, cos);
        sin = m.insert_instruction(ins, make_op("concat", {{"axis", -1}}), sin, sin);

        std::vector<shape::dynamic_dimension> cache_dims{input_dims.at(0),
                                                        shape::dynamic_dimension{sym::lit(1)},
                                                        input_dims.at(2),
                                                        shape::dynamic_dimension{
                                                            sym::lit(head_size)}};
        cos = insert_runtime_reshape(m, ins, cos, cache_dims, {in});
        sin = insert_runtime_reshape(m, ins, sin, cache_dims, {in});
        return {cos, sin};
    }

    std::pair<instruction_ref, instruction_ref> gather_cache(module& m,
                                                             instruction_ref ins,
                                                             instruction_ref in,
                                                             instruction_ref pos_ids,
                                                             instruction_ref cos_cache,
                                                             instruction_ref sin_cache) const
    {
        auto in_lens = in->get_shape().lens();
        // Expect input layout: [batch, heads, seq, head_size]
        if(in_lens.size() != 4)
        {
            MIGRAPHX_THROW("rotary_embedding: expected input of rank 4 with layout "
                           "[batch, heads, seq, head_size] in 4-arg mode");
        }

        auto batch     = in_lens[0];
        auto seq_len   = in_lens[2];
        auto head_size = in_lens[3];

        if(head_size % 2 != 0)
        {
            MIGRAPHX_THROW(
                "rotary_embedding: head_size must be even so that head_size/2 can be used for "
                "rotary embedding");
        }

        auto half_head = head_size / 2;

        // Basic compatibility check: cosine/sine caches must have last dimension == half_head
        auto cos_lens = cos_cache->get_shape().lens();
        auto sin_lens = sin_cache->get_shape().lens();
        if(cos_lens.empty() or cos_lens.back() != half_head)
        {
            MIGRAPHX_THROW(
                "rotary_embedding: cos_cache last dimension must equal head_size/2 to be "
                "compatible with input");
        }
        if(sin_lens.empty() or sin_lens.back() != half_head)
        {
            MIGRAPHX_THROW(
                "rotary_embedding: sin_cache last dimension must equal head_size/2 to be "
                "compatible with input");
        }
        auto pos_elems = pos_ids->get_shape().elements();
        instruction_ref indices;

        if(pos_elems == batch * seq_len and seq_len > 1)
        {
            indices = m.insert_instruction(
                ins, make_op("reshape", {{"dims", {batch, seq_len, 1}}}), pos_ids);
        }
        else
        {
            instruction_ref pos;
            if(pos_elems == 1 and batch > 1)
            {
                pos = m.insert_instruction(ins, make_op("reshape", {{"dims", {1, 1, 1}}}), pos_ids);
                pos = m.insert_instruction(
                    ins, make_op("multibroadcast", {{"out_lens", {batch, 1, 1}}}), pos);
            }
            else
            {
                pos = m.insert_instruction(
                    ins, make_op("reshape", {{"dims", {batch, 1, 1}}}), pos_ids);
            }

            if(seq_len > 1)
            {
                pos = m.insert_instruction(
                    ins, make_op("multibroadcast", {{"out_lens", {batch, seq_len, 1}}}), pos);
                std::vector<int> range_vec(seq_len);
                std::iota(range_vec.begin(), range_vec.end(), 0);
                auto range_lit = m.add_literal(migraphx::literal{
                    migraphx::shape{pos_ids->get_shape().type(), {1, seq_len, 1}}, range_vec});
                auto range_bc  = m.insert_instruction(
                    ins, make_op("multibroadcast", {{"out_lens", {batch, seq_len, 1}}}), range_lit);
                indices = insert_common_op(m, ins, make_op("add"), {pos, range_bc});
            }
            else
            {
                indices = pos;
            }
        }

        instruction_ref cos_gathered;
        instruction_ref sin_gathered;
        cos_gathered =
            m.insert_instruction(ins, make_op("gathernd", {{"batch_dims", 0}}), cos_cache, indices);
        sin_gathered =
            m.insert_instruction(ins, make_op("gathernd", {{"batch_dims", 0}}), sin_cache, indices);

        if(interleaved)
        {
            auto cos_elems = cos_gathered->get_shape().elements();
            auto sin_elems = sin_gathered->get_shape().elements();
            cos_gathered   = m.insert_instruction(
                ins, make_op("reshape", {{"dims", {cos_elems, 1}}}), cos_gathered);
            sin_gathered = m.insert_instruction(
                ins, make_op("reshape", {{"dims", {sin_elems, 1}}}), sin_gathered);
        }

        auto cos_doubled = m.insert_instruction(
            ins, make_op("concat", {{"axis", -1}}), cos_gathered, cos_gathered);
        auto sin_doubled = m.insert_instruction(
            ins, make_op("concat", {{"axis", -1}}), sin_gathered, sin_gathered);

        auto cos_rs = m.insert_instruction(
            ins, make_op("reshape", {{"dims", {batch, 1, seq_len, head_size}}}), cos_doubled);
        auto sin_rs = m.insert_instruction(
            ins, make_op("reshape", {{"dims", {batch, 1, seq_len, head_size}}}), sin_doubled);

        return {cos_rs, sin_rs};
    }

    std::vector<instruction_ref> apply_rotation(module& m,
                                                instruction_ref ins,
                                                instruction_ref in,
                                                instruction_ref cos,
                                                instruction_ref sin) const
    {
        const auto& input_shape = in->get_shape();
        auto d = fixed_dimension(input_shape, input_shape.ndim() - 1, "head_size");
        auto half_d  = d / 2;
        auto dtype   = input_shape.type();
        assert((d % 2) == 0);
        auto signs = m.add_literal(migraphx::literal{migraphx::shape{dtype, {2}}, {-1.0f, 1.0f}});

        instruction_ref rotated;

        if(interleaved)
        {
            auto in_lens = input_shape.lens();
            signs = m.insert_instruction(ins, make_op("reshape", {{"dims", {1, 2}}}), signs);
            signs = m.insert_instruction(
                ins, make_op("multibroadcast", {{"out_lens", {half_d, 2}}}), signs);
            signs = m.insert_instruction(ins, make_op("reshape", {{"dims", {d}}}), signs);

            auto n     = in->get_shape().elements() / 2;
            auto rs_in = m.insert_instruction(ins, make_op("reshape", {{"dims", {n, 2}}}), in);
            auto evens = m.insert_instruction(
                ins, make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {1}}}), rs_in);
            auto odds = m.insert_instruction(
                ins, make_op("slice", {{"axes", {1}}, {"starts", {1}}, {"ends", {2}}}), rs_in);
            auto swapped =
                m.insert_instruction(ins, make_op("concat", {{"axis", -1}}), odds, evens);
            rotated = m.insert_instruction(ins, make_op("reshape", {{"dims", in_lens}}), swapped);
        }
        else
        {
            signs = m.insert_instruction(ins, make_op("reshape", {{"dims", {2, 1}}}), signs);
            signs = m.insert_instruction(
                ins, make_op("multibroadcast", {{"out_lens", {2, half_d}}}), signs);
            signs = m.insert_instruction(ins, make_op("reshape", {{"dims", {d}}}), signs);

            auto first_half = m.insert_instruction(
                ins, make_op("slice", {{"axes", {-1}}, {"starts", {0}}, {"ends", {half_d}}}), in);
            auto second_half = m.insert_instruction(
                ins, make_op("slice", {{"axes", {-1}}, {"starts", {half_d}}, {"ends", {d}}}), in);
            rotated = m.insert_instruction(
                ins, make_op("concat", {{"axis", -1}}), second_half, first_half);
        }

        if(input_shape.symbolic())
        {
            signs = m.insert_instruction(
                ins,
                make_op("multibroadcast",
                        {{"out_dyn_dims", to_value(input_shape.dyn_dims())}}),
                signs,
                in);
        }
        else
        {
            signs = m.insert_instruction(
                ins, make_op("multibroadcast", {{"out_lens", input_shape.lens()}}), signs);
        }

        auto mul_cos = insert_common_op(m, ins, make_op("mul"), {in, cos});
        auto mul_sin = insert_common_op(m, ins, make_op("mul"), {signs, sin});
        mul_sin      = insert_common_op(m, ins, make_op("mul"), {rotated, mul_sin});
        return {insert_common_op(m, ins, make_op("add"), {mul_cos, mul_sin})};
    }
};

} // namespace builder
} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
