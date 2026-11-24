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
#include <migraphx/runtime_shape.hpp>

#include <numeric>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {
namespace builder {

struct rotary_embedding : op_builder<rotary_embedding>
{
    bool interleaved = false;

    enum class position_mode
    {
        explicit_positions,
        offset_with_range,
        single_token
    };

    struct gather_context
    {
        bool symbolic = false;
        std::size_t batch{};
        std::size_t sequence{};
        std::size_t head_size{};
        std::vector<shape::dynamic_dimension> input_dims;
        position_mode positions = position_mode::single_token;
    };

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

        auto [cos, sin] = gather_cache(m, ins, in, pos_ids, cos_cache, sin_cache);
        return apply_rotation(m, ins, in, cos, sin);
    }

    gather_context make_gather_context(instruction_ref in,
                                       instruction_ref pos_ids,
                                       instruction_ref cos_cache,
                                       instruction_ref sin_cache) const
    {
        const auto& input_shape = in->get_shape();
        if(input_shape.ndim() != 4)
            MIGRAPHX_THROW("rotary_embedding: expected input of rank 4 with layout "
                           "[batch, heads, seq, head_size] in 4-arg mode");
        const bool symbolic = input_shape.symbolic();

        auto head_size = fixed_dimension(input_shape, 3, "rotary_embedding", "head_size");
        if(head_size % 2 != 0)
            MIGRAPHX_THROW("rotary_embedding: head_size must be even");
        const auto half_head = head_size / 2;
        auto validate_cache  = [&](instruction_ref cache, const std::string& name) {
            const auto& cache_shape = cache->get_shape();
            if(cache_shape.ndim() == 0 or (symbolic and cache_shape.ndim() != 2) or
               fixed_dimension(
                   cache_shape, cache_shape.ndim() - 1, "rotary_embedding", name + " width") !=
                   half_head)
                MIGRAPHX_THROW("rotary_embedding: " + name +
                               " last dimension must equal head_size/2");
        };
        validate_cache(cos_cache, "cos_cache");
        validate_cache(sin_cache, "sin_cache");

        gather_context result;
        result.symbolic   = symbolic;
        result.head_size  = head_size;
        result.input_dims = input_shape.to_symbolic().dyn_dims();
        if(symbolic)
        {
            const auto position_dims = pos_ids->get_shape().to_symbolic().dyn_dims();
            if(position_dims.size() != 2 or
               not sym::same_symbol(position_dims.at(0).sym_expr,
                                    result.input_dims.at(0).sym_expr) or
               not sym::same_symbol(position_dims.at(1).sym_expr, result.input_dims.at(2).sym_expr))
                MIGRAPHX_THROW(
                    "rotary_embedding: position_ids must match input batch and sequence");
            result.positions = position_mode::explicit_positions;
            return result;
        }

        const auto& input_lens = input_shape.lens();
        result.batch           = input_lens.at(0);
        result.sequence        = input_lens.at(2);
        const auto pos_elems   = pos_ids->get_shape().elements();
        if(result.sequence > 1 and pos_elems == result.batch * result.sequence)
            result.positions = position_mode::explicit_positions;
        else if(result.sequence > 1)
            result.positions = position_mode::offset_with_range;
        else
            result.positions = position_mode::single_token;
        return result;
    }

    static instruction_ref
    insert_symbolic_reshape(module& m,
                            instruction_ref ins,
                            instruction_ref input,
                            const std::vector<shape::dynamic_dimension>& dims,
                            const std::vector<instruction_ref>& shape_sources)
    {
        auto result = try_insert_runtime_reshape(m, ins, input, dims, shape_sources);
        if(not result.has_value())
            MIGRAPHX_THROW("rotary_embedding: runtime reshape dimensions have no shape source");
        return *result;
    }

    static instruction_ref insert_reshape(module& m,
                                          instruction_ref ins,
                                          instruction_ref input,
                                          const gather_context& context,
                                          const std::vector<std::size_t>& lens,
                                          const std::vector<shape::dynamic_dimension>& dims,
                                          const std::vector<instruction_ref>& shape_sources)
    {
        if(not context.symbolic)
            return m.insert_instruction(ins, make_op("reshape", {{"dims", lens}}), input);
        return insert_symbolic_reshape(m, ins, input, dims, shape_sources);
    }

    static instruction_ref build_gather_indices(module& m,
                                                instruction_ref ins,
                                                instruction_ref pos_ids,
                                                const gather_context& context)
    {
        if(context.positions == position_mode::explicit_positions)
        {
            std::vector<shape::dynamic_dimension> index_dims{context.input_dims.at(0),
                                                             context.input_dims.at(2),
                                                             shape::dynamic_dimension{sym::lit(1)}};
            return insert_reshape(m,
                                  ins,
                                  pos_ids,
                                  context,
                                  {context.batch, context.sequence, 1},
                                  index_dims,
                                  {pos_ids});
        }

        instruction_ref pos;
        if(pos_ids->get_shape().elements() == 1 and context.batch > 1)
        {
            pos = m.insert_instruction(ins, make_op("reshape", {{"dims", {1, 1, 1}}}), pos_ids);
            pos = m.insert_instruction(
                ins, make_op("multibroadcast", {{"out_lens", {context.batch, 1, 1}}}), pos);
        }
        else
        {
            pos = m.insert_instruction(
                ins, make_op("reshape", {{"dims", {context.batch, 1, 1}}}), pos_ids);
        }

        if(context.positions == position_mode::single_token)
            return pos;

        pos = m.insert_instruction(
            ins,
            make_op("multibroadcast", {{"out_lens", {context.batch, context.sequence, 1}}}),
            pos);
        std::vector<int> range_values(context.sequence);
        std::iota(range_values.begin(), range_values.end(), 0);
        auto range = m.add_literal(migraphx::literal{
            migraphx::shape{pos_ids->get_shape().type(), {1, context.sequence, 1}}, range_values});
        range      = m.insert_instruction(
            ins,
            make_op("multibroadcast", {{"out_lens", {context.batch, context.sequence, 1}}}),
            range);
        return insert_common_op(m, ins, make_op("add"), {pos, range});
    }

    std::pair<instruction_ref, instruction_ref> gather_cache(module& m,
                                                             instruction_ref ins,
                                                             instruction_ref in,
                                                             instruction_ref pos_ids,
                                                             instruction_ref cos_cache,
                                                             instruction_ref sin_cache) const
    {
        auto context = make_gather_context(in, pos_ids, cos_cache, sin_cache);
        auto indices = build_gather_indices(m, ins, pos_ids, context);
        auto cos =
            m.insert_instruction(ins, make_op("gathernd", {{"batch_dims", 0}}), cos_cache, indices);
        auto sin =
            m.insert_instruction(ins, make_op("gathernd", {{"batch_dims", 0}}), sin_cache, indices);

        if(interleaved and context.symbolic)
        {
            std::vector<shape::dynamic_dimension> doubled_dims{
                context.input_dims.at(0),
                context.input_dims.at(2),
                shape::dynamic_dimension{sym::lit(context.head_size)}};
            auto double_values = [&](instruction_ref values) {
                values = m.insert_instruction(ins, make_op("unsqueeze", {{"axes", {3}}}), values);
                values =
                    m.insert_instruction(ins, make_op("concat", {{"axis", 3}}), values, values);
                return insert_symbolic_reshape(m, ins, values, doubled_dims, {in});
            };
            cos = double_values(cos);
            sin = double_values(sin);
        }
        else
        {
            if(interleaved)
            {
                auto cos_elems = cos->get_shape().elements();
                auto sin_elems = sin->get_shape().elements();
                cos =
                    m.insert_instruction(ins, make_op("reshape", {{"dims", {cos_elems, 1}}}), cos);
                sin =
                    m.insert_instruction(ins, make_op("reshape", {{"dims", {sin_elems, 1}}}), sin);
            }
            cos = m.insert_instruction(ins, make_op("concat", {{"axis", -1}}), cos, cos);
            sin = m.insert_instruction(ins, make_op("concat", {{"axis", -1}}), sin, sin);
        }

        std::vector<shape::dynamic_dimension> cache_dims{
            context.input_dims.at(0),
            shape::dynamic_dimension{sym::lit(1)},
            context.input_dims.at(2),
            shape::dynamic_dimension{sym::lit(context.head_size)}};
        cos = insert_reshape(m,
                             ins,
                             cos,
                             context,
                             {context.batch, 1, context.sequence, context.head_size},
                             cache_dims,
                             {in});
        sin = insert_reshape(m,
                             ins,
                             sin,
                             context,
                             {context.batch, 1, context.sequence, context.head_size},
                             cache_dims,
                             {in});
        return {cos, sin};
    }

    std::vector<instruction_ref> apply_rotation(module& m,
                                                instruction_ref ins,
                                                instruction_ref in,
                                                instruction_ref cos,
                                                instruction_ref sin) const
    {
        const auto& input_shape = in->get_shape();
        auto d =
            fixed_dimension(input_shape, input_shape.ndim() - 1, "rotary_embedding", "head_size");
        auto half_d = d / 2;
        auto dtype  = input_shape.type();
        assert((d % 2) == 0);
        auto signs = m.add_literal(migraphx::literal{migraphx::shape{dtype, {2}}, {-1.0f, 1.0f}});

        instruction_ref rotated;

        if(interleaved)
        {
            signs = m.insert_instruction(ins, make_op("reshape", {{"dims", {1, 2}}}), signs);
            signs = m.insert_instruction(
                ins, make_op("multibroadcast", {{"out_lens", {half_d, 2}}}), signs);
            signs = m.insert_instruction(ins, make_op("reshape", {{"dims", {d}}}), signs);

            if(input_shape.symbolic())
            {
                auto pair_dims   = input_shape.dyn_dims();
                pair_dims.back() = shape::dynamic_dimension{sym::lit(half_d)};
                pair_dims.push_back(shape::dynamic_dimension{sym::lit(2)});
                auto rs_in = insert_symbolic_reshape(m, ins, in, pair_dims, {in});
                auto evens = m.insert_instruction(
                    ins, make_op("slice", {{"axes", {4}}, {"starts", {0}}, {"ends", {1}}}), rs_in);
                auto odds = m.insert_instruction(
                    ins, make_op("slice", {{"axes", {4}}, {"starts", {1}}, {"ends", {2}}}), rs_in);
                auto swapped =
                    m.insert_instruction(ins, make_op("concat", {{"axis", 4}}), odds, evens);
                rotated = insert_symbolic_reshape(m, ins, swapped, input_shape.dyn_dims(), {in});
            }
            else
            {
                auto in_lens = input_shape.lens();
                auto n       = input_shape.elements() / 2;
                auto rs_in = m.insert_instruction(ins, make_op("reshape", {{"dims", {n, 2}}}), in);
                auto evens = m.insert_instruction(
                    ins, make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {1}}}), rs_in);
                auto odds = m.insert_instruction(
                    ins, make_op("slice", {{"axes", {1}}, {"starts", {1}}, {"ends", {2}}}), rs_in);
                auto swapped =
                    m.insert_instruction(ins, make_op("concat", {{"axis", -1}}), odds, evens);
                rotated =
                    m.insert_instruction(ins, make_op("reshape", {{"dims", in_lens}}), swapped);
            }
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
                make_op("multibroadcast", {{"out_dyn_dims", to_value(input_shape.dyn_dims())}}),
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
