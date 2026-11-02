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
#include <migraphx/common.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/module.hpp>
#include <migraphx/program.hpp>
#include <migraphx/register_target.hpp>
#include <test.hpp>

static std::vector<float> eval_ref(const migraphx::module& m)
{
    migraphx::program p{m};
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> v;
    result.visit([&](auto r) { v.assign(r.begin(), r.end()); });
    return v;
}

// Replace a concat of `inputs` with insert_concat_broadcasts, and check that the
// result matches the concat and concatenates `compact_elements` elements
static void check_concat_broadcasts(migraphx::module& m,
                                    const std::vector<migraphx::instruction_ref>& inputs,
                                    std::size_t axis,
                                    std::size_t compact_elements)
{
    auto concat = m.add_instruction(migraphx::make_op("concat", {{"axis", axis}}), inputs);
    m.add_return({concat});
    auto expected = eval_ref(m);

    auto bcast = migraphx::insert_concat_broadcasts(m, concat, inputs, axis);
    EXPECT(bcast.has_value());
    if(not bcast.has_value())
        return;
    EXPECT((*bcast)->name() == "multibroadcast");
    EXPECT((*bcast)->get_shape().lens() == concat->get_shape().lens());
    auto compact = (*bcast)->inputs().front();
    EXPECT(compact->name() == "concat");
    EXPECT(compact->get_shape().elements() == compact_elements);
    m.replace_instruction(concat, *bcast);
    EXPECT(eval_ref(m) == expected);
}

// insert_concat_broadcasts returns nullopt and leaves the module unchanged
static void check_no_concat_broadcasts(migraphx::module& m,
                                       const std::vector<migraphx::instruction_ref>& inputs,
                                       std::size_t axis)
{
    auto concat = m.add_instruction(migraphx::make_op("concat", {{"axis", axis}}), inputs);
    auto before = m;
    EXPECT(not migraphx::insert_concat_broadcasts(m, concat, inputs, axis).has_value());
    EXPECT(m == before);
}

TEST_CASE(concat_broadcasts_stacked_weights)
{
    auto ws = migraphx::shape{migraphx::shape::float_type, {8, 5}};
    migraphx::module m;
    std::vector<migraphx::instruction_ref> inputs;
    for(int i = 0; i < 3; ++i)
    {
        auto w = m.add_literal(migraphx::generate_literal(ws, i));
        auto wb =
            m.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {2, 8, 5}}}), w);
        inputs.push_back(m.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0}}}), wb));
    }
    check_concat_broadcasts(m, inputs, 0, 3 * ws.elements());
}

TEST_CASE(concat_broadcasts_bias)
{
    auto bs = migraphx::shape{migraphx::shape::float_type, {3}};
    migraphx::module m;
    std::vector<migraphx::instruction_ref> inputs;
    for(int i = 0; i < 4; ++i)
    {
        auto b = m.add_literal(migraphx::generate_literal(bs, i));
        inputs.push_back(m.add_instruction(
            migraphx::make_op("broadcast", {{"axis", 2}, {"out_lens", {1, 2, 3}}}), b));
    }
    check_concat_broadcasts(m, inputs, 0, 4 * bs.elements());
}

TEST_CASE(concat_broadcasts_transposed_weights)
{
    auto ws = migraphx::shape{migraphx::shape::float_type, {5, 8}};
    migraphx::module m;
    std::vector<migraphx::instruction_ref> inputs;
    for(int i = 0; i < 3; ++i)
    {
        auto w  = m.add_literal(migraphx::generate_literal(ws, i));
        auto wt = m.add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0}}}), w);
        auto wb =
            m.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {2, 8, 5}}}), wt);
        inputs.push_back(m.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0}}}), wb));
    }
    check_concat_broadcasts(m, inputs, 0, 3 * ws.elements());
}

TEST_CASE(concat_broadcasts_reshape_after_broadcast)
{
    auto bs = migraphx::shape{migraphx::shape::float_type, {6}};
    migraphx::module m;
    std::vector<migraphx::instruction_ref> inputs;
    for(int i = 0; i < 2; ++i)
    {
        auto b  = m.add_literal(migraphx::generate_literal(bs, i));
        auto bb = m.add_instruction(
            migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", {4, 6}}}), b);
        inputs.push_back(
            m.add_instruction(migraphx::make_op("reshape", {{"dims", {1, 4, 2, 3}}}), bb));
    }
    check_concat_broadcasts(m, inputs, 0, 2 * bs.elements());
}

TEST_CASE(concat_broadcasts_different_broadcasts)
{
    migraphx::module m;
    auto a  = m.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {3}}, 0));
    auto b  = m.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {2, 1}}, 1));
    auto ab = m.add_instruction(
        migraphx::make_op("broadcast", {{"axis", 2}, {"out_lens", {1, 2, 3}}}), a);
    auto bb = m.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {1, 2, 3}}}), b);
    check_no_concat_broadcasts(m, {ab, bb}, 0);
}

TEST_CASE(concat_broadcasts_not_broadcast)
{
    migraphx::module m;
    auto a  = m.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {2, 3}}, 0));
    auto b  = m.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {2, 3}}, 1));
    auto au = m.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0}}}), a);
    auto bu = m.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0}}}), b);
    check_no_concat_broadcasts(m, {au, bu}, 0);
}

TEST_CASE(concat_broadcasts_axis_not_1)
{
    migraphx::module m;
    std::vector<migraphx::instruction_ref> inputs;
    for(int i = 0; i < 2; ++i)
    {
        auto b = m.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {3}}, i));
        inputs.push_back(m.add_instruction(
            migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", {2, 3}}}), b));
    }
    check_no_concat_broadcasts(m, inputs, 0);
}

TEST_CASE(concat_broadcasts_merged_broadcast_axis)
{
    // The reshape merges a broadcast axis with a non-broadcast one, so the
    // broadcast can't be collapsed
    migraphx::module m;
    std::vector<migraphx::instruction_ref> inputs;
    for(int i = 0; i < 2; ++i)
    {
        auto b  = m.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {3}}, i));
        auto bb = m.add_instruction(
            migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", {2, 3}}}), b);
        inputs.push_back(m.add_instruction(migraphx::make_op("reshape", {{"dims", {1, 6}}}), bb));
    }
    check_no_concat_broadcasts(m, inputs, 0);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
