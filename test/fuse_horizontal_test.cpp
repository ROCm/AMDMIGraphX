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
#include <migraphx/fuse_horizontal.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/generate.hpp>
#include <basic_ops.hpp>
#include <test.hpp>

static void run_pass(migraphx::module& m)
{
    migraphx::run_passes(m, {migraphx::fuse_horizontal{}});
}

// 4 gathers with same embedding dim → should fuse into 1 batched gather
TEST_CASE(gather_horiz_fusion_basic)
{
    migraphx::module m1;
    {
        auto emb1 =
            m1.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {3, 2}}, 0));
        auto emb2 =
            m1.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {4, 2}}, 1));
        auto emb3 =
            m1.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {2, 2}}, 2));
        auto emb4 =
            m1.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {5, 2}}, 3));

        auto idx1 = m1.add_parameter("idx1", {migraphx::shape::int32_type, {2}});
        auto idx2 = m1.add_parameter("idx2", {migraphx::shape::int32_type, {3}});
        auto idx3 = m1.add_parameter("idx3", {migraphx::shape::int32_type, {1}});
        auto idx4 = m1.add_parameter("idx4", {migraphx::shape::int32_type, {2}});

        auto g1 = m1.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), emb1, idx1);
        auto g2 = m1.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), emb2, idx2);
        auto g3 = m1.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), emb3, idx3);
        auto g4 = m1.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), emb4, idx4);

        // Combine all outputs so every gather stays live through DCE
        auto c = m1.add_instruction(migraphx::make_op("concat", {{"axis", 0}}),
                                    std::vector<migraphx::instruction_ref>{g1, g2, g3, g4});
        m1.add_instruction(pass_op{}, c);
    }
    run_pass(m1);

    migraphx::module m2;
    {
        // Embedding literals (added first → pushed to front → end up at the back of no-dep list)
        auto emb1 =
            m2.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {3, 2}}, 0));
        auto emb2 =
            m2.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {4, 2}}, 1));
        auto emb3 =
            m2.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {2, 2}}, 2));
        auto emb4 =
            m2.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {5, 2}}, 3));

        // Parameters (added second → in middle of no-dep list)
        auto idx1 = m2.add_parameter("idx1", {migraphx::shape::int32_type, {2}});
        auto idx2 = m2.add_parameter("idx2", {migraphx::shape::int32_type, {3}});
        auto idx3 = m2.add_parameter("idx3", {migraphx::shape::int32_type, {1}});
        auto idx4 = m2.add_parameter("idx4", {migraphx::shape::int32_type, {2}});

        // Offset literals (added last → pushed to very front of no-dep list,
        // matching order of add_literal calls inside the pass's fuse loop)
        auto offset2 = m2.add_literal(
            migraphx::literal{migraphx::shape{migraphx::shape::int32_type}, {std::size_t(3)}});
        auto offset3 = m2.add_literal(
            migraphx::literal{migraphx::shape{migraphx::shape::int32_type}, {std::size_t(7)}});
        auto offset4 = m2.add_literal(
            migraphx::literal{migraphx::shape{migraphx::shape::int32_type}, {std::size_t(9)}});

        // Concatenated embedding table: [3+4+2+5, 2] = [14, 2]
        auto concat_emb = m2.add_instruction(
            migraphx::make_op("concat", {{"axis", 0}}),
            std::vector<migraphx::instruction_ref>{emb1, emb2, emb3, emb4});

        // Adjust indices with cumulative offsets
        auto bc2 = m2.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {3}}}), offset2);
        auto adj_idx2 = m2.add_instruction(migraphx::make_op("add"), idx2, bc2);

        auto bc3 = m2.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {1}}}), offset3);
        auto adj_idx3 = m2.add_instruction(migraphx::make_op("add"), idx3, bc3);

        auto bc4 = m2.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {2}}}), offset4);
        auto adj_idx4 = m2.add_instruction(migraphx::make_op("add"), idx4, bc4);

        // Concatenated adjusted indices: [2+3+1+2] = [8]
        auto concat_idx = m2.add_instruction(
            migraphx::make_op("concat", {{"axis", 0}}),
            std::vector<migraphx::instruction_ref>{idx1, adj_idx2, adj_idx3, adj_idx4});

        // Single batched gather
        auto bg =
            m2.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), concat_emb, concat_idx);

        // Slice results back
        auto s1 = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {0}}, {"ends", {2}}}), bg);
        auto s2 = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {2}}, {"ends", {5}}}), bg);
        auto s3 = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {5}}, {"ends", {6}}}), bg);
        auto s4 = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {6}}, {"ends", {8}}}), bg);

        // Same concat combiner as m1 (now referencing slices instead of gathers)
        auto c = m2.add_instruction(migraphx::make_op("concat", {{"axis", 0}}),
                                    std::vector<migraphx::instruction_ref>{s1, s2, s3, s4});
        m2.add_instruction(pass_op{}, c);
    }
    EXPECT(m1 == m2);
}

// Only 3 gathers (below min_batch_size=4) → no fusion
TEST_CASE(gather_horiz_no_fusion_below_threshold)
{
    migraphx::module m1;
    {
        auto emb1 = m1.add_literal(
            migraphx::generate_literal({migraphx::shape::float_type, {3, 2}}, 0));
        auto emb2 = m1.add_literal(
            migraphx::generate_literal({migraphx::shape::float_type, {4, 2}}, 1));
        auto emb3 = m1.add_literal(
            migraphx::generate_literal({migraphx::shape::float_type, {2, 2}}, 2));

        auto idx1 = m1.add_parameter("idx1", {migraphx::shape::int32_type, {2}});
        auto idx2 = m1.add_parameter("idx2", {migraphx::shape::int32_type, {3}});
        auto idx3 = m1.add_parameter("idx3", {migraphx::shape::int32_type, {1}});

        auto g1 = m1.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), emb1, idx1);
        auto g2 = m1.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), emb2, idx2);
        auto g3 = m1.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), emb3, idx3);

        auto c = m1.add_instruction(migraphx::make_op("concat", {{"axis", 0}}),
                                    std::vector<migraphx::instruction_ref>{g1, g2, g3});
        m1.add_instruction(pass_op{}, c);
    }
    auto m2 = m1;
    run_pass(m1);
    EXPECT(m1 == m2);
}

// Embeddings are parameters (not constants) → no fusion
TEST_CASE(gather_horiz_no_fusion_non_constant_embedding)
{
    migraphx::module m1;
    {
        auto emb1 = m1.add_parameter("emb1", {migraphx::shape::float_type, {3, 2}});
        auto emb2 = m1.add_parameter("emb2", {migraphx::shape::float_type, {4, 2}});
        auto emb3 = m1.add_parameter("emb3", {migraphx::shape::float_type, {2, 2}});
        auto emb4 = m1.add_parameter("emb4", {migraphx::shape::float_type, {5, 2}});

        auto idx1 = m1.add_parameter("idx1", {migraphx::shape::int32_type, {2}});
        auto idx2 = m1.add_parameter("idx2", {migraphx::shape::int32_type, {3}});
        auto idx3 = m1.add_parameter("idx3", {migraphx::shape::int32_type, {1}});
        auto idx4 = m1.add_parameter("idx4", {migraphx::shape::int32_type, {2}});

        auto g1 = m1.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), emb1, idx1);
        auto g2 = m1.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), emb2, idx2);
        auto g3 = m1.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), emb3, idx3);
        auto g4 = m1.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), emb4, idx4);

        auto c = m1.add_instruction(migraphx::make_op("concat", {{"axis", 0}}),
                                    std::vector<migraphx::instruction_ref>{g1, g2, g3, g4});
        m1.add_instruction(pass_op{}, c);
    }
    auto m2 = m1;
    run_pass(m1);
    EXPECT(m1 == m2);
}

// Gather axis=1 instead of axis=0 → no fusion
TEST_CASE(gather_horiz_no_fusion_wrong_axis)
{
    migraphx::module m1;
    {
        auto emb1 = m1.add_literal(
            migraphx::generate_literal({migraphx::shape::float_type, {3, 4}}, 0));
        auto emb2 = m1.add_literal(
            migraphx::generate_literal({migraphx::shape::float_type, {3, 5}}, 1));
        auto emb3 = m1.add_literal(
            migraphx::generate_literal({migraphx::shape::float_type, {3, 6}}, 2));
        auto emb4 = m1.add_literal(
            migraphx::generate_literal({migraphx::shape::float_type, {3, 7}}, 3));

        auto idx1 = m1.add_parameter("idx1", {migraphx::shape::int32_type, {2}});
        auto idx2 = m1.add_parameter("idx2", {migraphx::shape::int32_type, {2}});
        auto idx3 = m1.add_parameter("idx3", {migraphx::shape::int32_type, {2}});
        auto idx4 = m1.add_parameter("idx4", {migraphx::shape::int32_type, {2}});

        // axis=1 gathers → all outputs are [3, 2]
        auto g1 = m1.add_instruction(migraphx::make_op("gather", {{"axis", 1}}), emb1, idx1);
        auto g2 = m1.add_instruction(migraphx::make_op("gather", {{"axis", 1}}), emb2, idx2);
        auto g3 = m1.add_instruction(migraphx::make_op("gather", {{"axis", 1}}), emb3, idx3);
        auto g4 = m1.add_instruction(migraphx::make_op("gather", {{"axis", 1}}), emb4, idx4);

        auto c = m1.add_instruction(migraphx::make_op("concat", {{"axis", 0}}),
                                    std::vector<migraphx::instruction_ref>{g1, g2, g3, g4});
        m1.add_instruction(pass_op{}, c);
    }
    auto m2 = m1;
    run_pass(m1);
    EXPECT(m1 == m2);
}

// Each embedding has a different embedding dim → separate groups of 1, no fusion
TEST_CASE(gather_horiz_no_fusion_different_emb_dims)
{
    migraphx::module m1;
    {
        auto emb1 = m1.add_literal(
            migraphx::generate_literal({migraphx::shape::float_type, {3, 2}}, 0));
        auto emb2 = m1.add_literal(
            migraphx::generate_literal({migraphx::shape::float_type, {4, 4}}, 1));
        auto emb3 = m1.add_literal(
            migraphx::generate_literal({migraphx::shape::float_type, {2, 8}}, 2));
        auto emb4 = m1.add_literal(
            migraphx::generate_literal({migraphx::shape::float_type, {5, 16}}, 3));

        // All indices same size so outputs are compatible for concat on axis=1
        auto idx1 = m1.add_parameter("idx1", {migraphx::shape::int32_type, {2}});
        auto idx2 = m1.add_parameter("idx2", {migraphx::shape::int32_type, {2}});
        auto idx3 = m1.add_parameter("idx3", {migraphx::shape::int32_type, {2}});
        auto idx4 = m1.add_parameter("idx4", {migraphx::shape::int32_type, {2}});

        // outputs: [2,2], [2,4], [2,8], [2,16]
        auto g1 = m1.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), emb1, idx1);
        auto g2 = m1.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), emb2, idx2);
        auto g3 = m1.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), emb3, idx3);
        auto g4 = m1.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), emb4, idx4);

        // concat on axis=1 since first dims match (2) but second dims differ
        auto c = m1.add_instruction(migraphx::make_op("concat", {{"axis", 1}}),
                                    std::vector<migraphx::instruction_ref>{g1, g2, g3, g4});
        m1.add_instruction(pass_op{}, c);
    }
    auto m2 = m1;
    run_pass(m1);
    EXPECT(m1 == m2);
}

// 3D embedding tables (not 2D) → no fusion
TEST_CASE(gather_horiz_no_fusion_3d_embedding)
{
    migraphx::module m1;
    {
        auto emb1 = m1.add_literal(
            migraphx::generate_literal({migraphx::shape::float_type, {2, 3, 4}}, 0));
        auto emb2 = m1.add_literal(
            migraphx::generate_literal({migraphx::shape::float_type, {2, 3, 4}}, 1));
        auto emb3 = m1.add_literal(
            migraphx::generate_literal({migraphx::shape::float_type, {2, 3, 4}}, 2));
        auto emb4 = m1.add_literal(
            migraphx::generate_literal({migraphx::shape::float_type, {2, 3, 4}}, 3));

        auto idx1 = m1.add_parameter("idx1", {migraphx::shape::int32_type, {2}});
        auto idx2 = m1.add_parameter("idx2", {migraphx::shape::int32_type, {3}});
        auto idx3 = m1.add_parameter("idx3", {migraphx::shape::int32_type, {1}});
        auto idx4 = m1.add_parameter("idx4", {migraphx::shape::int32_type, {2}});

        // outputs: [2,3,4], [3,3,4], [1,3,4], [2,3,4] → concat axis=0 → [8,3,4]
        auto g1 = m1.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), emb1, idx1);
        auto g2 = m1.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), emb2, idx2);
        auto g3 = m1.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), emb3, idx3);
        auto g4 = m1.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), emb4, idx4);

        auto c = m1.add_instruction(migraphx::make_op("concat", {{"axis", 0}}),
                                    std::vector<migraphx::instruction_ref>{g1, g2, g3, g4});
        m1.add_instruction(pass_op{}, c);
    }
    auto m2 = m1;
    run_pass(m1);
    EXPECT(m1 == m2);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
