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
    migraphx::run_passes(m, {migraphx::fuse_horizontal{}, migraphx::dead_code_elimination{}});
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
        m1.add_instruction(migraphx::make_op("concat", {{"axis", 0}}),
                           std::vector<migraphx::instruction_ref>{g1, g2, g3, g4});
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
        auto concat_emb =
            m2.add_instruction(migraphx::make_op("concat", {{"axis", 0}}),
                               std::vector<migraphx::instruction_ref>{emb1, emb2, emb3, emb4});

        // Adjust indices with cumulative offsets
        auto bc2 =
            m2.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {3}}}), offset2);
        auto adj_idx2 = m2.add_instruction(migraphx::make_op("add"), idx2, bc2);

        auto bc3 =
            m2.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {1}}}), offset3);
        auto adj_idx3 = m2.add_instruction(migraphx::make_op("add"), idx3, bc3);

        auto bc4 =
            m2.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {2}}}), offset4);
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
        m2.add_instruction(migraphx::make_op("concat", {{"axis", 0}}),
                           std::vector<migraphx::instruction_ref>{s1, s2, s3, s4});
    }
    EXPECT(m1 == m2);
}

// Only 3 gathers (below min_batch_size=4) → no fusion
TEST_CASE(gather_horiz_no_fusion_below_threshold)
{
    migraphx::module m1;
    {
        auto emb1 =
            m1.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {3, 2}}, 0));
        auto emb2 =
            m1.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {4, 2}}, 1));
        auto emb3 =
            m1.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {2, 2}}, 2));

        auto idx1 = m1.add_parameter("idx1", {migraphx::shape::int32_type, {2}});
        auto idx2 = m1.add_parameter("idx2", {migraphx::shape::int32_type, {3}});
        auto idx3 = m1.add_parameter("idx3", {migraphx::shape::int32_type, {1}});

        auto g1 = m1.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), emb1, idx1);
        auto g2 = m1.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), emb2, idx2);
        auto g3 = m1.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), emb3, idx3);

        m1.add_instruction(migraphx::make_op("concat", {{"axis", 0}}),
                           std::vector<migraphx::instruction_ref>{g1, g2, g3});
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

        m1.add_instruction(migraphx::make_op("concat", {{"axis", 0}}),
                           std::vector<migraphx::instruction_ref>{g1, g2, g3, g4});
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
        auto emb1 =
            m1.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {3, 4}}, 0));
        auto emb2 =
            m1.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {3, 5}}, 1));
        auto emb3 =
            m1.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {3, 6}}, 2));
        auto emb4 =
            m1.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {3, 7}}, 3));

        auto idx1 = m1.add_parameter("idx1", {migraphx::shape::int32_type, {2}});
        auto idx2 = m1.add_parameter("idx2", {migraphx::shape::int32_type, {2}});
        auto idx3 = m1.add_parameter("idx3", {migraphx::shape::int32_type, {2}});
        auto idx4 = m1.add_parameter("idx4", {migraphx::shape::int32_type, {2}});

        // axis=1 gathers → all outputs are [3, 2]
        auto g1 = m1.add_instruction(migraphx::make_op("gather", {{"axis", 1}}), emb1, idx1);
        auto g2 = m1.add_instruction(migraphx::make_op("gather", {{"axis", 1}}), emb2, idx2);
        auto g3 = m1.add_instruction(migraphx::make_op("gather", {{"axis", 1}}), emb3, idx3);
        auto g4 = m1.add_instruction(migraphx::make_op("gather", {{"axis", 1}}), emb4, idx4);

        m1.add_instruction(migraphx::make_op("concat", {{"axis", 0}}),
                           std::vector<migraphx::instruction_ref>{g1, g2, g3, g4});
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
        auto emb1 =
            m1.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {3, 2}}, 0));
        auto emb2 =
            m1.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {4, 4}}, 1));
        auto emb3 =
            m1.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {2, 8}}, 2));
        auto emb4 =
            m1.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {5, 16}}, 3));

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
        m1.add_instruction(migraphx::make_op("concat", {{"axis", 1}}),
                           std::vector<migraphx::instruction_ref>{g1, g2, g3, g4});
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
        auto emb1 =
            m1.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {2, 3, 4}}, 0));
        auto emb2 =
            m1.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {2, 3, 4}}, 1));
        auto emb3 =
            m1.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {2, 3, 4}}, 2));
        auto emb4 =
            m1.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {2, 3, 4}}, 3));

        auto idx1 = m1.add_parameter("idx1", {migraphx::shape::int32_type, {2}});
        auto idx2 = m1.add_parameter("idx2", {migraphx::shape::int32_type, {3}});
        auto idx3 = m1.add_parameter("idx3", {migraphx::shape::int32_type, {1}});
        auto idx4 = m1.add_parameter("idx4", {migraphx::shape::int32_type, {2}});

        // outputs: [2,3,4], [3,3,4], [1,3,4], [2,3,4] → concat axis=0 → [8,3,4]
        auto g1 = m1.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), emb1, idx1);
        auto g2 = m1.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), emb2, idx2);
        auto g3 = m1.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), emb3, idx3);
        auto g4 = m1.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), emb4, idx4);

        m1.add_instruction(migraphx::make_op("concat", {{"axis", 0}}),
                           std::vector<migraphx::instruction_ref>{g1, g2, g3, g4});
    }
    auto m2 = m1;
    run_pass(m1);
    EXPECT(m1 == m2);
}

// First gather's output is used before the second gather — consumers are interleaved
// The pass should still fuse and move_output_instructions_after handles reordering
TEST_CASE(gather_horiz_fusion_interleaved_consumers)
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
        auto idx2 = m1.add_parameter("idx2", {migraphx::shape::int32_type, {2}});
        auto idx3 = m1.add_parameter("idx3", {migraphx::shape::int32_type, {2}});
        auto idx4 = m1.add_parameter("idx4", {migraphx::shape::int32_type, {2}});

        auto g1 = m1.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), emb1, idx1);

        // g1's output is consumed here — between g1 and g2
        auto relu1 = m1.add_instruction(migraphx::make_op("relu"), g1);

        auto g2 = m1.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), emb2, idx2);
        auto g3 = m1.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), emb3, idx3);
        auto g4 = m1.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), emb4, idx4);

        m1.add_instruction(migraphx::make_op("concat", {{"axis", 0}}),
                           std::vector<migraphx::instruction_ref>{relu1, g2, g3, g4});
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto emb1 =
            m2.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {3, 2}}, 0));
        auto emb2 =
            m2.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {4, 2}}, 1));
        auto emb3 =
            m2.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {2, 2}}, 2));
        auto emb4 =
            m2.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {5, 2}}, 3));

        auto idx1 = m2.add_parameter("idx1", {migraphx::shape::int32_type, {2}});
        auto idx2 = m2.add_parameter("idx2", {migraphx::shape::int32_type, {2}});
        auto idx3 = m2.add_parameter("idx3", {migraphx::shape::int32_type, {2}});
        auto idx4 = m2.add_parameter("idx4", {migraphx::shape::int32_type, {2}});

        auto offset2 = m2.add_literal(
            migraphx::literal{migraphx::shape{migraphx::shape::int32_type}, {std::size_t(3)}});
        auto offset3 = m2.add_literal(
            migraphx::literal{migraphx::shape{migraphx::shape::int32_type}, {std::size_t(7)}});
        auto offset4 = m2.add_literal(
            migraphx::literal{migraphx::shape{migraphx::shape::int32_type}, {std::size_t(9)}});

        auto concat_emb =
            m2.add_instruction(migraphx::make_op("concat", {{"axis", 0}}),
                               std::vector<migraphx::instruction_ref>{emb1, emb2, emb3, emb4});

        auto bc2 =
            m2.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {2}}}), offset2);
        auto adj_idx2 = m2.add_instruction(migraphx::make_op("add"), idx2, bc2);

        auto bc3 =
            m2.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {2}}}), offset3);
        auto adj_idx3 = m2.add_instruction(migraphx::make_op("add"), idx3, bc3);

        auto bc4 =
            m2.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {2}}}), offset4);
        auto adj_idx4 = m2.add_instruction(migraphx::make_op("add"), idx4, bc4);

        auto concat_idx = m2.add_instruction(
            migraphx::make_op("concat", {{"axis", 0}}),
            std::vector<migraphx::instruction_ref>{idx1, adj_idx2, adj_idx3, adj_idx4});

        auto bg =
            m2.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), concat_emb, concat_idx);

        auto s1 = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {0}}, {"ends", {2}}}), bg);
        auto s2 = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {2}}, {"ends", {4}}}), bg);
        auto s3 = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {4}}, {"ends", {6}}}), bg);
        auto s4 = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {6}}, {"ends", {8}}}), bg);

        // relu was on g1, now on s1 — moved after slices
        auto relu1 = m2.add_instruction(migraphx::make_op("relu"), s1);

        m2.add_instruction(migraphx::make_op("concat", {{"axis", 0}}),
                           std::vector<migraphx::instruction_ref>{relu1, s2, s3, s4});
    }
    EXPECT(m1 == m2);
}

// Shared index: all 4 gathers use the same index parameter
TEST_CASE(gather_horiz_fusion_shared_index)
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

        auto idx = m1.add_parameter("idx", {migraphx::shape::int32_type, {2}});

        auto g1 = m1.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), emb1, idx);
        auto g2 = m1.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), emb2, idx);
        auto g3 = m1.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), emb3, idx);
        auto g4 = m1.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), emb4, idx);

        m1.add_instruction(migraphx::make_op("concat", {{"axis", 0}}),
                           std::vector<migraphx::instruction_ref>{g1, g2, g3, g4});
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto emb1 =
            m2.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {3, 2}}, 0));
        auto emb2 =
            m2.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {4, 2}}, 1));
        auto emb3 =
            m2.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {2, 2}}, 2));
        auto emb4 =
            m2.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {5, 2}}, 3));

        auto idx = m2.add_parameter("idx", {migraphx::shape::int32_type, {2}});

        auto offset2 = m2.add_literal(
            migraphx::literal{migraphx::shape{migraphx::shape::int32_type}, {std::size_t(3)}});
        auto offset3 = m2.add_literal(
            migraphx::literal{migraphx::shape{migraphx::shape::int32_type}, {std::size_t(7)}});
        auto offset4 = m2.add_literal(
            migraphx::literal{migraphx::shape{migraphx::shape::int32_type}, {std::size_t(9)}});

        auto concat_emb =
            m2.add_instruction(migraphx::make_op("concat", {{"axis", 0}}),
                               std::vector<migraphx::instruction_ref>{emb1, emb2, emb3, emb4});

        auto bc2 =
            m2.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {2}}}), offset2);
        auto adj_idx2 = m2.add_instruction(migraphx::make_op("add"), idx, bc2);

        auto bc3 =
            m2.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {2}}}), offset3);
        auto adj_idx3 = m2.add_instruction(migraphx::make_op("add"), idx, bc3);

        auto bc4 =
            m2.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {2}}}), offset4);
        auto adj_idx4 = m2.add_instruction(migraphx::make_op("add"), idx, bc4);

        auto concat_idx = m2.add_instruction(
            migraphx::make_op("concat", {{"axis", 0}}),
            std::vector<migraphx::instruction_ref>{idx, adj_idx2, adj_idx3, adj_idx4});

        auto bg =
            m2.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), concat_emb, concat_idx);

        auto s1 = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {0}}, {"ends", {2}}}), bg);
        auto s2 = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {2}}, {"ends", {4}}}), bg);
        auto s3 = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {4}}, {"ends", {6}}}), bg);
        auto s4 = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {6}}, {"ends", {8}}}), bg);

        m2.add_instruction(migraphx::make_op("concat", {{"axis", 0}}),
                           std::vector<migraphx::instruction_ref>{s1, s2, s3, s4});
    }
    EXPECT(m1 == m2);
}

// Dependent gathers: g2 depends on g1's output → only independent ones fuse
// Since g1→g2 dependency exists, group_by won't group them together.
// With only 3 remaining independent gathers, below min_group_size=4, no fusion.
TEST_CASE(gather_horiz_no_fusion_dependent)
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
        auto idx3 = m1.add_parameter("idx3", {migraphx::shape::int32_type, {2}});
        auto idx4 = m1.add_parameter("idx4", {migraphx::shape::int32_type, {2}});

        auto g1 = m1.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), emb1, idx1);

        // g2 uses g1's output shape to derive its index (dependency)
        auto reshape_g1 = m1.add_instruction(migraphx::make_op("reshape", {{"dims", {4}}}), g1);
        auto g2 = m1.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), emb2, reshape_g1);

        auto g3 = m1.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), emb3, idx3);
        auto g4 = m1.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), emb4, idx4);

        m1.add_instruction(migraphx::make_op("concat", {{"axis", 0}}),
                           std::vector<migraphx::instruction_ref>{g1, g2, g3, g4});
    }
    auto m2 = m1;
    run_pass(m1);
    EXPECT(m1 == m2);
}

// ---------------------------------------------------------------------------
// Generic dot horizontal fusion tests
//
// All structurally-identical dots with constant weights are batched into a
// single GEMM via unsqueeze+concat on axis 0, with slice+squeeze to extract
// individual results.  Downstream pointwise ops (SiLU, GeLU, etc.) are
// lifted by find_squeeze_splits in simplify_algebra, not here.
// ---------------------------------------------------------------------------

// Two 2D dots with same output shape → batched into single dot
TEST_CASE(dot_horiz_fusion_basic)
{
    migraphx::module m1;
    {
        auto w1 =
            m1.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {8, 4}}, 0));
        auto w2 =
            m1.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {8, 4}}, 1));

        auto x1 = m1.add_parameter("x1", {migraphx::shape::float_type, {4, 8}});
        auto x2 = m1.add_parameter("x2", {migraphx::shape::float_type, {4, 8}});

        auto d1 = m1.add_instruction(migraphx::make_op("dot"), x1, w1);
        auto d2 = m1.add_instruction(migraphx::make_op("dot"), x2, w2);

        m1.add_instruction(migraphx::make_op("concat", {{"axis", 0}}),
                           std::vector<migraphx::instruction_ref>{d1, d2});
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto w1 =
            m2.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {8, 4}}, 0));
        auto w2 =
            m2.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {8, 4}}, 1));

        auto x1 = m2.add_parameter("x1", {migraphx::shape::float_type, {4, 8}});
        auto x2 = m2.add_parameter("x2", {migraphx::shape::float_type, {4, 8}});

        auto ux1     = m2.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0}}}), x1);
        auto ux2     = m2.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0}}}), x2);
        auto bat_act = m2.add_instruction(migraphx::make_op("concat", {{"axis", 0}}),
                                          std::vector<migraphx::instruction_ref>{ux1, ux2});

        auto uw1    = m2.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0}}}), w1);
        auto uw2    = m2.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0}}}), w2);
        auto bat_wt = m2.add_instruction(migraphx::make_op("concat", {{"axis", 0}}),
                                         std::vector<migraphx::instruction_ref>{uw1, uw2});

        auto bd = m2.add_instruction(migraphx::make_op("dot"), bat_act, bat_wt);

        auto s1 = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {0}}, {"ends", {1}}}), bd);
        auto sq1 = m2.add_instruction(migraphx::make_op("squeeze", {{"axes", {0}}}), s1);

        auto s2 = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {1}}, {"ends", {2}}}), bd);
        auto sq2 = m2.add_instruction(migraphx::make_op("squeeze", {{"axes", {0}}}), s2);

        m2.add_instruction(migraphx::make_op("concat", {{"axis", 0}}),
                           std::vector<migraphx::instruction_ref>{sq1, sq2});
    }
    EXPECT(m1 == m2);
}

// 3D dots with same output shape → batched
TEST_CASE(dot_horiz_fusion_3d)
{
    migraphx::module m1;
    {
        auto w1 =
            m1.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {2, 8, 4}}, 0));
        auto w2 =
            m1.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {2, 8, 4}}, 1));

        auto x1 = m1.add_parameter("x1", {migraphx::shape::float_type, {2, 4, 8}});
        auto x2 = m1.add_parameter("x2", {migraphx::shape::float_type, {2, 4, 8}});

        auto d1 = m1.add_instruction(migraphx::make_op("dot"), x1, w1);
        auto d2 = m1.add_instruction(migraphx::make_op("dot"), x2, w2);

        m1.add_instruction(migraphx::make_op("concat", {{"axis", 0}}),
                           std::vector<migraphx::instruction_ref>{d1, d2});
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto w1 =
            m2.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {2, 8, 4}}, 0));
        auto w2 =
            m2.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {2, 8, 4}}, 1));

        auto x1 = m2.add_parameter("x1", {migraphx::shape::float_type, {2, 4, 8}});
        auto x2 = m2.add_parameter("x2", {migraphx::shape::float_type, {2, 4, 8}});

        auto ux1     = m2.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0}}}), x1);
        auto ux2     = m2.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0}}}), x2);
        auto bat_act = m2.add_instruction(migraphx::make_op("concat", {{"axis", 0}}),
                                          std::vector<migraphx::instruction_ref>{ux1, ux2});

        auto uw1    = m2.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0}}}), w1);
        auto uw2    = m2.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0}}}), w2);
        auto bat_wt = m2.add_instruction(migraphx::make_op("concat", {{"axis", 0}}),
                                         std::vector<migraphx::instruction_ref>{uw1, uw2});

        auto bd = m2.add_instruction(migraphx::make_op("dot"), bat_act, bat_wt);

        auto s1 = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {0}}, {"ends", {1}}}), bd);
        auto sq1 = m2.add_instruction(migraphx::make_op("squeeze", {{"axes", {0}}}), s1);

        auto s2 = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {1}}, {"ends", {2}}}), bd);
        auto sq2 = m2.add_instruction(migraphx::make_op("squeeze", {{"axes", {0}}}), s2);

        m2.add_instruction(migraphx::make_op("concat", {{"axis", 0}}),
                           std::vector<migraphx::instruction_ref>{sq1, sq2});
    }
    EXPECT(m1 == m2);
}

// Weights are parameters (not constants) → no fusion
TEST_CASE(dot_horiz_no_fusion_non_constant_weight)
{
    migraphx::module m1;
    {
        auto x1 = m1.add_parameter("x1", {migraphx::shape::float_type, {4, 8}});
        auto x2 = m1.add_parameter("x2", {migraphx::shape::float_type, {4, 8}});
        auto w1 = m1.add_parameter("w1", {migraphx::shape::float_type, {8, 4}});
        auto w2 = m1.add_parameter("w2", {migraphx::shape::float_type, {8, 4}});

        auto d1 = m1.add_instruction(migraphx::make_op("dot"), x1, w1);
        auto d2 = m1.add_instruction(migraphx::make_op("dot"), x2, w2);

        m1.add_instruction(migraphx::make_op("concat", {{"axis", 0}}),
                           std::vector<migraphx::instruction_ref>{d1, d2});
    }
    auto m2 = m1;
    run_pass(m1);
    EXPECT(m1 == m2);
}

// Different output shapes → no fusion
TEST_CASE(dot_horiz_no_fusion_different_k)
{
    migraphx::module m1;
    {
        auto w1 =
            m1.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {8, 4}}, 0));
        auto w2 =
            m1.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {16, 4}}, 1));

        auto x1 = m1.add_parameter("x1", {migraphx::shape::float_type, {4, 8}});
        auto x2 = m1.add_parameter("x2", {migraphx::shape::float_type, {4, 16}});

        auto d1 = m1.add_instruction(migraphx::make_op("dot"), x1, w1);
        auto d2 = m1.add_instruction(migraphx::make_op("dot"), x2, w2);

        m1.add_instruction(migraphx::make_op("concat", {{"axis", 0}}),
                           std::vector<migraphx::instruction_ref>{d1, d2});
    }
    auto m2 = m1;
    run_pass(m1);
    EXPECT(m1 == m2);
}

// Different batch dimensions → no fusion
TEST_CASE(dot_horiz_no_fusion_different_batch)
{
    migraphx::module m1;
    {
        auto w1 =
            m1.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {1, 8, 4}}, 0));
        auto w2 =
            m1.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {2, 8, 4}}, 1));

        auto x1 = m1.add_parameter("x1", {migraphx::shape::float_type, {1, 4, 8}});
        auto x2 = m1.add_parameter("x2", {migraphx::shape::float_type, {2, 4, 8}});

        auto d1 = m1.add_instruction(migraphx::make_op("dot"), x1, w1);
        auto d2 = m1.add_instruction(migraphx::make_op("dot"), x2, w2);

        m1.add_instruction(migraphx::make_op("concat", {{"axis", 0}}),
                           std::vector<migraphx::instruction_ref>{d1, d2});
    }
    auto m2 = m1;
    run_pass(m1);
    EXPECT(m1 == m2);
}

// Different M dimensions → no fusion
TEST_CASE(dot_horiz_no_fusion_different_m)
{
    migraphx::module m1;
    {
        auto w1 =
            m1.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {8, 4}}, 0));
        auto w2 =
            m1.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {8, 4}}, 1));

        auto x1 = m1.add_parameter("x1", {migraphx::shape::float_type, {4, 8}});
        auto x2 = m1.add_parameter("x2", {migraphx::shape::float_type, {6, 8}});

        auto d1 = m1.add_instruction(migraphx::make_op("dot"), x1, w1);
        auto d2 = m1.add_instruction(migraphx::make_op("dot"), x2, w2);

        m1.add_instruction(migraphx::make_op("concat", {{"axis", 0}}),
                           std::vector<migraphx::instruction_ref>{d1, d2});
    }
    auto m2 = m1;
    run_pass(m1);
    EXPECT(m1 == m2);
}

// Single MLP tower (only one dot per shape) → no fusion
TEST_CASE(dot_horiz_no_fusion_single_tower)
{
    migraphx::module m1;
    {
        auto w1 =
            m1.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {8, 4}}, 0));
        auto b1 = m1.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {4}}, 1));
        auto w2 =
            m1.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {4, 2}}, 2));

        auto x = m1.add_parameter("x", {migraphx::shape::float_type, {4, 8}});

        auto d1 = m1.add_instruction(migraphx::make_op("dot"), x, w1);
        auto bc =
            m1.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {4, 4}}}), b1);
        auto add = m1.add_instruction(migraphx::make_op("add"), d1, bc);
        auto sig = m1.add_instruction(migraphx::make_op("sigmoid"), add);
        auto mul = m1.add_instruction(migraphx::make_op("mul"), sig, add);
        m1.add_instruction(migraphx::make_op("dot"), mul, w2);
    }
    auto m2 = m1;
    run_pass(m1);
    EXPECT(m1 == m2);
}

// Dots followed by add(dot, broadcast(const)) → batched dot + batched add
TEST_CASE(dot_horiz_fusion_absorb_bias)
{
    migraphx::module m1;
    {
        auto w1 =
            m1.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {8, 4}}, 0));
        auto w2 =
            m1.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {8, 4}}, 1));
        auto b1 = m1.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {4}}, 2));
        auto b2 = m1.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {4}}, 3));

        auto x1 = m1.add_parameter("x1", {migraphx::shape::float_type, {4, 8}});
        auto x2 = m1.add_parameter("x2", {migraphx::shape::float_type, {4, 8}});

        auto d1 = m1.add_instruction(migraphx::make_op("dot"), x1, w1);
        auto bc1 =
            m1.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {4, 4}}}), b1);
        auto a1 = m1.add_instruction(migraphx::make_op("add"), d1, bc1);

        auto d2 = m1.add_instruction(migraphx::make_op("dot"), x2, w2);
        auto bc2 =
            m1.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {4, 4}}}), b2);
        auto a2 = m1.add_instruction(migraphx::make_op("add"), d2, bc2);

        m1.add_instruction(migraphx::make_op("concat", {{"axis", 0}}),
                           std::vector<migraphx::instruction_ref>{a1, a2});
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto w1 =
            m2.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {8, 4}}, 0));
        auto w2 =
            m2.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {8, 4}}, 1));
        auto b1 = m2.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {4}}, 2));
        auto b2 = m2.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {4}}, 3));

        auto x1 = m2.add_parameter("x1", {migraphx::shape::float_type, {4, 8}});
        auto x2 = m2.add_parameter("x2", {migraphx::shape::float_type, {4, 8}});

        // Broadcasts stay at their original positions (before the new instructions)
        auto bc1 =
            m2.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {4, 4}}}), b1);
        auto bc2 =
            m2.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {4, 4}}}), b2);

        auto ux1     = m2.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0}}}), x1);
        auto ux2     = m2.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0}}}), x2);
        auto bat_act = m2.add_instruction(migraphx::make_op("concat", {{"axis", 0}}),
                                          std::vector<migraphx::instruction_ref>{ux1, ux2});

        auto uw1    = m2.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0}}}), w1);
        auto uw2    = m2.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0}}}), w2);
        auto bat_wt = m2.add_instruction(migraphx::make_op("concat", {{"axis", 0}}),
                                         std::vector<migraphx::instruction_ref>{uw1, uw2});

        auto bd = m2.add_instruction(migraphx::make_op("dot"), bat_act, bat_wt);

        auto ubc1      = m2.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0}}}), bc1);
        auto ubc2      = m2.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0}}}), bc2);
        auto stacked_b = m2.add_instruction(migraphx::make_op("concat", {{"axis", 0}}),
                                            std::vector<migraphx::instruction_ref>{ubc1, ubc2});
        auto bd_add    = m2.add_instruction(migraphx::make_op("add"), bd, stacked_b);

        auto s1 = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {0}}, {"ends", {1}}}), bd_add);
        auto sq1 = m2.add_instruction(migraphx::make_op("squeeze", {{"axes", {0}}}), s1);

        auto s2 = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {1}}, {"ends", {2}}}), bd_add);
        auto sq2 = m2.add_instruction(migraphx::make_op("squeeze", {{"axes", {0}}}), s2);

        m2.add_instruction(migraphx::make_op("concat", {{"axis", 0}}),
                           std::vector<migraphx::instruction_ref>{sq1, sq2});
    }
    EXPECT(m1 == m2);
}

// Dots followed by SiLU (sigmoid*mul) → batched dot + batched SiLU
TEST_CASE(dot_horiz_fusion_absorb_silu)
{
    migraphx::module m1;
    {
        auto w1 =
            m1.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {8, 4}}, 0));
        auto w2 =
            m1.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {8, 4}}, 1));

        auto x1 = m1.add_parameter("x1", {migraphx::shape::float_type, {4, 8}});
        auto x2 = m1.add_parameter("x2", {migraphx::shape::float_type, {4, 8}});

        auto d1   = m1.add_instruction(migraphx::make_op("dot"), x1, w1);
        auto sig1 = m1.add_instruction(migraphx::make_op("sigmoid"), d1);
        auto mul1 = m1.add_instruction(migraphx::make_op("mul"), d1, sig1);

        auto d2   = m1.add_instruction(migraphx::make_op("dot"), x2, w2);
        auto sig2 = m1.add_instruction(migraphx::make_op("sigmoid"), d2);
        auto mul2 = m1.add_instruction(migraphx::make_op("mul"), d2, sig2);

        m1.add_instruction(migraphx::make_op("concat", {{"axis", 0}}),
                           std::vector<migraphx::instruction_ref>{mul1, mul2});
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto w1 =
            m2.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {8, 4}}, 0));
        auto w2 =
            m2.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {8, 4}}, 1));

        auto x1 = m2.add_parameter("x1", {migraphx::shape::float_type, {4, 8}});
        auto x2 = m2.add_parameter("x2", {migraphx::shape::float_type, {4, 8}});

        auto ux1     = m2.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0}}}), x1);
        auto ux2     = m2.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0}}}), x2);
        auto bat_act = m2.add_instruction(migraphx::make_op("concat", {{"axis", 0}}),
                                          std::vector<migraphx::instruction_ref>{ux1, ux2});

        auto uw1    = m2.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0}}}), w1);
        auto uw2    = m2.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0}}}), w2);
        auto bat_wt = m2.add_instruction(migraphx::make_op("concat", {{"axis", 0}}),
                                         std::vector<migraphx::instruction_ref>{uw1, uw2});

        auto bd      = m2.add_instruction(migraphx::make_op("dot"), bat_act, bat_wt);
        auto sig_all = m2.add_instruction(migraphx::make_op("sigmoid"), bd);
        auto silu    = m2.add_instruction(migraphx::make_op("mul"), bd, sig_all);

        auto s1 = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {0}}, {"ends", {1}}}), silu);
        auto sq1 = m2.add_instruction(migraphx::make_op("squeeze", {{"axes", {0}}}), s1);

        auto s2 = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {1}}, {"ends", {2}}}), silu);
        auto sq2 = m2.add_instruction(migraphx::make_op("squeeze", {{"axes", {0}}}), s2);

        m2.add_instruction(migraphx::make_op("concat", {{"axis", 0}}),
                           std::vector<migraphx::instruction_ref>{sq1, sq2});
    }
    EXPECT(m1 == m2);
}

// Dots + bias + SiLU → batched dot + batched add + batched SiLU
TEST_CASE(dot_horiz_fusion_absorb_bias_silu)
{
    migraphx::module m1;
    {
        auto w1 =
            m1.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {8, 4}}, 0));
        auto w2 =
            m1.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {8, 4}}, 1));
        auto b1 = m1.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {4}}, 2));
        auto b2 = m1.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {4}}, 3));

        auto x1 = m1.add_parameter("x1", {migraphx::shape::float_type, {4, 8}});
        auto x2 = m1.add_parameter("x2", {migraphx::shape::float_type, {4, 8}});

        auto d1 = m1.add_instruction(migraphx::make_op("dot"), x1, w1);
        auto bc1 =
            m1.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {4, 4}}}), b1);
        auto a1   = m1.add_instruction(migraphx::make_op("add"), d1, bc1);
        auto sig1 = m1.add_instruction(migraphx::make_op("sigmoid"), a1);
        auto mul1 = m1.add_instruction(migraphx::make_op("mul"), a1, sig1);

        auto d2 = m1.add_instruction(migraphx::make_op("dot"), x2, w2);
        auto bc2 =
            m1.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {4, 4}}}), b2);
        auto a2   = m1.add_instruction(migraphx::make_op("add"), d2, bc2);
        auto sig2 = m1.add_instruction(migraphx::make_op("sigmoid"), a2);
        auto mul2 = m1.add_instruction(migraphx::make_op("mul"), a2, sig2);

        m1.add_instruction(migraphx::make_op("concat", {{"axis", 0}}),
                           std::vector<migraphx::instruction_ref>{mul1, mul2});
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto w1 =
            m2.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {8, 4}}, 0));
        auto w2 =
            m2.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {8, 4}}, 1));
        auto b1 = m2.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {4}}, 2));
        auto b2 = m2.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {4}}, 3));

        auto x1 = m2.add_parameter("x1", {migraphx::shape::float_type, {4, 8}});
        auto x2 = m2.add_parameter("x2", {migraphx::shape::float_type, {4, 8}});

        auto bc1 =
            m2.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {4, 4}}}), b1);
        auto bc2 =
            m2.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {4, 4}}}), b2);

        auto ux1     = m2.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0}}}), x1);
        auto ux2     = m2.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0}}}), x2);
        auto bat_act = m2.add_instruction(migraphx::make_op("concat", {{"axis", 0}}),
                                          std::vector<migraphx::instruction_ref>{ux1, ux2});

        auto uw1    = m2.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0}}}), w1);
        auto uw2    = m2.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0}}}), w2);
        auto bat_wt = m2.add_instruction(migraphx::make_op("concat", {{"axis", 0}}),
                                         std::vector<migraphx::instruction_ref>{uw1, uw2});

        auto bd = m2.add_instruction(migraphx::make_op("dot"), bat_act, bat_wt);

        auto ubc1      = m2.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0}}}), bc1);
        auto ubc2      = m2.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0}}}), bc2);
        auto stacked_b = m2.add_instruction(migraphx::make_op("concat", {{"axis", 0}}),
                                            std::vector<migraphx::instruction_ref>{ubc1, ubc2});
        auto bd_add    = m2.add_instruction(migraphx::make_op("add"), bd, stacked_b);

        auto sig_all = m2.add_instruction(migraphx::make_op("sigmoid"), bd_add);
        auto silu    = m2.add_instruction(migraphx::make_op("mul"), bd_add, sig_all);

        auto s1 = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {0}}, {"ends", {1}}}), silu);
        auto sq1 = m2.add_instruction(migraphx::make_op("squeeze", {{"axes", {0}}}), s1);

        auto s2 = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {1}}, {"ends", {2}}}), silu);
        auto sq2 = m2.add_instruction(migraphx::make_op("squeeze", {{"axes", {0}}}), s2);

        m2.add_instruction(migraphx::make_op("concat", {{"axis", 0}}),
                           std::vector<migraphx::instruction_ref>{sq1, sq2});
    }
    EXPECT(m1 == m2);
}

// When some dots in a potential batch have SiLU and others don't,
// they must be placed in separate groups so that SiLU absorption
// succeeds for the SiLU dots.
TEST_CASE(dot_horiz_fusion_partial_silu_split)
{
    migraphx::module m1;
    {
        auto w1 =
            m1.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {8, 4}}, 0));
        auto w2 =
            m1.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {8, 4}}, 1));
        auto w3 =
            m1.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {8, 4}}, 2));

        auto x1 = m1.add_parameter("x1", {migraphx::shape::float_type, {4, 8}});
        auto x2 = m1.add_parameter("x2", {migraphx::shape::float_type, {4, 8}});
        auto x3 = m1.add_parameter("x3", {migraphx::shape::float_type, {4, 8}});

        // d1 and d2 have SiLU
        auto d1   = m1.add_instruction(migraphx::make_op("dot"), x1, w1);
        auto sig1 = m1.add_instruction(migraphx::make_op("sigmoid"), d1);
        auto mul1 = m1.add_instruction(migraphx::make_op("mul"), d1, sig1);

        auto d2   = m1.add_instruction(migraphx::make_op("dot"), x2, w2);
        auto sig2 = m1.add_instruction(migraphx::make_op("sigmoid"), d2);
        auto mul2 = m1.add_instruction(migraphx::make_op("mul"), d2, sig2);

        // d3 has NO SiLU — just a plain relu
        auto d3 = m1.add_instruction(migraphx::make_op("dot"), x3, w3);
        auto r3 = m1.add_instruction(migraphx::make_op("relu"), d3);

        m1.add_instruction(migraphx::make_op("concat", {{"axis", 0}}),
                           std::vector<migraphx::instruction_ref>{mul1, mul2, r3});
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto w1 =
            m2.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {8, 4}}, 0));
        auto w2 =
            m2.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {8, 4}}, 1));
        auto w3 =
            m2.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {8, 4}}, 2));

        auto x1 = m2.add_parameter("x1", {migraphx::shape::float_type, {4, 8}});
        auto x2 = m2.add_parameter("x2", {migraphx::shape::float_type, {4, 8}});
        auto x3 = m2.add_parameter("x3", {migraphx::shape::float_type, {4, 8}});

        // SiLU group: d1, d2 → batched dot + batched SiLU
        auto ux1     = m2.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0}}}), x1);
        auto ux2     = m2.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0}}}), x2);
        auto bat_act = m2.add_instruction(migraphx::make_op("concat", {{"axis", 0}}),
                                          std::vector<migraphx::instruction_ref>{ux1, ux2});
        auto uw1    = m2.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0}}}), w1);
        auto uw2    = m2.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0}}}), w2);
        auto bat_wt = m2.add_instruction(migraphx::make_op("concat", {{"axis", 0}}),
                                         std::vector<migraphx::instruction_ref>{uw1, uw2});
        auto bd      = m2.add_instruction(migraphx::make_op("dot"), bat_act, bat_wt);
        auto sig_all = m2.add_instruction(migraphx::make_op("sigmoid"), bd);
        auto silu    = m2.add_instruction(migraphx::make_op("mul"), bd, sig_all);
        auto s1 = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {0}}, {"ends", {1}}}), silu);
        auto sq1 = m2.add_instruction(migraphx::make_op("squeeze", {{"axes", {0}}}), s1);
        auto s2 = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {1}}, {"ends", {2}}}), silu);
        auto sq2 = m2.add_instruction(migraphx::make_op("squeeze", {{"axes", {0}}}), s2);

        // Non-SiLU d3 stays alone (below min_group_size threshold)
        auto d3 = m2.add_instruction(migraphx::make_op("dot"), x3, w3);
        auto r3 = m2.add_instruction(migraphx::make_op("relu"), d3);

        m2.add_instruction(migraphx::make_op("concat", {{"axis", 0}}),
                           std::vector<migraphx::instruction_ref>{sq1, sq2, r3});
    }
    EXPECT(m1 == m2);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
