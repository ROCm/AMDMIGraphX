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

// ===========================================================================
// Dot horizontal fusion tests
// ===========================================================================

// 4 dots with constant weights and same shapes → fuse into 1 batched dot
TEST_CASE(dot_horiz_fusion_basic)
{
    migraphx::module m1;
    {
        auto input = m1.add_parameter("input", {migraphx::shape::float_type, {1, 64, 128}});

        auto w0 = m1.add_literal(
            migraphx::generate_literal({migraphx::shape::float_type, {1, 128, 64}}, 0));
        auto w1 = m1.add_literal(
            migraphx::generate_literal({migraphx::shape::float_type, {1, 128, 64}}, 1));
        auto w2 = m1.add_literal(
            migraphx::generate_literal({migraphx::shape::float_type, {1, 128, 64}}, 2));
        auto w3 = m1.add_literal(
            migraphx::generate_literal({migraphx::shape::float_type, {1, 128, 64}}, 3));

        auto d0 = m1.add_instruction(migraphx::make_op("dot"), input, w0);
        auto d1 = m1.add_instruction(migraphx::make_op("dot"), input, w1);
        auto d2 = m1.add_instruction(migraphx::make_op("dot"), input, w2);
        auto d3 = m1.add_instruction(migraphx::make_op("dot"), input, w3);

        m1.add_instruction(migraphx::make_op("concat", {{"axis", 2}}),
                           std::vector<migraphx::instruction_ref>{d0, d1, d2, d3});
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto input = m2.add_parameter("input", {migraphx::shape::float_type, {1, 64, 128}});

        auto w0 = m2.add_literal(
            migraphx::generate_literal({migraphx::shape::float_type, {1, 128, 64}}, 0));
        auto w1 = m2.add_literal(
            migraphx::generate_literal({migraphx::shape::float_type, {1, 128, 64}}, 1));
        auto w2 = m2.add_literal(
            migraphx::generate_literal({migraphx::shape::float_type, {1, 128, 64}}, 2));
        auto w3 = m2.add_literal(
            migraphx::generate_literal({migraphx::shape::float_type, {1, 128, 64}}, 3));

        // Unsqueeze each A input
        auto a0 = m2.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0}}}), input);
        auto a1 = m2.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0}}}), input);
        auto a2 = m2.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0}}}), input);
        auto a3 = m2.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0}}}), input);

        // Unsqueeze each B weight
        auto b0 = m2.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0}}}), w0);
        auto b1 = m2.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0}}}), w1);
        auto b2 = m2.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0}}}), w2);
        auto b3 = m2.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0}}}), w3);

        auto stacked_a = m2.add_instruction(
            migraphx::make_op("concat", {{"axis", 0}}),
            std::vector<migraphx::instruction_ref>{a0, a1, a2, a3});
        auto stacked_b = m2.add_instruction(
            migraphx::make_op("concat", {{"axis", 0}}),
            std::vector<migraphx::instruction_ref>{b0, b1, b2, b3});

        auto batched = m2.add_instruction(migraphx::make_op("dot"), stacked_a, stacked_b);

        auto s0 = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {0}}, {"ends", {1}}}), batched);
        auto s1 = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {1}}, {"ends", {2}}}), batched);
        auto s2 = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {2}}, {"ends", {3}}}), batched);
        auto s3 = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {3}}, {"ends", {4}}}), batched);

        auto sq0 = m2.add_instruction(migraphx::make_op("squeeze", {{"axes", {0}}}), s0);
        auto sq1 = m2.add_instruction(migraphx::make_op("squeeze", {{"axes", {0}}}), s1);
        auto sq2 = m2.add_instruction(migraphx::make_op("squeeze", {{"axes", {0}}}), s2);
        auto sq3 = m2.add_instruction(migraphx::make_op("squeeze", {{"axes", {0}}}), s3);

        m2.add_instruction(migraphx::make_op("concat", {{"axis", 2}}),
                           std::vector<migraphx::instruction_ref>{sq0, sq1, sq2, sq3});
    }
    EXPECT(m1 == m2);
}

// Dots with non-constant weights → no fusion
TEST_CASE(dot_horiz_no_fusion_non_constant_weights)
{
    migraphx::module m1;
    {
        auto input = m1.add_parameter("input", {migraphx::shape::float_type, {1, 64, 128}});
        auto w0    = m1.add_parameter("w0", {migraphx::shape::float_type, {1, 128, 64}});
        auto w1    = m1.add_parameter("w1", {migraphx::shape::float_type, {1, 128, 64}});

        auto d0 = m1.add_instruction(migraphx::make_op("dot"), input, w0);
        auto d1 = m1.add_instruction(migraphx::make_op("dot"), input, w1);

        m1.add_instruction(migraphx::make_op("concat", {{"axis", 2}}),
                           std::vector<migraphx::instruction_ref>{d0, d1});
    }
    auto m2 = m1;
    run_pass(m1);
    EXPECT(m1 == m2);
}

// Dots with different output shapes → separate groups, no fusion (below threshold)
TEST_CASE(dot_horiz_no_fusion_different_shapes)
{
    migraphx::module m1;
    {
        auto input = m1.add_parameter("input", {migraphx::shape::float_type, {1, 64, 128}});
        auto w0    = m1.add_literal(
            migraphx::generate_literal({migraphx::shape::float_type, {1, 128, 64}}, 0));
        auto w1 = m1.add_literal(
            migraphx::generate_literal({migraphx::shape::float_type, {1, 128, 32}}, 1));

        auto d0 = m1.add_instruction(migraphx::make_op("dot"), input, w0);
        auto d1 = m1.add_instruction(migraphx::make_op("dot"), input, w1);

        auto r0 = m1.add_instruction(migraphx::make_op("relu"), d0);
        auto r1 = m1.add_instruction(migraphx::make_op("relu"), d1);

        m1.add_instruction(migraphx::make_op("concat", {{"axis", 2}}),
                           std::vector<migraphx::instruction_ref>{r0, r1});
    }
    auto m2 = m1;
    run_pass(m1);
    EXPECT(m1 == m2);
}

// Dots with add consumers → NOT fused by dot_horizontal (MLIR handles them better)
TEST_CASE(dot_horiz_no_fusion_add_consumer)
{
    migraphx::module m1;
    {
        auto input = m1.add_parameter("input", {migraphx::shape::float_type, {1, 64, 128}});

        auto w0 = m1.add_literal(
            migraphx::generate_literal({migraphx::shape::float_type, {1, 128, 64}}, 0));
        auto w1 = m1.add_literal(
            migraphx::generate_literal({migraphx::shape::float_type, {1, 128, 64}}, 1));
        auto b0 = m1.add_literal(
            migraphx::generate_literal({migraphx::shape::float_type, {1, 64, 64}}, 10));
        auto b1 = m1.add_literal(
            migraphx::generate_literal({migraphx::shape::float_type, {1, 64, 64}}, 11));

        auto d0 = m1.add_instruction(migraphx::make_op("dot"), input, w0);
        auto d1 = m1.add_instruction(migraphx::make_op("dot"), input, w1);
        auto a0 = m1.add_instruction(migraphx::make_op("add"), d0, b0);
        auto a1 = m1.add_instruction(migraphx::make_op("add"), d1, b1);

        m1.add_instruction(migraphx::make_op("concat", {{"axis", 2}}),
                           std::vector<migraphx::instruction_ref>{a0, a1});
    }
    auto m2 = m1;
    run_pass(m1);
    EXPECT(m1 == m2);
}

// ===========================================================================
// Expert head (SwiGLU + dot + add) horizontal fusion tests
// ===========================================================================

// 4 expert heads with SwiGLU pattern → fuse into batched sigmoid+mul+dot+add
TEST_CASE(expert_head_fusion_basic)
{
    migraphx::module m1;
    {
        auto input = m1.add_parameter("input", {migraphx::shape::float_type, {1, 64, 512}});

        auto s0 = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {2}}, {"starts", {0}}, {"ends", {128}}}), input);
        auto s1 = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {2}}, {"starts", {128}}, {"ends", {256}}}), input);
        auto s2 = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {2}}, {"starts", {256}}, {"ends", {384}}}), input);
        auto s3 = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {2}}, {"starts", {384}}, {"ends", {512}}}), input);

        auto w0 = m1.add_literal(
            migraphx::generate_literal({migraphx::shape::float_type, {1, 128, 128}}, 0));
        auto w1 = m1.add_literal(
            migraphx::generate_literal({migraphx::shape::float_type, {1, 128, 128}}, 1));
        auto w2 = m1.add_literal(
            migraphx::generate_literal({migraphx::shape::float_type, {1, 128, 128}}, 2));
        auto w3 = m1.add_literal(
            migraphx::generate_literal({migraphx::shape::float_type, {1, 128, 128}}, 3));

        auto b0 = m1.add_literal(
            migraphx::generate_literal({migraphx::shape::float_type, {1, 64, 128}}, 10));
        auto b1 = m1.add_literal(
            migraphx::generate_literal({migraphx::shape::float_type, {1, 64, 128}}, 11));
        auto b2 = m1.add_literal(
            migraphx::generate_literal({migraphx::shape::float_type, {1, 64, 128}}, 12));
        auto b3 = m1.add_literal(
            migraphx::generate_literal({migraphx::shape::float_type, {1, 64, 128}}, 13));

        // SwiGLU: sigmoid(x) * x
        auto sig0 = m1.add_instruction(migraphx::make_op("sigmoid"), s0);
        auto mul0 = m1.add_instruction(migraphx::make_op("mul"), s0, sig0);
        auto sig1 = m1.add_instruction(migraphx::make_op("sigmoid"), s1);
        auto mul1 = m1.add_instruction(migraphx::make_op("mul"), s1, sig1);
        auto sig2 = m1.add_instruction(migraphx::make_op("sigmoid"), s2);
        auto mul2 = m1.add_instruction(migraphx::make_op("mul"), s2, sig2);
        auto sig3 = m1.add_instruction(migraphx::make_op("sigmoid"), s3);
        auto mul3 = m1.add_instruction(migraphx::make_op("mul"), s3, sig3);

        auto d0 = m1.add_instruction(migraphx::make_op("dot"), mul0, w0);
        auto d1 = m1.add_instruction(migraphx::make_op("dot"), mul1, w1);
        auto d2 = m1.add_instruction(migraphx::make_op("dot"), mul2, w2);
        auto d3 = m1.add_instruction(migraphx::make_op("dot"), mul3, w3);

        auto a0 = m1.add_instruction(migraphx::make_op("add"), d0, b0);
        auto a1 = m1.add_instruction(migraphx::make_op("add"), d1, b1);
        auto a2 = m1.add_instruction(migraphx::make_op("add"), d2, b2);
        auto a3 = m1.add_instruction(migraphx::make_op("add"), d3, b3);

        m1.add_instruction(migraphx::make_op("concat", {{"axis", 2}}),
                           std::vector<migraphx::instruction_ref>{a0, a1, a2, a3});
    }
    run_pass(m1);

    auto dot_count = std::count_if(m1.begin(), m1.end(), [](const auto& ins) {
        return ins.name() == "dot";
    });
    auto sigmoid_count = std::count_if(m1.begin(), m1.end(), [](const auto& ins) {
        return ins.name() == "sigmoid";
    });
    auto add_count = std::count_if(m1.begin(), m1.end(), [](const auto& ins) {
        return ins.name() == "add";
    });
    EXPECT(dot_count == 1);
    EXPECT(sigmoid_count == 1);
    EXPECT(add_count == 1);
}

// Non-SwiGLU dots with add should NOT trigger expert head fusion
TEST_CASE(expert_head_no_fusion_missing_sigmoid)
{
    migraphx::module m1;
    {
        auto input = m1.add_parameter("input", {migraphx::shape::float_type, {1, 64, 128}});

        auto w0 = m1.add_literal(
            migraphx::generate_literal({migraphx::shape::float_type, {1, 128, 64}}, 0));
        auto w1 = m1.add_literal(
            migraphx::generate_literal({migraphx::shape::float_type, {1, 128, 64}}, 1));
        auto w2 = m1.add_literal(
            migraphx::generate_literal({migraphx::shape::float_type, {1, 128, 64}}, 2));
        auto w3 = m1.add_literal(
            migraphx::generate_literal({migraphx::shape::float_type, {1, 128, 64}}, 3));

        auto b0 = m1.add_literal(
            migraphx::generate_literal({migraphx::shape::float_type, {1, 64, 64}}, 10));
        auto b1 = m1.add_literal(
            migraphx::generate_literal({migraphx::shape::float_type, {1, 64, 64}}, 11));
        auto b2 = m1.add_literal(
            migraphx::generate_literal({migraphx::shape::float_type, {1, 64, 64}}, 12));
        auto b3 = m1.add_literal(
            migraphx::generate_literal({migraphx::shape::float_type, {1, 64, 64}}, 13));

        // Plain dot + add (no SwiGLU) → should not trigger expert_head fusion
        // and dot_horizontal is blocked by add consumer
        auto d0 = m1.add_instruction(migraphx::make_op("dot"), input, w0);
        auto d1 = m1.add_instruction(migraphx::make_op("dot"), input, w1);
        auto d2 = m1.add_instruction(migraphx::make_op("dot"), input, w2);
        auto d3 = m1.add_instruction(migraphx::make_op("dot"), input, w3);

        auto a0 = m1.add_instruction(migraphx::make_op("add"), d0, b0);
        auto a1 = m1.add_instruction(migraphx::make_op("add"), d1, b1);
        auto a2 = m1.add_instruction(migraphx::make_op("add"), d2, b2);
        auto a3 = m1.add_instruction(migraphx::make_op("add"), d3, b3);

        m1.add_instruction(migraphx::make_op("concat", {{"axis", 2}}),
                           std::vector<migraphx::instruction_ref>{a0, a1, a2, a3});
    }
    auto m2 = m1;
    run_pass(m1);
    EXPECT(m1 == m2);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
