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
#include <migraphx/fuse_pointwise.hpp>
#include <migraphx/simplify_algebra.hpp>
#include <migraphx/simplify_reshapes.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/program.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/generate.hpp>
#include <basic_ops.hpp>
#include <test.hpp>
#include <string>
#include <vector>

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

// Dependent gathers: g2 depends on g1, so it lands in its own subgroup. The remaining
// independent subgroup {g1, g3, g4} is below min_group_size=4, so nothing fuses.
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

TEST_CASE(same_table_gathers_basic)
{
    migraphx::module m1;
    {
        auto emb =
            m1.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {6, 2}}, 0));

        auto idx1 = m1.add_parameter("idx1", {migraphx::shape::int32_type, {4}});
        auto idx2 = m1.add_parameter("idx2", {migraphx::shape::int32_type, {5}});
        auto idx3 = m1.add_parameter("idx3", {migraphx::shape::int32_type, {4}});
        auto idx4 = m1.add_parameter("idx4", {migraphx::shape::int32_type, {6}});

        auto g1 = m1.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), emb, idx1);
        auto g2 = m1.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), emb, idx2);
        auto g3 = m1.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), emb, idx3);
        auto g4 = m1.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), emb, idx4);

        m1.add_return({g1, g2, g3, g4});
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto emb =
            m2.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {6, 2}}, 0));

        auto idx1 = m2.add_parameter("idx1", {migraphx::shape::int32_type, {4}});
        auto idx2 = m2.add_parameter("idx2", {migraphx::shape::int32_type, {5}});
        auto idx3 = m2.add_parameter("idx3", {migraphx::shape::int32_type, {4}});
        auto idx4 = m2.add_parameter("idx4", {migraphx::shape::int32_type, {6}});

        auto concat_idx =
            m2.add_instruction(migraphx::make_op("concat", {{"axis", 0}}),
                               std::vector<migraphx::instruction_ref>{idx1, idx2, idx3, idx4});

        auto bg = m2.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), emb, concat_idx);

        auto s1 = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {0}}, {"ends", {4}}}), bg);
        auto s2 = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {4}}, {"ends", {9}}}), bg);
        auto s3 = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {9}}, {"ends", {13}}}), bg);
        auto s4 = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {13}}, {"ends", {19}}}), bg);

        m2.add_return({s1, s2, s3, s4});
    }
    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(same_table_gathers_two_siblings)
{
    migraphx::module m1;
    {
        auto emb =
            m1.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {4, 3}}, 0));

        auto idx1 = m1.add_parameter("idx1", {migraphx::shape::int32_type, {4}});
        auto idx2 = m1.add_parameter("idx2", {migraphx::shape::int32_type, {5}});

        auto g1 = m1.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), emb, idx1);
        auto g2 = m1.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), emb, idx2);

        m1.add_return({g1, g2});
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto emb =
            m2.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {4, 3}}, 0));

        auto idx1 = m2.add_parameter("idx1", {migraphx::shape::int32_type, {4}});
        auto idx2 = m2.add_parameter("idx2", {migraphx::shape::int32_type, {5}});

        auto concat_idx = m2.add_instruction(migraphx::make_op("concat", {{"axis", 0}}),
                                             std::vector<migraphx::instruction_ref>{idx1, idx2});

        auto bg = m2.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), emb, concat_idx);

        auto s1 = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {0}}, {"ends", {4}}}), bg);
        auto s2 = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {4}}, {"ends", {9}}}), bg);

        m2.add_return({s1, s2});
    }
    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(same_table_gathers_single_no_rewrite)
{
    migraphx::module m1;
    {
        auto emb =
            m1.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {3, 4}}, 0));
        auto idx = m1.add_parameter("idx", {migraphx::shape::int32_type, {4}});
        auto g   = m1.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), emb, idx);
        m1.add_return({g});
    }
    auto m2 = m1;
    run_pass(m1);
    EXPECT(m1 == m2);
}

TEST_CASE(same_table_gathers_shared_index)
{
    migraphx::module m1;
    {
        auto emb =
            m1.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {5, 2}}, 0));
        auto idx = m1.add_parameter("idx", {migraphx::shape::int32_type, {4}});

        auto g1 = m1.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), emb, idx);
        auto g2 = m1.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), emb, idx);
        auto g3 = m1.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), emb, idx);

        m1.add_return({g1, g2, g3});
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto emb =
            m2.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {5, 2}}, 0));
        auto idx = m2.add_parameter("idx", {migraphx::shape::int32_type, {4}});

        auto concat_idx = m2.add_instruction(migraphx::make_op("concat", {{"axis", 0}}),
                                             std::vector<migraphx::instruction_ref>{idx, idx, idx});

        auto bg = m2.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), emb, concat_idx);

        auto s1 = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {0}}, {"ends", {4}}}), bg);
        auto s2 = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {4}}, {"ends", {8}}}), bg);
        auto s3 = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {8}}, {"ends", {12}}}), bg);

        m2.add_return({s1, s2, s3});
    }
    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(same_table_gathers_2d_indices)
{
    migraphx::module m1;
    {
        auto emb =
            m1.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {7, 2}}, 0));

        auto idx1 = m1.add_parameter("idx1", {migraphx::shape::int32_type, {4, 3}});
        auto idx2 = m1.add_parameter("idx2", {migraphx::shape::int32_type, {5, 3}});
        auto idx3 = m1.add_parameter("idx3", {migraphx::shape::int32_type, {6, 3}});

        auto g1 = m1.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), emb, idx1);
        auto g2 = m1.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), emb, idx2);
        auto g3 = m1.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), emb, idx3);

        m1.add_return({g1, g2, g3});
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto emb =
            m2.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {7, 2}}, 0));

        auto idx1 = m2.add_parameter("idx1", {migraphx::shape::int32_type, {4, 3}});
        auto idx2 = m2.add_parameter("idx2", {migraphx::shape::int32_type, {5, 3}});
        auto idx3 = m2.add_parameter("idx3", {migraphx::shape::int32_type, {6, 3}});

        auto concat_idx =
            m2.add_instruction(migraphx::make_op("concat", {{"axis", 0}}),
                               std::vector<migraphx::instruction_ref>{idx1, idx2, idx3});

        auto bg = m2.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), emb, concat_idx);

        auto s1 = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {0}}, {"ends", {4}}}), bg);
        auto s2 = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {4}}, {"ends", {9}}}), bg);
        auto s3 = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {9}}, {"ends", {15}}}), bg);

        m2.add_return({s1, s2, s3});
    }
    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(same_table_gathers_split_by_idx_type)
{
    migraphx::module m1;
    {
        auto emb =
            m1.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {4, 2}}, 0));

        auto idx32a = m1.add_parameter("idx32a", {migraphx::shape::int32_type, {4}});
        auto idx32b = m1.add_parameter("idx32b", {migraphx::shape::int32_type, {5}});
        auto idx64  = m1.add_parameter("idx64", {migraphx::shape::int64_type, {4}});

        auto g_a = m1.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), emb, idx32a);
        auto g_b = m1.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), emb, idx32b);
        auto g_c = m1.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), emb, idx64);

        m1.add_return({g_a, g_b, g_c});
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto emb =
            m2.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {4, 2}}, 0));

        auto idx32a = m2.add_parameter("idx32a", {migraphx::shape::int32_type, {4}});
        auto idx32b = m2.add_parameter("idx32b", {migraphx::shape::int32_type, {5}});
        auto idx64  = m2.add_parameter("idx64", {migraphx::shape::int64_type, {4}});

        auto concat_idx =
            m2.add_instruction(migraphx::make_op("concat", {{"axis", 0}}),
                               std::vector<migraphx::instruction_ref>{idx32a, idx32b});

        auto bg = m2.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), emb, concat_idx);

        auto s_a = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {0}}, {"ends", {4}}}), bg);
        auto s_b = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {4}}, {"ends", {9}}}), bg);

        // Lone int64 gather is left alone (group size = 1)
        auto g_c = m2.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), emb, idx64);

        m2.add_return({s_a, s_b, g_c});
    }
    EXPECT(m1.sort() == m2.sort());
}

// Same-table gathers with differing index shapes (1-D and 2-D) now merge into a single
// batched gather via the flattened path (indices flattened to 1-D, gathered, reshaped back).
TEST_CASE(same_table_gathers_mixed_index_shapes)
{
    migraphx::module m1;
    {
        auto emb =
            m1.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {4, 2}}, 0));

        auto idx_1d_a = m1.add_parameter("idx_1d_a", {migraphx::shape::int32_type, {4}});
        auto idx_1d_b = m1.add_parameter("idx_1d_b", {migraphx::shape::int32_type, {5}});
        auto idx_2d_a = m1.add_parameter("idx_2d_a", {migraphx::shape::int32_type, {4, 3}});
        auto idx_2d_b = m1.add_parameter("idx_2d_b", {migraphx::shape::int32_type, {5, 3}});

        auto g1 = m1.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), emb, idx_1d_a);
        auto g2 = m1.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), emb, idx_1d_b);
        auto g3 = m1.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), emb, idx_2d_a);
        auto g4 = m1.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), emb, idx_2d_b);

        m1.add_return({g1, g2, g3, g4});
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto emb =
            m2.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {4, 2}}, 0));

        auto idx_1d_a = m2.add_parameter("idx_1d_a", {migraphx::shape::int32_type, {4}});
        auto idx_1d_b = m2.add_parameter("idx_1d_b", {migraphx::shape::int32_type, {5}});
        auto idx_2d_a = m2.add_parameter("idx_2d_a", {migraphx::shape::int32_type, {4, 3}});
        auto idx_2d_b = m2.add_parameter("idx_2d_b", {migraphx::shape::int32_type, {5, 3}});

        // All four gathers share the table but have mixed index ranks (1-D and 2-D), so they
        // cannot be concatenated on axis 0.  They fuse via the flattened path: each 2-D index
        // is reshaped to 1-D, all are concatenated, one batched gather runs, and each range is
        // sliced back out and reshaped to its original output shape.  Element counts are
        // 4, 5, 12, 15 -> ends 4, 9, 21, 36.
        auto f3 = m2.add_instruction(migraphx::make_op("reshape", {{"dims", {12}}}), idx_2d_a);
        auto f4 = m2.add_instruction(migraphx::make_op("reshape", {{"dims", {15}}}), idx_2d_b);

        auto big_idx =
            m2.add_instruction(migraphx::make_op("concat", {{"axis", 0}}),
                               std::vector<migraphx::instruction_ref>{idx_1d_a, idx_1d_b, f3, f4});

        auto bg = m2.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), emb, big_idx);

        auto s1 = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {0}}, {"ends", {4}}}), bg);
        auto s2 = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {4}}, {"ends", {9}}}), bg);
        auto s3 = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {9}}, {"ends", {21}}}), bg);
        auto r3 = m2.add_instruction(migraphx::make_op("reshape", {{"dims", {4, 3, 2}}}), s3);
        auto s4 = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {21}}, {"ends", {36}}}), bg);
        auto r4 = m2.add_instruction(migraphx::make_op("reshape", {{"dims", {5, 3, 2}}}), s4);

        m2.add_return({s1, s2, r3, r4});
    }
    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(same_table_gathers_multiple_tables)
{
    migraphx::module m1;
    {
        auto emb_a =
            m1.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {3, 2}}, 0));
        auto emb_b =
            m1.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {4, 2}}, 1));

        auto idx_a1 = m1.add_parameter("idx_a1", {migraphx::shape::int32_type, {4}});
        auto idx_a2 = m1.add_parameter("idx_a2", {migraphx::shape::int32_type, {5}});
        auto idx_b1 = m1.add_parameter("idx_b1", {migraphx::shape::int32_type, {4}});
        auto idx_b2 = m1.add_parameter("idx_b2", {migraphx::shape::int32_type, {5}});

        auto ga1 = m1.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), emb_a, idx_a1);
        auto gb1 = m1.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), emb_b, idx_b1);
        auto ga2 = m1.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), emb_a, idx_a2);
        auto gb2 = m1.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), emb_b, idx_b2);

        m1.add_return({ga1, gb1, ga2, gb2});
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto emb_a =
            m2.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {3, 2}}, 0));
        auto emb_b =
            m2.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {4, 2}}, 1));

        auto idx_a1 = m2.add_parameter("idx_a1", {migraphx::shape::int32_type, {4}});
        auto idx_a2 = m2.add_parameter("idx_a2", {migraphx::shape::int32_type, {5}});
        auto idx_b1 = m2.add_parameter("idx_b1", {migraphx::shape::int32_type, {4}});
        auto idx_b2 = m2.add_parameter("idx_b2", {migraphx::shape::int32_type, {5}});

        // Cross-embedding fusion runs first and, because both tables share the same
        // embedding dim, bundles all four gathers into one batched gather.  Table dedup keeps
        // each table once: concat is [3+4, 2] = [7, 2], and only table B's indices are shifted
        // by +3 (table A's offset is 0).
        auto concat_emb = m2.add_instruction(migraphx::make_op("concat", {{"axis", 0}}),
                                             std::vector<migraphx::instruction_ref>{emb_a, emb_b});

        auto offb1 = m2.add_literal(
            migraphx::literal{migraphx::shape{migraphx::shape::int32_type}, {std::size_t(3)}});
        auto bcb1 =
            m2.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {4}}}), offb1);
        auto adj_b1 = m2.add_instruction(migraphx::make_op("add"), idx_b1, bcb1);

        auto offb2 = m2.add_literal(
            migraphx::literal{migraphx::shape{migraphx::shape::int32_type}, {std::size_t(3)}});
        auto bcb2 =
            m2.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {5}}}), offb2);
        auto adj_b2 = m2.add_instruction(migraphx::make_op("add"), idx_b2, bcb2);

        // Indices concatenated in gather (position) order: ga1, gb1, ga2, gb2.
        auto concat_idx = m2.add_instruction(
            migraphx::make_op("concat", {{"axis", 0}}),
            std::vector<migraphx::instruction_ref>{idx_a1, adj_b1, idx_a2, adj_b2});

        auto bg =
            m2.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), concat_emb, concat_idx);

        auto sa1 = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {0}}, {"ends", {4}}}), bg);
        auto sb1 = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {4}}, {"ends", {8}}}), bg);
        auto sa2 = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {8}}, {"ends", {13}}}), bg);
        auto sb2 = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {13}}, {"ends", {18}}}), bg);

        m2.add_return({sa1, sb1, sa2, sb2});
    }
    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(same_table_gathers_no_rewrite_non_constant_data)
{
    migraphx::module m1;
    {
        auto emb  = m1.add_parameter("emb", {migraphx::shape::float_type, {6, 2}});
        auto idx1 = m1.add_parameter("idx1", {migraphx::shape::int32_type, {2}});
        auto idx2 = m1.add_parameter("idx2", {migraphx::shape::int32_type, {3}});
        auto idx3 = m1.add_parameter("idx3", {migraphx::shape::int32_type, {1}});

        auto g1 = m1.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), emb, idx1);
        auto g2 = m1.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), emb, idx2);
        auto g3 = m1.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), emb, idx3);

        m1.add_instruction(migraphx::make_op("concat", {{"axis", 0}}),
                           std::vector<migraphx::instruction_ref>{g1, g2, g3});
    }
    auto m2 = m1;
    run_pass(m1);
    EXPECT(m1 == m2);
}

TEST_CASE(same_table_gathers_no_rewrite_1d_data)
{
    migraphx::module m1;
    {
        auto emb =
            m1.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {6}}, 0));

        auto idx1 = m1.add_parameter("idx1", {migraphx::shape::int32_type, {2}});
        auto idx2 = m1.add_parameter("idx2", {migraphx::shape::int32_type, {3}});
        auto idx3 = m1.add_parameter("idx3", {migraphx::shape::int32_type, {1}});

        auto g1 = m1.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), emb, idx1);
        auto g2 = m1.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), emb, idx2);
        auto g3 = m1.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), emb, idx3);

        m1.add_instruction(migraphx::make_op("concat", {{"axis", 0}}),
                           std::vector<migraphx::instruction_ref>{g1, g2, g3});
    }
    auto m2 = m1;
    run_pass(m1);
    EXPECT(m1 == m2);
}

TEST_CASE(same_table_gathers_no_rewrite_3d_data)
{
    migraphx::module m1;
    {
        auto emb =
            m1.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {2, 3, 4}}, 0));

        auto idx1 = m1.add_parameter("idx1", {migraphx::shape::int32_type, {1}});
        auto idx2 = m1.add_parameter("idx2", {migraphx::shape::int32_type, {1}});

        auto g1 = m1.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), emb, idx1);
        auto g2 = m1.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), emb, idx2);

        m1.add_instruction(migraphx::make_op("concat", {{"axis", 0}}),
                           std::vector<migraphx::instruction_ref>{g1, g2});
    }
    auto m2 = m1;
    run_pass(m1);
    EXPECT(m1 == m2);
}

TEST_CASE(same_table_gathers_no_rewrite_axis_one)
{
    migraphx::module m1;
    {
        auto emb =
            m1.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {3, 6}}, 0));

        auto idx1 = m1.add_parameter("idx1", {migraphx::shape::int32_type, {2}});
        auto idx2 = m1.add_parameter("idx2", {migraphx::shape::int32_type, {2}});
        auto idx3 = m1.add_parameter("idx3", {migraphx::shape::int32_type, {2}});

        auto g1 = m1.add_instruction(migraphx::make_op("gather", {{"axis", 1}}), emb, idx1);
        auto g2 = m1.add_instruction(migraphx::make_op("gather", {{"axis", 1}}), emb, idx2);
        auto g3 = m1.add_instruction(migraphx::make_op("gather", {{"axis", 1}}), emb, idx3);

        m1.add_instruction(migraphx::make_op("concat", {{"axis", 0}}),
                           std::vector<migraphx::instruction_ref>{g1, g2, g3});
    }
    auto m2 = m1;
    run_pass(m1);
    EXPECT(m1 == m2);
}

TEST_CASE(same_table_gathers_no_rewrite_scalar_index)
{
    migraphx::module m1;
    {
        auto emb =
            m1.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {6, 2}}, 0));

        auto idx1 = m1.add_literal(
            migraphx::literal{migraphx::shape{migraphx::shape::int32_type}, {std::int32_t(0)}});
        auto idx2 = m1.add_literal(
            migraphx::literal{migraphx::shape{migraphx::shape::int32_type}, {std::int32_t(1)}});
        auto idx3 = m1.add_literal(
            migraphx::literal{migraphx::shape{migraphx::shape::int32_type}, {std::int32_t(2)}});

        auto g1 = m1.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), emb, idx1);
        auto g2 = m1.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), emb, idx2);
        auto g3 = m1.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), emb, idx3);

        auto sum1 = m1.add_instruction(migraphx::make_op("add"), g1, g2);
        auto sum2 = m1.add_instruction(migraphx::make_op("add"), sum1, g3);
        m1.add_return({sum2});
    }
    auto m2 = m1;
    run_pass(m1);
    EXPECT(m1 == m2);
}

TEST_CASE(same_table_gathers_no_rewrite_small_batch)
{
    migraphx::module m1;
    {
        auto emb =
            m1.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {6, 2}}, 0));

        auto idx1 = m1.add_parameter("idx1", {migraphx::shape::int32_type, {2}});
        auto idx2 = m1.add_parameter("idx2", {migraphx::shape::int32_type, {3}});
        auto idx3 = m1.add_parameter("idx3", {migraphx::shape::int32_type, {2}});

        auto g1 = m1.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), emb, idx1);
        auto g2 = m1.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), emb, idx2);
        auto g3 = m1.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), emb, idx3);

        m1.add_return({g1, g2, g3});
    }
    auto m2 = m1;
    run_pass(m1);
    EXPECT(m1 == m2);
}

TEST_CASE(same_table_gathers_no_rewrite_mixed_batch_single_eligible)
{
    migraphx::module m1;
    {
        auto emb =
            m1.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {6, 2}}, 0));

        auto idx1 = m1.add_parameter("idx1", {migraphx::shape::int32_type, {2}});
        auto idx2 = m1.add_parameter("idx2", {migraphx::shape::int32_type, {3}});
        auto idx3 = m1.add_parameter("idx3", {migraphx::shape::int32_type, {4}});

        auto g1 = m1.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), emb, idx1);
        auto g2 = m1.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), emb, idx2);
        auto g3 = m1.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), emb, idx3);

        m1.add_return({g1, g2, g3});
    }
    auto m2 = m1;
    run_pass(m1);
    EXPECT(m1 == m2);
}

TEST_CASE(same_table_gathers_idempotent)
{
    migraphx::module m1;
    {
        auto emb =
            m1.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {6, 2}}, 0));

        auto idx1 = m1.add_parameter("idx1", {migraphx::shape::int32_type, {4}});
        auto idx2 = m1.add_parameter("idx2", {migraphx::shape::int32_type, {5}});
        auto idx3 = m1.add_parameter("idx3", {migraphx::shape::int32_type, {6}});

        auto g1 = m1.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), emb, idx1);
        auto g2 = m1.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), emb, idx2);
        auto g3 = m1.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), emb, idx3);

        m1.add_return({g1, g2, g3});
    }
    run_pass(m1);
    auto snapshot = m1;
    run_pass(m1);
    EXPECT(m1.sort() == snapshot.sort());
}

// Same-table gathers with dynamic-shaped indices → no fusion.
TEST_CASE(same_table_gathers_no_rewrite_dynamic_index)
{
    using dd = migraphx::shape::dynamic_dimension;
    migraphx::shape idx_s{migraphx::shape::int32_type, {dd{4, 8}}};

    migraphx::module m1;
    {
        auto emb =
            m1.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {6, 2}}, 0));

        auto idx1 = m1.add_parameter("idx1", idx_s);
        auto idx2 = m1.add_parameter("idx2", idx_s);
        auto idx3 = m1.add_parameter("idx3", idx_s);

        auto g1 = m1.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), emb, idx1);
        auto g2 = m1.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), emb, idx2);
        auto g3 = m1.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), emb, idx3);

        m1.add_return({g1, g2, g3});
    }
    auto m2 = m1;
    run_pass(m1);
    EXPECT(m1 == m2);
}

// Cross-embedding fusion candidates with dynamic-shaped indices → no fusion.
TEST_CASE(gather_horiz_no_fusion_dynamic_index)
{
    using dd = migraphx::shape::dynamic_dimension;
    migraphx::shape idx_s{migraphx::shape::int32_type, {dd{1, 8}}};

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

        auto idx1 = m1.add_parameter("idx1", idx_s);
        auto idx2 = m1.add_parameter("idx2", idx_s);
        auto idx3 = m1.add_parameter("idx3", idx_s);
        auto idx4 = m1.add_parameter("idx4", idx_s);

        auto g1 = m1.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), emb1, idx1);
        auto g2 = m1.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), emb2, idx2);
        auto g3 = m1.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), emb3, idx3);
        auto g4 = m1.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), emb4, idx4);

        m1.add_return({g1, g2, g3, g4});
    }
    auto m2 = m1;
    run_pass(m1);
    EXPECT(m1 == m2);
}

// Independent dots with identical activation/weight shapes and constant
// weights should batch into a single GEMM, then slice+squeeze back.
TEST_CASE(dot_horiz_fusion_basic)
{
    migraphx::module m1;
    {
        auto a0 = m1.add_parameter("a0", {migraphx::shape::float_type, {2, 3}});
        auto a1 = m1.add_parameter("a1", {migraphx::shape::float_type, {2, 3}});
        auto a2 = m1.add_parameter("a2", {migraphx::shape::float_type, {2, 3}});
        auto w0 =
            m1.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {3, 4}}, 0));
        auto w1 =
            m1.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {3, 4}}, 1));
        auto w2 =
            m1.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {3, 4}}, 2));
        auto d0 = m1.add_instruction(migraphx::make_op("dot"), a0, w0);
        auto d1 = m1.add_instruction(migraphx::make_op("dot"), a1, w1);
        auto d2 = m1.add_instruction(migraphx::make_op("dot"), a2, w2);
        m1.add_return({d0, d1, d2});
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto a0 = m2.add_parameter("a0", {migraphx::shape::float_type, {2, 3}});
        auto a1 = m2.add_parameter("a1", {migraphx::shape::float_type, {2, 3}});
        auto a2 = m2.add_parameter("a2", {migraphx::shape::float_type, {2, 3}});
        auto w0 =
            m2.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {3, 4}}, 0));
        auto w1 =
            m2.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {3, 4}}, 1));
        auto w2 =
            m2.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {3, 4}}, 2));

        auto ua0  = m2.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0}}}), a0);
        auto ua1  = m2.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0}}}), a1);
        auto ua2  = m2.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0}}}), a2);
        auto bact = m2.add_instruction(migraphx::make_op("concat", {{"axis", 0}}), ua0, ua1, ua2);
        auto uw0  = m2.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0}}}), w0);
        auto uw1  = m2.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0}}}), w1);
        auto uw2  = m2.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0}}}), w2);
        auto bwt  = m2.add_instruction(migraphx::make_op("concat", {{"axis", 0}}), uw0, uw1, uw2);
        auto bd   = m2.add_instruction(migraphx::make_op("dot"), bact, bwt);

        auto s0 = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {0}}, {"ends", {1}}}), bd);
        auto sq0 = m2.add_instruction(migraphx::make_op("squeeze", {{"axes", {0}}}), s0);
        auto s1  = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {1}}, {"ends", {2}}}), bd);
        auto sq1 = m2.add_instruction(migraphx::make_op("squeeze", {{"axes", {0}}}), s1);
        auto s2  = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {2}}, {"ends", {3}}}), bd);
        auto sq2 = m2.add_instruction(migraphx::make_op("squeeze", {{"axes", {0}}}), s2);
        m2.add_return({sq0, sq1, sq2});
    }

    EXPECT(m1.sort() == m2.sort());
}

// Three parallel dot->add->dot chains share one group key. Dependent dots must not fuse
// together, but each level forms an independent subgroup that fuses on its own.
TEST_CASE(dot_horiz_fusion_chained_groups)
{
    migraphx::module m1;
    {
        auto x0 = m1.add_parameter("x0", {migraphx::shape::float_type, {2, 4}});
        auto x1 = m1.add_parameter("x1", {migraphx::shape::float_type, {2, 4}});
        auto x2 = m1.add_parameter("x2", {migraphx::shape::float_type, {2, 4}});
        auto b0 = m1.add_parameter("b0", {migraphx::shape::float_type, {2, 4}});
        auto b1 = m1.add_parameter("b1", {migraphx::shape::float_type, {2, 4}});
        auto b2 = m1.add_parameter("b2", {migraphx::shape::float_type, {2, 4}});
        auto w00 =
            m1.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {4, 4}}, 0));
        auto w01 =
            m1.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {4, 4}}, 1));
        auto w02 =
            m1.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {4, 4}}, 2));
        auto w10 =
            m1.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {4, 4}}, 3));
        auto w11 =
            m1.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {4, 4}}, 4));
        auto w12 =
            m1.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {4, 4}}, 5));
        auto d00 = m1.add_instruction(migraphx::make_op("dot"), x0, w00);
        auto a0  = m1.add_instruction(migraphx::make_op("add"), d00, b0);
        auto d10 = m1.add_instruction(migraphx::make_op("dot"), a0, w10);
        auto d01 = m1.add_instruction(migraphx::make_op("dot"), x1, w01);
        auto a1  = m1.add_instruction(migraphx::make_op("add"), d01, b1);
        auto d11 = m1.add_instruction(migraphx::make_op("dot"), a1, w11);
        auto d02 = m1.add_instruction(migraphx::make_op("dot"), x2, w02);
        auto a2  = m1.add_instruction(migraphx::make_op("add"), d02, b2);
        auto d12 = m1.add_instruction(migraphx::make_op("dot"), a2, w12);
        m1.add_return({d10, d11, d12});
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto x0 = m2.add_parameter("x0", {migraphx::shape::float_type, {2, 4}});
        auto x1 = m2.add_parameter("x1", {migraphx::shape::float_type, {2, 4}});
        auto x2 = m2.add_parameter("x2", {migraphx::shape::float_type, {2, 4}});
        auto b0 = m2.add_parameter("b0", {migraphx::shape::float_type, {2, 4}});
        auto b1 = m2.add_parameter("b1", {migraphx::shape::float_type, {2, 4}});
        auto b2 = m2.add_parameter("b2", {migraphx::shape::float_type, {2, 4}});
        auto w00 =
            m2.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {4, 4}}, 0));
        auto w01 =
            m2.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {4, 4}}, 1));
        auto w02 =
            m2.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {4, 4}}, 2));
        auto w10 =
            m2.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {4, 4}}, 3));
        auto w11 =
            m2.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {4, 4}}, 4));
        auto w12 =
            m2.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {4, 4}}, 5));

        auto ux0   = m2.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0}}}), x0);
        auto ux1   = m2.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0}}}), x1);
        auto ux2   = m2.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0}}}), x2);
        auto bact0 = m2.add_instruction(migraphx::make_op("concat", {{"axis", 0}}), ux0, ux1, ux2);
        auto uw00  = m2.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0}}}), w00);
        auto uw01  = m2.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0}}}), w01);
        auto uw02  = m2.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0}}}), w02);
        auto bwt0 =
            m2.add_instruction(migraphx::make_op("concat", {{"axis", 0}}), uw00, uw01, uw02);
        auto bd0 = m2.add_instruction(migraphx::make_op("dot"), bact0, bwt0);
        auto s00 = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {0}}, {"ends", {1}}}), bd0);
        auto sq00 = m2.add_instruction(migraphx::make_op("squeeze", {{"axes", {0}}}), s00);
        auto s01  = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {1}}, {"ends", {2}}}), bd0);
        auto sq01 = m2.add_instruction(migraphx::make_op("squeeze", {{"axes", {0}}}), s01);
        auto s02  = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {2}}, {"ends", {3}}}), bd0);
        auto sq02 = m2.add_instruction(migraphx::make_op("squeeze", {{"axes", {0}}}), s02);

        auto a0 = m2.add_instruction(migraphx::make_op("add"), sq00, b0);
        auto a1 = m2.add_instruction(migraphx::make_op("add"), sq01, b1);
        auto a2 = m2.add_instruction(migraphx::make_op("add"), sq02, b2);

        auto ua0   = m2.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0}}}), a0);
        auto ua1   = m2.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0}}}), a1);
        auto ua2   = m2.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0}}}), a2);
        auto bact1 = m2.add_instruction(migraphx::make_op("concat", {{"axis", 0}}), ua0, ua1, ua2);
        auto uw10  = m2.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0}}}), w10);
        auto uw11  = m2.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0}}}), w11);
        auto uw12  = m2.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0}}}), w12);
        auto bwt1 =
            m2.add_instruction(migraphx::make_op("concat", {{"axis", 0}}), uw10, uw11, uw12);
        auto bd1 = m2.add_instruction(migraphx::make_op("dot"), bact1, bwt1);
        auto s10 = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {0}}, {"ends", {1}}}), bd1);
        auto sq10 = m2.add_instruction(migraphx::make_op("squeeze", {{"axes", {0}}}), s10);
        auto s11  = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {1}}, {"ends", {2}}}), bd1);
        auto sq11 = m2.add_instruction(migraphx::make_op("squeeze", {{"axes", {0}}}), s11);
        auto s12  = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {2}}, {"ends", {3}}}), bd1);
        auto sq12 = m2.add_instruction(migraphx::make_op("squeeze", {{"axes", {0}}}), s12);
        m2.add_return({sq10, sq11, sq12});
    }

    EXPECT(m1.sort() == m2.sort());
}

// A dependent chain in the same key group does not block fusion: the chain's first dot is
// independent of the other candidates and fuses with them, while the downstream dot lands
// in its own subgroup below min_group_size.
TEST_CASE(dot_horiz_fusion_independent_subset)
{
    migraphx::module m1;
    {
        auto x  = m1.add_parameter("x", {migraphx::shape::float_type, {2, 4}});
        auto b  = m1.add_parameter("b", {migraphx::shape::float_type, {2, 4}});
        auto a0 = m1.add_parameter("a0", {migraphx::shape::float_type, {2, 4}});
        auto a1 = m1.add_parameter("a1", {migraphx::shape::float_type, {2, 4}});
        auto a2 = m1.add_parameter("a2", {migraphx::shape::float_type, {2, 4}});
        auto w0 =
            m1.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {4, 4}}, 0));
        auto w1 =
            m1.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {4, 4}}, 1));
        auto wi0 =
            m1.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {4, 4}}, 2));
        auto wi1 =
            m1.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {4, 4}}, 3));
        auto wi2 =
            m1.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {4, 4}}, 4));

        auto dependent0 = m1.add_instruction(migraphx::make_op("dot"), x, w0);
        auto add        = m1.add_instruction(migraphx::make_op("add"), dependent0, b);
        auto dependent1 = m1.add_instruction(migraphx::make_op("dot"), add, w1);
        auto d0         = m1.add_instruction(migraphx::make_op("dot"), a0, wi0);
        auto d1         = m1.add_instruction(migraphx::make_op("dot"), a1, wi1);
        auto d2         = m1.add_instruction(migraphx::make_op("dot"), a2, wi2);
        m1.add_return({dependent1, d0, d1, d2});
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto x  = m2.add_parameter("x", {migraphx::shape::float_type, {2, 4}});
        auto b  = m2.add_parameter("b", {migraphx::shape::float_type, {2, 4}});
        auto a0 = m2.add_parameter("a0", {migraphx::shape::float_type, {2, 4}});
        auto a1 = m2.add_parameter("a1", {migraphx::shape::float_type, {2, 4}});
        auto a2 = m2.add_parameter("a2", {migraphx::shape::float_type, {2, 4}});
        auto w0 =
            m2.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {4, 4}}, 0));
        auto w1 =
            m2.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {4, 4}}, 1));
        auto wi0 =
            m2.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {4, 4}}, 2));
        auto wi1 =
            m2.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {4, 4}}, 3));
        auto wi2 =
            m2.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {4, 4}}, 4));

        auto ux  = m2.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0}}}), x);
        auto ua0 = m2.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0}}}), a0);
        auto ua1 = m2.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0}}}), a1);
        auto ua2 = m2.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0}}}), a2);
        auto bact =
            m2.add_instruction(migraphx::make_op("concat", {{"axis", 0}}), ux, ua0, ua1, ua2);
        auto uw0  = m2.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0}}}), w0);
        auto uwi0 = m2.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0}}}), wi0);
        auto uwi1 = m2.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0}}}), wi1);
        auto uwi2 = m2.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0}}}), wi2);
        auto bwt =
            m2.add_instruction(migraphx::make_op("concat", {{"axis", 0}}), uw0, uwi0, uwi1, uwi2);
        auto bd = m2.add_instruction(migraphx::make_op("dot"), bact, bwt);

        auto sx = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {0}}, {"ends", {1}}}), bd);
        auto sqx = m2.add_instruction(migraphx::make_op("squeeze", {{"axes", {0}}}), sx);
        auto s0  = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {1}}, {"ends", {2}}}), bd);
        auto sq0 = m2.add_instruction(migraphx::make_op("squeeze", {{"axes", {0}}}), s0);
        auto s1  = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {2}}, {"ends", {3}}}), bd);
        auto sq1 = m2.add_instruction(migraphx::make_op("squeeze", {{"axes", {0}}}), s1);
        auto s2  = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {3}}, {"ends", {4}}}), bd);
        auto sq2 = m2.add_instruction(migraphx::make_op("squeeze", {{"axes", {0}}}), s2);

        auto add        = m2.add_instruction(migraphx::make_op("add"), sqx, b);
        auto dependent1 = m2.add_instruction(migraphx::make_op("dot"), add, w1);
        m2.add_return({dependent1, sq0, sq1, sq2});
    }

    EXPECT(m1.sort() == m2.sort());
}

// An unrolled recurrent cell mixes independent input dots with chained hidden-state dots.
// The input dots and the initial hidden dot form one independent subgroup and fuse; each
// chained hidden dot lands in its own subgroup below min_group_size.
TEST_CASE(dot_horiz_fusion_unrolled_recurrent_cell)
{
    migraphx::module m1;
    {
        auto wih =
            m1.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {4, 4}}, 0));
        auto whh =
            m1.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {4, 4}}, 1));
        auto h0 = m1.add_parameter("h0", {migraphx::shape::float_type, {1, 4}});
        auto x0 = m1.add_parameter("x0", {migraphx::shape::float_type, {1, 4}});
        auto x1 = m1.add_parameter("x1", {migraphx::shape::float_type, {1, 4}});
        auto x2 = m1.add_parameter("x2", {migraphx::shape::float_type, {1, 4}});
        auto x3 = m1.add_parameter("x3", {migraphx::shape::float_type, {1, 4}});

        auto dx0 = m1.add_instruction(migraphx::make_op("dot"), x0, wih);
        auto dh0 = m1.add_instruction(migraphx::make_op("dot"), h0, whh);
        auto s0  = m1.add_instruction(migraphx::make_op("add"), dx0, dh0);
        auto h1  = m1.add_instruction(migraphx::make_op("sigmoid"), s0);
        auto dx1 = m1.add_instruction(migraphx::make_op("dot"), x1, wih);
        auto dh1 = m1.add_instruction(migraphx::make_op("dot"), h1, whh);
        auto s1  = m1.add_instruction(migraphx::make_op("add"), dx1, dh1);
        auto h2  = m1.add_instruction(migraphx::make_op("sigmoid"), s1);
        auto dx2 = m1.add_instruction(migraphx::make_op("dot"), x2, wih);
        auto dh2 = m1.add_instruction(migraphx::make_op("dot"), h2, whh);
        auto s2  = m1.add_instruction(migraphx::make_op("add"), dx2, dh2);
        auto h3  = m1.add_instruction(migraphx::make_op("sigmoid"), s2);
        auto dx3 = m1.add_instruction(migraphx::make_op("dot"), x3, wih);
        auto dh3 = m1.add_instruction(migraphx::make_op("dot"), h3, whh);
        auto s3  = m1.add_instruction(migraphx::make_op("add"), dx3, dh3);
        auto h4  = m1.add_instruction(migraphx::make_op("sigmoid"), s3);
        m1.add_return({h4});
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto wih =
            m2.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {4, 4}}, 0));
        auto whh =
            m2.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {4, 4}}, 1));
        auto h0 = m2.add_parameter("h0", {migraphx::shape::float_type, {1, 4}});
        auto x0 = m2.add_parameter("x0", {migraphx::shape::float_type, {1, 4}});
        auto x1 = m2.add_parameter("x1", {migraphx::shape::float_type, {1, 4}});
        auto x2 = m2.add_parameter("x2", {migraphx::shape::float_type, {1, 4}});
        auto x3 = m2.add_parameter("x3", {migraphx::shape::float_type, {1, 4}});

        // Independent subgroup in position order: dx0, dh0, dx1, dx2, dx3
        auto ux0 = m2.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0}}}), x0);
        auto uh0 = m2.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0}}}), h0);
        auto ux1 = m2.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0}}}), x1);
        auto ux2 = m2.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0}}}), x2);
        auto ux3 = m2.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0}}}), x3);
        auto bact =
            m2.add_instruction(migraphx::make_op("concat", {{"axis", 0}}), ux0, uh0, ux1, ux2, ux3);
        auto uw0 = m2.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0}}}), wih);
        auto uw1 = m2.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0}}}), whh);
        auto uw2 = m2.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0}}}), wih);
        auto uw3 = m2.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0}}}), wih);
        auto uw4 = m2.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0}}}), wih);
        auto bwt =
            m2.add_instruction(migraphx::make_op("concat", {{"axis", 0}}), uw0, uw1, uw2, uw3, uw4);
        auto bd = m2.add_instruction(migraphx::make_op("dot"), bact, bwt);

        std::vector<migraphx::instruction_ref> sq(5);
        for(std::size_t i = 0; i < 5; i++)
        {
            auto s = m2.add_instruction(
                migraphx::make_op(
                    "slice", {{"axes", {0}}, {"starts", {int64_t(i)}}, {"ends", {int64_t(i + 1)}}}),
                bd);
            sq[i] = m2.add_instruction(migraphx::make_op("squeeze", {{"axes", {0}}}), s);
        }

        auto s0  = m2.add_instruction(migraphx::make_op("add"), sq[0], sq[1]);
        auto h1  = m2.add_instruction(migraphx::make_op("sigmoid"), s0);
        auto dh1 = m2.add_instruction(migraphx::make_op("dot"), h1, whh);
        auto s1  = m2.add_instruction(migraphx::make_op("add"), sq[2], dh1);
        auto h2  = m2.add_instruction(migraphx::make_op("sigmoid"), s1);
        auto dh2 = m2.add_instruction(migraphx::make_op("dot"), h2, whh);
        auto s2  = m2.add_instruction(migraphx::make_op("add"), sq[3], dh2);
        auto h3  = m2.add_instruction(migraphx::make_op("sigmoid"), s2);
        auto dh3 = m2.add_instruction(migraphx::make_op("dot"), h3, whh);
        auto s3  = m2.add_instruction(migraphx::make_op("add"), sq[4], dh3);
        auto h4  = m2.add_instruction(migraphx::make_op("sigmoid"), s3);
        m2.add_return({h4});
    }

    EXPECT(m1.sort() == m2.sort());
}

// Expected form of one fused level in dot_horiz_fusion_layered_independent_groups: batch the
// three dots of `act` against w0..w2, slice each result back out, and join them with adds.
static migraphx::instruction_ref add_batched_dot_level(migraphx::module& m,
                                                       migraphx::instruction_ref act,
                                                       migraphx::instruction_ref w0,
                                                       migraphx::instruction_ref w1,
                                                       migraphx::instruction_ref w2)
{
    auto ua0  = m.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0}}}), act);
    auto ua1  = m.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0}}}), act);
    auto ua2  = m.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0}}}), act);
    auto bact = m.add_instruction(migraphx::make_op("concat", {{"axis", 0}}), ua0, ua1, ua2);
    auto uw0  = m.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0}}}), w0);
    auto uw1  = m.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0}}}), w1);
    auto uw2  = m.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0}}}), w2);
    auto bwt  = m.add_instruction(migraphx::make_op("concat", {{"axis", 0}}), uw0, uw1, uw2);
    auto bd   = m.add_instruction(migraphx::make_op("dot"), bact, bwt);

    std::vector<migraphx::instruction_ref> sq(3);
    for(std::size_t i = 0; i < 3; i++)
    {
        auto s = m.add_instruction(
            migraphx::make_op(
                "slice", {{"axes", {0}}, {"starts", {int64_t(i)}}, {"ends", {int64_t(i + 1)}}}),
            bd);
        sq[i] = m.add_instruction(migraphx::make_op("squeeze", {{"axes", {0}}}), s);
    }
    auto s01 = m.add_instruction(migraphx::make_op("add"), sq[0], sq[1]);
    return m.add_instruction(migraphx::make_op("add"), s01, sq[2]);
}

// Layered graph where every level is joined before the next:
//   node1 -> {A1,A2,A3} -> node2 -> {B1,B2,B3} -> node3 -> {C1,C2,C3} -> node4
// All nine dots share one group key; each level is an independent subgroup and fuses.
TEST_CASE(dot_horiz_fusion_layered_independent_groups)
{
    migraphx::shape xs{migraphx::shape::float_type, {2, 4}};
    migraphx::shape ws{migraphx::shape::float_type, {4, 4}};

    migraphx::module m1;
    {
        auto x   = m1.add_parameter("x", xs);
        auto wa0 = m1.add_literal(migraphx::generate_literal(ws, 0));
        auto wa1 = m1.add_literal(migraphx::generate_literal(ws, 1));
        auto wa2 = m1.add_literal(migraphx::generate_literal(ws, 2));
        auto wb0 = m1.add_literal(migraphx::generate_literal(ws, 3));
        auto wb1 = m1.add_literal(migraphx::generate_literal(ws, 4));
        auto wb2 = m1.add_literal(migraphx::generate_literal(ws, 5));
        auto wc0 = m1.add_literal(migraphx::generate_literal(ws, 6));
        auto wc1 = m1.add_literal(migraphx::generate_literal(ws, 7));
        auto wc2 = m1.add_literal(migraphx::generate_literal(ws, 8));

        auto a0    = m1.add_instruction(migraphx::make_op("dot"), x, wa0);
        auto a1    = m1.add_instruction(migraphx::make_op("dot"), x, wa1);
        auto a2    = m1.add_instruction(migraphx::make_op("dot"), x, wa2);
        auto a01   = m1.add_instruction(migraphx::make_op("add"), a0, a1);
        auto node2 = m1.add_instruction(migraphx::make_op("add"), a01, a2);

        auto b0    = m1.add_instruction(migraphx::make_op("dot"), node2, wb0);
        auto b1    = m1.add_instruction(migraphx::make_op("dot"), node2, wb1);
        auto b2    = m1.add_instruction(migraphx::make_op("dot"), node2, wb2);
        auto b01   = m1.add_instruction(migraphx::make_op("add"), b0, b1);
        auto node3 = m1.add_instruction(migraphx::make_op("add"), b01, b2);

        auto c0    = m1.add_instruction(migraphx::make_op("dot"), node3, wc0);
        auto c1    = m1.add_instruction(migraphx::make_op("dot"), node3, wc1);
        auto c2    = m1.add_instruction(migraphx::make_op("dot"), node3, wc2);
        auto c01   = m1.add_instruction(migraphx::make_op("add"), c0, c1);
        auto node4 = m1.add_instruction(migraphx::make_op("add"), c01, c2);
        m1.add_return({node4});
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto x   = m2.add_parameter("x", xs);
        auto wa0 = m2.add_literal(migraphx::generate_literal(ws, 0));
        auto wa1 = m2.add_literal(migraphx::generate_literal(ws, 1));
        auto wa2 = m2.add_literal(migraphx::generate_literal(ws, 2));
        auto wb0 = m2.add_literal(migraphx::generate_literal(ws, 3));
        auto wb1 = m2.add_literal(migraphx::generate_literal(ws, 4));
        auto wb2 = m2.add_literal(migraphx::generate_literal(ws, 5));
        auto wc0 = m2.add_literal(migraphx::generate_literal(ws, 6));
        auto wc1 = m2.add_literal(migraphx::generate_literal(ws, 7));
        auto wc2 = m2.add_literal(migraphx::generate_literal(ws, 8));

        auto node2 = add_batched_dot_level(m2, x, wa0, wa1, wa2);
        auto node3 = add_batched_dot_level(m2, node2, wb0, wb1, wb2);
        auto node4 = add_batched_dot_level(m2, node3, wc0, wc1, wc2);
        m2.add_return({node4});
    }

    EXPECT(m1.sort() == m2.sort());
}

// Two levels of same-table gathers where the second level's indices derive from the first
// level's outputs: each level is an independent subgroup and fuses on its own.
TEST_CASE(same_table_gathers_layered_independent_groups)
{
    migraphx::shape ts{migraphx::shape::float_type, {8, 2}};
    migraphx::shape is{migraphx::shape::int32_type, {4}};

    migraphx::module m1;
    {
        auto tbl = m1.add_literal(migraphx::generate_literal(ts, 0));
        auto ia  = m1.add_parameter("ia", is);
        auto ib  = m1.add_parameter("ib", is);

        auto ga0 = m1.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), tbl, ia);
        auto ga1 = m1.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), tbl, ib);

        auto cvt0 = m1.add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::int32_type}}), ga0);
        auto cvt1 = m1.add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::int32_type}}), ga1);

        auto gb0 = m1.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), tbl, cvt0);
        auto gb1 = m1.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), tbl, cvt1);
        m1.add_return({gb0, gb1});
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto tbl = m2.add_literal(migraphx::generate_literal(ts, 0));
        auto ia  = m2.add_parameter("ia", is);
        auto ib  = m2.add_parameter("ib", is);

        auto cidx0 = m2.add_instruction(migraphx::make_op("concat", {{"axis", 0}}), ia, ib);
        auto bg0   = m2.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), tbl, cidx0);
        auto sa0   = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {0}}, {"ends", {4}}}), bg0);
        auto sa1 = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {4}}, {"ends", {8}}}), bg0);

        auto cvt0 = m2.add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::int32_type}}), sa0);
        auto cvt1 = m2.add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::int32_type}}), sa1);

        auto cidx1 = m2.add_instruction(migraphx::make_op("concat", {{"axis", 0}}), cvt0, cvt1);
        auto bg1   = m2.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), tbl, cidx1);
        auto sb0   = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {0}}, {"ends", {4}}}), bg1);
        auto sb1 = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {4}}, {"ends", {8}}}), bg1);
        m2.add_return({sb0, sb1});
    }

    EXPECT(m1.sort() == m2.sort());
}

// Dots whose weights are not compile-time constants are not candidates.
TEST_CASE(dot_horiz_fusion_non_constant_weight_unchanged)
{
    migraphx::module m1;
    {
        auto a0 = m1.add_parameter("a0", {migraphx::shape::float_type, {2, 3}});
        auto a1 = m1.add_parameter("a1", {migraphx::shape::float_type, {2, 3}});
        auto w0 = m1.add_parameter("w0", {migraphx::shape::float_type, {3, 4}});
        auto w1 = m1.add_parameter("w1", {migraphx::shape::float_type, {3, 4}});
        auto d0 = m1.add_instruction(migraphx::make_op("dot"), a0, w0);
        auto d1 = m1.add_instruction(migraphx::make_op("dot"), a1, w1);
        m1.add_return({d0, d1});
    }
    migraphx::module before = m1;
    run_pass(m1);

    EXPECT(m1.sort() == before.sort());
}

// Dots with different activation/weight shapes do not share a group key.
TEST_CASE(dot_horiz_fusion_mismatched_shapes_unchanged)
{
    migraphx::module m1;
    {
        auto a0 = m1.add_parameter("a0", {migraphx::shape::float_type, {2, 3}});
        auto a1 = m1.add_parameter("a1", {migraphx::shape::float_type, {2, 5}});
        auto w0 =
            m1.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {3, 4}}, 0));
        auto w1 =
            m1.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {5, 4}}, 1));
        auto d0 = m1.add_instruction(migraphx::make_op("dot"), a0, w0);
        auto d1 = m1.add_instruction(migraphx::make_op("dot"), a1, w1);
        m1.add_return({d0, d1});
    }
    migraphx::module before = m1;
    run_pass(m1);

    EXPECT(m1.sort() == before.sort());
}

// End-to-end: the pointwise hoist in find_splits (simplify_algebra) and
// dot_horizontal_fusion (fuse_horizontal) cooperate in one pipeline.  The module
// contains both opportunities and both reductions must occur.  This mirrors the
// GPU pipeline ordering where fuse_pointwise collapses each per-slice SiLU into
// a single pointwise op, simplify_algebra (optimize_module) then hoists it above
// the slices, and fuse_horizontal batches the parallel dots.
static void run_mlp_pipeline(migraphx::program& p)
{
    migraphx::run_passes(p,
                         {migraphx::fuse_pointwise{},
                          migraphx::dead_code_elimination{},
                          migraphx::simplify_algebra{},
                          migraphx::dead_code_elimination{},
                          migraphx::fuse_horizontal{},
                          migraphx::dead_code_elimination{}});
}

TEST_CASE(hoist_and_dot_fusion_end_to_end)
{
    migraphx::program p;
    {
        auto* m = p.get_main_module();
        // Hoist target: a SiLU (sigmoid * x) replicated across sibling row
        // slices of a common tensor that exactly tile it.
        auto t = m->add_parameter("t", {migraphx::shape::float_type, {4, 8}});
        std::vector<migraphx::instruction_ref> outs;
        for(int i = 0; i < 4; ++i)
        {
            auto s = m->add_instruction(
                migraphx::make_op("slice", {{"axes", {0}}, {"starts", {i}}, {"ends", {i + 1}}}), t);
            auto sig = m->add_instruction(migraphx::make_op("sigmoid"), s);
            outs.push_back(m->add_instruction(migraphx::make_op("mul"), s, sig));
        }

        // Dot-fusion target: parallel constant-weight dots with identical shapes.
        for(int i = 0; i < 3; ++i)
        {
            auto x =
                m->add_parameter("x" + std::to_string(i), {migraphx::shape::float_type, {2, 3}});
            auto w = m->add_literal(
                migraphx::generate_literal({migraphx::shape::float_type, {3, 5}}, i));
            outs.push_back(m->add_instruction(migraphx::make_op("dot"), x, w));
        }

        m->add_return(outs);
    }
    run_mlp_pipeline(p);

    std::size_t n_dot       = 0;
    std::size_t n_pointwise = 0;
    for(auto ins : iterator_for(*p.get_main_module()))
    {
        const auto& name = ins->name();
        if(name == "dot")
            ++n_dot;
        else if(name == "pointwise")
            ++n_pointwise;
    }

    // dot_horizontal_fusion collapses the 3 towers into a single batched GEMM.
    EXPECT(n_dot == 1);
    // fuse_pointwise turns each per-slice SiLU into one pointwise op, which the
    // find_splits hoist then collapses into a single pointwise on the bounding
    // slice.
    EXPECT(n_pointwise == 1);
}

// Parallel SwiGLU expert heads -- add(dot(mul(x, sigmoid(x)), W), bias) -- batch
// into a single GEMM via dot_horizontal_fusion even though each dot feeds an
// elementwise epilogue.  Nothing is stranded: find_splits (simplify_algebra)
// re-fuses the per-slice epilogue after the batched dot is sliced back out.
TEST_CASE(expert_head_dots_batch_with_constant_epilogue)
{
    migraphx::module m;
    {
        auto add_head = [&](const std::string& name, int seed) {
            auto x = m.add_parameter(name, {migraphx::shape::float_type, {2, 8}});
            auto w = m.add_literal(
                migraphx::generate_literal({migraphx::shape::float_type, {8, 8}}, seed));
            auto b = m.add_literal(
                migraphx::generate_literal({migraphx::shape::float_type, {2, 8}}, 10 + seed));
            auto sig = m.add_instruction(migraphx::make_op("sigmoid"), x);
            auto mul = m.add_instruction(migraphx::make_op("mul"), x, sig);
            auto d   = m.add_instruction(migraphx::make_op("dot"), mul, w);
            return m.add_instruction(migraphx::make_op("add"), d, b);
        };
        m.add_return({add_head("x0", 0), add_head("x1", 1), add_head("x2", 2), add_head("x3", 3)});
    }
    run_pass(m);

    std::size_t n_dot = 0;
    for(auto ins : iterator_for(m))
    {
        if(ins->name() == "dot")
            ++n_dot;
    }
    // The four per-head epilogue dots collapse into a single batched GEMM.
    EXPECT(n_dot == 1);
}

// A runtime (non-constant) epilogue operand does not affect GEMM batching: only
// the weight needs to be constant, so the dots still collapse into one batched
// GEMM regardless of what feeds the epilogue.
TEST_CASE(expert_head_dots_batch_with_runtime_epilogue)
{
    migraphx::module m;
    {
        auto add_head = [&](const std::string& xname, const std::string& vname, int seed) {
            auto x = m.add_parameter(xname, {migraphx::shape::float_type, {2, 8}});
            auto v = m.add_parameter(vname, {migraphx::shape::float_type, {2, 8}});
            auto w = m.add_literal(
                migraphx::generate_literal({migraphx::shape::float_type, {8, 8}}, seed));
            auto sig = m.add_instruction(migraphx::make_op("sigmoid"), x);
            auto mul = m.add_instruction(migraphx::make_op("mul"), x, sig);
            auto d   = m.add_instruction(migraphx::make_op("dot"), mul, w);
            return m.add_instruction(migraphx::make_op("add"), d, v);
        };
        m.add_return({add_head("x0", "v0", 0),
                      add_head("x1", "v1", 1),
                      add_head("x2", "v2", 2),
                      add_head("x3", "v3", 3)});
    }
    run_pass(m);

    std::size_t n_dot = 0;
    for(auto ins : iterator_for(m))
    {
        if(ins->name() == "dot")
            ++n_dot;
    }
    // Batching is independent of the epilogue operand; the dots still collapse.
    EXPECT(n_dot == 1);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
