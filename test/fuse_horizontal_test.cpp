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

TEST_CASE(same_table_gathers_split_by_trailing_dims)
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

        // 1D group fuses first (anchor sees idx_1d_a)
        auto concat_idx_1d =
            m2.add_instruction(migraphx::make_op("concat", {{"axis", 0}}),
                               std::vector<migraphx::instruction_ref>{idx_1d_a, idx_1d_b});

        auto bg_1d =
            m2.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), emb, concat_idx_1d);

        auto s1 = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {0}}, {"ends", {4}}}), bg_1d);
        auto s2 = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {4}}, {"ends", {9}}}), bg_1d);

        // 2D group fuses next
        auto concat_idx_2d =
            m2.add_instruction(migraphx::make_op("concat", {{"axis", 0}}),
                               std::vector<migraphx::instruction_ref>{idx_2d_a, idx_2d_b});

        auto bg_2d =
            m2.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), emb, concat_idx_2d);

        auto s3 = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {0}}, {"ends", {4}}}), bg_2d);
        auto s4 = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {4}}, {"ends", {9}}}), bg_2d);

        m2.add_return({s1, s2, s3, s4});
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

        // Table A is fused first (anchor sees ga1)
        auto concat_idx_a =
            m2.add_instruction(migraphx::make_op("concat", {{"axis", 0}}),
                               std::vector<migraphx::instruction_ref>{idx_a1, idx_a2});

        auto bg_a =
            m2.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), emb_a, concat_idx_a);

        auto sa1 = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {0}}, {"ends", {4}}}), bg_a);
        auto sa2 = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {4}}, {"ends", {9}}}), bg_a);

        // Table B is fused next
        auto concat_idx_b =
            m2.add_instruction(migraphx::make_op("concat", {{"axis", 0}}),
                               std::vector<migraphx::instruction_ref>{idx_b1, idx_b2});

        auto bg_b =
            m2.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), emb_b, concat_idx_b);

        auto sb1 = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {0}}, {"ends", {4}}}), bg_b);
        auto sb2 = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {4}}, {"ends", {9}}}), bg_b);

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

// Two independent dots with identical activation/weight shapes and constant
// weights should batch into a single GEMM, then slice+squeeze back.
TEST_CASE(dot_horiz_fusion_basic)
{
    migraphx::module m1;
    {
        auto a0 = m1.add_parameter("a0", {migraphx::shape::float_type, {2, 3}});
        auto a1 = m1.add_parameter("a1", {migraphx::shape::float_type, {2, 3}});
        auto w0 =
            m1.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {3, 4}}, 0));
        auto w1 =
            m1.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {3, 4}}, 1));
        auto d0 = m1.add_instruction(migraphx::make_op("dot"), a0, w0);
        auto d1 = m1.add_instruction(migraphx::make_op("dot"), a1, w1);
        m1.add_return({d0, d1});
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto a0 = m2.add_parameter("a0", {migraphx::shape::float_type, {2, 3}});
        auto a1 = m2.add_parameter("a1", {migraphx::shape::float_type, {2, 3}});
        auto w0 =
            m2.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {3, 4}}, 0));
        auto w1 =
            m2.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {3, 4}}, 1));

        auto ua0  = m2.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0}}}), a0);
        auto ua1  = m2.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0}}}), a1);
        auto bact = m2.add_instruction(migraphx::make_op("concat", {{"axis", 0}}), ua0, ua1);
        auto uw0  = m2.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0}}}), w0);
        auto uw1  = m2.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0}}}), w1);
        auto bwt  = m2.add_instruction(migraphx::make_op("concat", {{"axis", 0}}), uw0, uw1);
        auto bd   = m2.add_instruction(migraphx::make_op("dot"), bact, bwt);

        auto s0 = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {0}}, {"ends", {1}}}), bd);
        auto sq0 = m2.add_instruction(migraphx::make_op("squeeze", {{"axes", {0}}}), s0);
        auto s1  = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {1}}, {"ends", {2}}}), bd);
        auto sq1 = m2.add_instruction(migraphx::make_op("squeeze", {{"axes", {0}}}), s1);
        m2.add_return({sq0, sq1});
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

int main(int argc, const char* argv[]) { test::run(argc, argv); }
