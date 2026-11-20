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
#include <migraphx/gpu/fuse_chunk_split.hpp>
#include <migraphx/gpu/code_object_op.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/register_target.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/program.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/serialize.hpp>
#include <migraphx/algorithm.hpp>
#include <migraphx/ranges.hpp>
#include <test.hpp>
#include "kernel_names.hpp"
#include <pointwise.hpp>

static void run_pass(migraphx::program& p)
{
    migraphx::run_passes(p, {migraphx::gpu::fuse_chunk_split{}, migraphx::dead_code_elimination{}});
}

static migraphx::shape fshape(std::vector<std::size_t> lens)
{
    return {migraphx::shape::float_type, std::move(lens)};
}

// Chunk i of a batched per-chunk operand, the view the parser emits
static migraphx::instruction_ref
chunk(migraphx::module& m, migraphx::instruction_ref x, std::size_t i)
{
    auto s = m.add_instruction(
        migraphx::make_op("slice", {{"axes", {2}}, {"starts", {i}}, {"ends", {i + 1}}}), x);
    return m.add_instruction(migraphx::make_op("squeeze", {{"axes", {2}}}), s);
}

// S * decay + state_update
static auto gated_step()
{
    return [](auto* pm, const auto& xs) {
        auto sg = pm->add_instruction(migraphx::make_op("mul"), xs[0], xs[1]);
        return pm->add_instruction(migraphx::make_op("add"), sg, xs[2]);
    };
}

static migraphx::instruction_ref
get_tuple_elem(migraphx::module& m, migraphx::instruction_ref x, std::size_t i)
{
    return m.add_instruction(migraphx::make_op("get_tuple_elem", {{"index", i}}), x);
}

// Two chunks of two rows over two heads with a past state, the gated delta
// rule: u = update_input - read_matrix S, o = output_base + output_matrix S,
// S = S * decay + update_matrix u
TEST_CASE(gated_delta_with_state)
{
    auto ss  = fshape({1, 2, 4, 8});
    auto ks  = fshape({1, 2, 2, 2, 4});
    auto kts = fshape({1, 2, 2, 4, 2});
    auto vs  = fshape({1, 2, 2, 2, 8});
    auto gs  = fshape({1, 2, 2, 1, 1});
    migraphx::program p1;
    {
        auto* mm = p1.get_main_module();
        auto s0  = mm->add_parameter("s0", ss);
        auto read_matrix   = mm->add_parameter("read_matrix", ks);
        auto output_matrix = mm->add_parameter("output_matrix", ks);
        auto update_matrix = mm->add_parameter("update_matrix", kts);
        auto update_input  = mm->add_parameter("update_input", vs);
        auto output_base   = mm->add_parameter("output_base", vs);
        auto decay         = mm->add_parameter("decay", gs);
        auto s   = s0;
        std::vector<migraphx::instruction_ref> outs;
        for(std::size_t i = 0; i < 2; i++)
        {
            auto n  = std::to_string(3 * i);
            auto read_product =
                mm->add_instruction(migraphx::make_op("dot"), chunk(*mm, read_matrix, i), s);
            auto u = add_pointwise(p1,
                                   "main:pointwise" + n,
                                   {chunk(*mm, update_input, i), read_product},
                                   single_pointwise("sub"));
            auto output_product =
                mm->add_instruction(migraphx::make_op("dot"), chunk(*mm, output_matrix, i), s);
            outs.push_back(add_pointwise(p1,
                                         "main:pointwise" + std::to_string(3 * i + 1),
                                         {chunk(*mm, output_base, i), output_product},
                                         single_pointwise("add")));
            auto state_update =
                mm->add_instruction(migraphx::make_op("dot"), chunk(*mm, update_matrix, i), u);
            auto decay_i =
                mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", ss.lens()}}),
                                    chunk(*mm, decay, i));
            s = add_pointwise(p1,
                              "main:pointwise" + std::to_string(3 * i + 2),
                              {s, decay_i, state_update},
                              gated_step());
        }
        auto o = mm->add_instruction(migraphx::make_op("concat", {{"axis", 2}}), outs);
        mm->add_return({o, s});
    }
    run_pass(p1);

    migraphx::program p2;
    {
        auto* mm = p2.get_main_module();
        auto s0  = mm->add_parameter("s0", ss);
        auto read_matrix   = mm->add_parameter("read_matrix", ks);
        auto output_matrix = mm->add_parameter("output_matrix", ks);
        auto update_matrix = mm->add_parameter("update_matrix", kts);
        auto update_input  = mm->add_parameter("update_input", vs);
        auto output_base   = mm->add_parameter("output_base", vs);
        auto decay         = mm->add_parameter("decay", gs);
        auto* step =
            create_pointwise_module(p2, "main:chunk_scan0:step", {s0, s0, s0}, gated_step());
        auto* read =
            create_pointwise_module(p2, "main:chunk_scan0:read", {s0, s0}, single_pointwise("sub"));
        auto* out =
            create_pointwise_module(p2, "main:chunk_scan0:out", {s0, s0}, single_pointwise("add"));
        auto op = migraphx::make_op("gpu::chunk_scan",
                                    {{"output_shape", migraphx::to_value(fshape({1, 2, 4, 8}))},
                                     {"state_shape", migraphx::to_value(ss)},
                                     {"families",
                                      migraphx::to_value(std::vector<std::string>{"read_matrix",
                                                                                  "output_matrix",
                                                                                  "update_matrix",
                                                                                  "update_input",
                                                                                  "decay",
                                                                                  "output_base"})},
                                     {"chunks", 2},
                                     {"has_state", true}});
        auto decays = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {1, 2, 2, 4, 8}}}), decay);
        auto scan = mm->add_instruction(
            op,
            {s0, read_matrix, output_matrix, update_matrix, update_input, decays, output_base},
            {step, read, out});
        mm->add_return({get_tuple_elem(*mm, scan, 0), get_tuple_elem(*mm, scan, 1)});
    }
    EXPECT(p1.sort() == p2.sort());
}

// Three chunks without a past state under the delta rule: the first chunk
// emits its output_base and S_1 = update_matrix_0 update_input_0 directly, so
// its read_matrix and output_matrix are absent
TEST_CASE(delta_zero_start)
{
    auto ss  = fshape({1, 2, 4, 8});
    auto ks  = fshape({1, 2, 3, 2, 4});
    auto kts = fshape({1, 2, 3, 4, 2});
    auto vs  = fshape({1, 2, 3, 2, 8});
    migraphx::program p1;
    {
        auto* mm = p1.get_main_module();
        auto read_matrix   = mm->add_parameter("read_matrix", ks);
        auto output_matrix = mm->add_parameter("output_matrix", ks);
        auto update_matrix = mm->add_parameter("update_matrix", kts);
        auto update_input  = mm->add_parameter("update_input", vs);
        auto output_base   = mm->add_parameter("output_base", vs);
        std::vector<migraphx::instruction_ref> outs{chunk(*mm, output_base, 0)};
        auto s = mm->add_instruction(
            migraphx::make_op("dot"), chunk(*mm, update_matrix, 0), chunk(*mm, update_input, 0));
        for(std::size_t i = 1; i < 3; i++)
        {
            auto read_product =
                mm->add_instruction(migraphx::make_op("dot"), chunk(*mm, read_matrix, i), s);
            auto u = add_pointwise(p1,
                                   "main:pointwise" + std::to_string(3 * i),
                                   {chunk(*mm, update_input, i), read_product},
                                   single_pointwise("sub"));
            auto output_product =
                mm->add_instruction(migraphx::make_op("dot"), chunk(*mm, output_matrix, i), s);
            outs.push_back(add_pointwise(p1,
                                         "main:pointwise" + std::to_string(3 * i + 1),
                                         {chunk(*mm, output_base, i), output_product},
                                         single_pointwise("add")));
            auto state_update =
                mm->add_instruction(migraphx::make_op("dot"), chunk(*mm, update_matrix, i), u);
            s = add_pointwise(p1,
                              "main:pointwise" + std::to_string(3 * i + 2),
                              {s, state_update},
                              single_pointwise("add"));
        }
        auto o = mm->add_instruction(migraphx::make_op("concat", {{"axis", 2}}), outs);
        mm->add_return({o, s});
    }
    run_pass(p1);

    migraphx::program p2;
    {
        auto* mm = p2.get_main_module();
        auto read_matrix   = mm->add_parameter("read_matrix", ks);
        auto output_matrix = mm->add_parameter("output_matrix", ks);
        auto update_matrix = mm->add_parameter("update_matrix", kts);
        auto update_input  = mm->add_parameter("update_input", vs);
        auto output_base   = mm->add_parameter("output_base", vs);
        auto* step         = create_pointwise_module(
            p2, "main:chunk_scan0:step", {update_input, update_input}, single_pointwise("add"));
        auto* read = create_pointwise_module(
            p2, "main:chunk_scan0:read", {update_input, update_input}, single_pointwise("sub"));
        auto* out = create_pointwise_module(
            p2, "main:chunk_scan0:out", {update_input, update_input}, single_pointwise("add"));
        auto op = migraphx::make_op(
            "gpu::chunk_scan",
            {{"output_shape", migraphx::to_value(fshape({1, 2, 6, 8}))},
             {"state_shape", migraphx::to_value(ss)},
             {"families",
              migraphx::to_value(std::vector<std::string>{
                  "read_matrix", "output_matrix", "update_matrix", "update_input", "output_base"})},
             {"chunks", 3},
             {"has_state", false}});
        // chunk 0 reads no state, so the read and output matrices start at chunk 1
        auto tail = migraphx::make_op("slice", {{"axes", {2}}, {"starts", {1}}, {"ends", {3}}});
        auto read_matrices   = mm->add_instruction(tail, read_matrix);
        auto output_matrices = mm->add_instruction(tail, output_matrix);
        auto scan            = mm->add_instruction(
            op,
            {read_matrices, output_matrices, update_matrix, update_input, output_base},
            {step, read, out});
        mm->add_return({get_tuple_elem(*mm, scan, 0), get_tuple_elem(*mm, scan, 1)});
    }
    EXPECT(p1.sort() == p2.sort());
}

// The update is a precomputed per-chunk view and the outputs are the bare
// reads output_matrix S: only the step module is needed
TEST_CASE(precomputed_update)
{
    auto ss = fshape({1, 2, 4, 8});
    auto ks = fshape({1, 2, 2, 2, 4});
    auto us = fshape({1, 2, 2, 4, 8});
    migraphx::program p1;
    {
        auto* mm = p1.get_main_module();
        auto s0  = mm->add_parameter("s0", ss);
        auto output_matrix = mm->add_parameter("output_matrix", ks);
        auto state_update  = mm->add_parameter("state_update", us);
        auto s   = s0;
        std::vector<migraphx::instruction_ref> outs;
        for(std::size_t i = 0; i < 2; i++)
        {
            outs.push_back(
                mm->add_instruction(migraphx::make_op("dot"), chunk(*mm, output_matrix, i), s));
            s = add_pointwise(p1,
                              "main:pointwise" + std::to_string(i),
                              {s, chunk(*mm, state_update, i)},
                              single_pointwise("add"));
        }
        auto o = mm->add_instruction(migraphx::make_op("concat", {{"axis", 2}}), outs);
        mm->add_return({o, s});
    }
    run_pass(p1);

    migraphx::program p2;
    {
        auto* mm = p2.get_main_module();
        auto s0  = mm->add_parameter("s0", ss);
        auto output_matrix = mm->add_parameter("output_matrix", ks);
        auto state_update  = mm->add_parameter("state_update", us);
        auto* step =
            create_pointwise_module(p2, "main:chunk_scan0:step", {s0, s0}, single_pointwise("add"));
        auto op = migraphx::make_op(
            "gpu::chunk_scan",
            {{"output_shape", migraphx::to_value(fshape({1, 2, 4, 8}))},
             {"state_shape", migraphx::to_value(ss)},
             {"families",
              migraphx::to_value(std::vector<std::string>{"output_matrix", "state_update"})},
             {"chunks", 2},
             {"has_state", true}});
        auto scan = mm->add_instruction(op, {s0, output_matrix, state_update}, {step});
        mm->add_return({get_tuple_elem(*mm, scan, 0), get_tuple_elem(*mm, scan, 1)});
    }
    EXPECT(p1.sort() == p2.sort());
}

// The initial state may itself be a view, as a reshaped past state is
TEST_CASE(state_from_view)
{
    auto ss = fshape({1, 2, 4, 8});
    auto ks = fshape({1, 2, 2, 2, 4});
    auto us = fshape({1, 2, 2, 4, 8});
    migraphx::program p1;
    {
        auto* mm  = p1.get_main_module();
        auto past = mm->add_parameter("past", fshape({1, 2, 32}));
        auto output_matrix = mm->add_parameter("output_matrix", ks);
        auto state_update  = mm->add_parameter("state_update", us);
        auto s    = mm->add_instruction(migraphx::make_op("reshape", {{"dims", ss.lens()}}), past);
        std::vector<migraphx::instruction_ref> outs;
        for(std::size_t i = 0; i < 2; i++)
        {
            outs.push_back(
                mm->add_instruction(migraphx::make_op("dot"), chunk(*mm, output_matrix, i), s));
            s = add_pointwise(p1,
                              "main:pointwise" + std::to_string(i),
                              {s, chunk(*mm, state_update, i)},
                              single_pointwise("add"));
        }
        auto o = mm->add_instruction(migraphx::make_op("concat", {{"axis", 2}}), outs);
        mm->add_return({o, s});
    }
    run_pass(p1);

    migraphx::program p2;
    {
        auto* mm  = p2.get_main_module();
        auto past = mm->add_parameter("past", fshape({1, 2, 32}));
        auto output_matrix = mm->add_parameter("output_matrix", ks);
        auto state_update  = mm->add_parameter("state_update", us);
        auto s0   = mm->add_instruction(migraphx::make_op("reshape", {{"dims", ss.lens()}}), past);
        auto* step =
            create_pointwise_module(p2, "main:chunk_scan0:step", {s0, s0}, single_pointwise("add"));
        auto op = migraphx::make_op(
            "gpu::chunk_scan",
            {{"output_shape", migraphx::to_value(fshape({1, 2, 4, 8}))},
             {"state_shape", migraphx::to_value(ss)},
             {"families",
              migraphx::to_value(std::vector<std::string>{"output_matrix", "state_update"})},
             {"chunks", 2},
             {"has_state", true}});
        auto scan = mm->add_instruction(op, {s0, output_matrix, state_update}, {step});
        mm->add_return({get_tuple_elem(*mm, scan, 0), get_tuple_elem(*mm, scan, 1)});
    }
    EXPECT(p1.sort() == p2.sort());
}

// The dots may read another view of the past state than the transition does,
// as long as both hold the same matrices
TEST_CASE(state_from_sibling_views)
{
    auto ss = fshape({1, 2, 4, 8});
    auto ks = fshape({1, 2, 2, 2, 4});
    auto us = fshape({1, 2, 2, 4, 8});
    migraphx::program p1;
    {
        auto* mm  = p1.get_main_module();
        auto past = mm->add_parameter("past", fshape({1, 2, 32}));
        auto output_matrix = mm->add_parameter("output_matrix", ks);
        auto state_update  = mm->add_parameter("state_update", us);
        auto s    = mm->add_instruction(migraphx::make_op("reshape", {{"dims", ss.lens()}}), past);
        auto sv =
            mm->add_instruction(migraphx::make_op("reshape", {{"dims", {1, 2, 1, 4, 8}}}), past);
        std::vector<migraphx::instruction_ref> outs;
        for(std::size_t i = 0; i < 2; i++)
        {
            auto output_matrix_i = mm->add_instruction(
                migraphx::make_op("slice", {{"axes", {2}}, {"starts", {i}}, {"ends", {i + 1}}}),
                output_matrix);
            outs.push_back(mm->add_instruction(migraphx::make_op("dot"), output_matrix_i, sv));
            s  = add_pointwise(p1,
                               "main:pointwise" + std::to_string(i),
                               {s, chunk(*mm, state_update, i)},
                               single_pointwise("add"));
            sv = mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {2}}}), s);
        }
        auto o = mm->add_instruction(migraphx::make_op("concat", {{"axis", 3}}), outs);
        mm->add_return({o, s});
    }
    run_pass(p1);

    migraphx::program p2;
    {
        auto* mm  = p2.get_main_module();
        auto past = mm->add_parameter("past", fshape({1, 2, 32}));
        auto output_matrix = mm->add_parameter("output_matrix", ks);
        auto state_update  = mm->add_parameter("state_update", us);
        auto s0   = mm->add_instruction(migraphx::make_op("reshape", {{"dims", ss.lens()}}), past);
        auto* step =
            create_pointwise_module(p2, "main:chunk_scan0:step", {s0, s0}, single_pointwise("add"));
        auto op = migraphx::make_op(
            "gpu::chunk_scan",
            {{"output_shape", migraphx::to_value(fshape({1, 2, 4, 8}))},
             {"state_shape", migraphx::to_value(ss)},
             {"families",
              migraphx::to_value(std::vector<std::string>{"output_matrix", "state_update"})},
             {"chunks", 2},
             {"has_state", true}});
        auto scan = mm->add_instruction(op, {s0, output_matrix, state_update}, {step});
        auto o    = mm->add_instruction(migraphx::make_op("reshape", {{"dims", {1, 2, 1, 4, 8}}}),
                                        get_tuple_elem(*mm, scan, 0));
        mm->add_return({o, get_tuple_elem(*mm, scan, 1)});
    }
    EXPECT(p1.sort() == p2.sort());
}

// A second recurrence fed by the first's final state is its own scan
TEST_CASE(chained_recurrences)
{
    auto ss = fshape({1, 2, 4, 8});
    auto ks = fshape({1, 2, 2, 2, 4});
    auto us = fshape({1, 2, 2, 2, 4, 8});
    migraphx::program p1;
    {
        auto* mm = p1.get_main_module();
        auto s0  = mm->add_parameter("s0", ss);
        auto output_matrix  = mm->add_parameter("output_matrix", ks);
        auto output_matrix2 = mm->add_parameter("output_matrix2", ks);
        auto state_update   = mm->add_parameter("state_update", us);
        auto s   = s0;
        std::vector<migraphx::instruction_ref> outs1;
        std::vector<migraphx::instruction_ref> outs2;
        for(std::size_t i = 0; i < 2; i++)
        {
            outs1.push_back(
                mm->add_instruction(migraphx::make_op("dot"), chunk(*mm, output_matrix, i), s));
            auto state_update1 = mm->add_instruction(
                migraphx::make_op("slice",
                                  {{"axes", {2, 3}}, {"starts", {0, i}}, {"ends", {1, i + 1}}}),
                state_update);
            s = add_pointwise(p1,
                              "main:pointwise" + std::to_string(i),
                              {s,
                               mm->add_instruction(migraphx::make_op("squeeze", {{"axes", {2, 3}}}),
                                                   state_update1)},
                              single_pointwise("add"));
        }
        for(std::size_t i = 0; i < 2; i++)
        {
            outs2.push_back(
                mm->add_instruction(migraphx::make_op("dot"), chunk(*mm, output_matrix2, i), s));
            auto state_update2 = mm->add_instruction(
                migraphx::make_op("slice",
                                  {{"axes", {2, 3}}, {"starts", {1, i}}, {"ends", {2, i + 1}}}),
                state_update);
            s = add_pointwise(p1,
                              "main:pointwise" + std::to_string(2 + i),
                              {s,
                               mm->add_instruction(migraphx::make_op("squeeze", {{"axes", {2, 3}}}),
                                                   state_update2)},
                              single_pointwise("add"));
        }
        auto o1 = mm->add_instruction(migraphx::make_op("concat", {{"axis", 2}}), outs1);
        auto o2 = mm->add_instruction(migraphx::make_op("concat", {{"axis", 2}}), outs2);
        mm->add_return({o1, o2, s});
    }
    run_pass(p1);

    migraphx::program p2;
    {
        auto* mm = p2.get_main_module();
        auto s0  = mm->add_parameter("s0", ss);
        auto output_matrix  = mm->add_parameter("output_matrix", ks);
        auto output_matrix2 = mm->add_parameter("output_matrix2", ks);
        auto state_update   = mm->add_parameter("state_update", us);
        auto* step0 =
            create_pointwise_module(p2, "main:chunk_scan0:step", {s0, s0}, single_pointwise("add"));
        auto* step1 =
            create_pointwise_module(p2, "main:chunk_scan1:step", {s0, s0}, single_pointwise("add"));
        auto op = migraphx::make_op(
            "gpu::chunk_scan",
            {{"output_shape", migraphx::to_value(fshape({1, 2, 4, 8}))},
             {"state_shape", migraphx::to_value(ss)},
             {"families",
              migraphx::to_value(std::vector<std::string>{"output_matrix", "state_update"})},
             {"chunks", 2},
             {"has_state", true}});
        // each recurrence takes its half of state_update, with the chunks on axis 2
        auto half0 = mm->add_instruction(
            migraphx::make_op("slice", {{"axes", {2, 3}}, {"starts", {0, 0}}, {"ends", {1, 2}}}),
            state_update);
        auto state_update0 =
            mm->add_instruction(migraphx::make_op("squeeze", {{"axes", {2}}}), half0);
        auto scan0 = mm->add_instruction(op, {s0, output_matrix, state_update0}, {step0});
        auto s1    = get_tuple_elem(*mm, scan0, 1);
        auto half1 = mm->add_instruction(
            migraphx::make_op("slice", {{"axes", {2, 3}}, {"starts", {1, 0}}, {"ends", {2, 2}}}),
            state_update);
        auto state_update1 =
            mm->add_instruction(migraphx::make_op("squeeze", {{"axes", {2}}}), half1);
        auto scan1 = mm->add_instruction(op, {s1, output_matrix2, state_update1}, {step1});
        mm->add_return({get_tuple_elem(*mm, scan0, 0),
                        get_tuple_elem(*mm, scan1, 0),
                        get_tuple_elem(*mm, scan1, 1)});
    }
    EXPECT(p1.sort() == p2.sort());
}

// Chunks whose transitions differ are not one recurrence
TEST_CASE(different_steps_not_fused)
{
    auto ss = fshape({1, 2, 4, 8});
    auto ks = fshape({1, 2, 2, 2, 4});
    auto us = fshape({1, 2, 2, 4, 8});
    migraphx::program p1;
    {
        auto* mm = p1.get_main_module();
        auto s0  = mm->add_parameter("s0", ss);
        auto output_matrix = mm->add_parameter("output_matrix", ks);
        auto state_update  = mm->add_parameter("state_update", us);
        auto output0 =
            mm->add_instruction(migraphx::make_op("dot"), chunk(*mm, output_matrix, 0), s0);
        auto s1 = add_pointwise(
            p1, "main:pointwise0", {s0, chunk(*mm, state_update, 0)}, single_pointwise("add"));
        auto output1 =
            mm->add_instruction(migraphx::make_op("dot"), chunk(*mm, output_matrix, 1), s1);
        auto s2 = add_pointwise(
            p1, "main:pointwise1", {s1, chunk(*mm, state_update, 1)}, single_pointwise("sub"));
        auto o = mm->add_instruction(migraphx::make_op("concat", {{"axis", 2}}), output0, output1);
        mm->add_return({o, s2});
    }
    migraphx::program p2 = p1;
    run_pass(p1);
    EXPECT(p1.sort() == p2.sort());
}

// The same body over swapped inputs is another transition: S - state_update
// for one chunk and state_update - S for the next
TEST_CASE(swapped_step_inputs_not_fused)
{
    auto ss = fshape({1, 2, 4, 8});
    auto ks = fshape({1, 2, 2, 2, 4});
    auto us = fshape({1, 2, 2, 4, 8});
    migraphx::program p1;
    {
        auto* mm           = p1.get_main_module();
        auto s0            = mm->add_parameter("s0", ss);
        auto output_matrix = mm->add_parameter("output_matrix", ks);
        auto state_update  = mm->add_parameter("state_update", us);
        auto output0 =
            mm->add_instruction(migraphx::make_op("dot"), chunk(*mm, output_matrix, 0), s0);
        auto s1 = add_pointwise(
            p1, "main:pointwise0", {s0, chunk(*mm, state_update, 0)}, single_pointwise("sub"));
        auto output1 =
            mm->add_instruction(migraphx::make_op("dot"), chunk(*mm, output_matrix, 1), s1);
        auto s2 = add_pointwise(
            p1, "main:pointwise1", {chunk(*mm, state_update, 1), s1}, single_pointwise("sub"));
        auto o = mm->add_instruction(migraphx::make_op("concat", {{"axis", 2}}), output0, output1);
        mm->add_return({o, s2});
    }
    migraphx::program p2 = p1;
    run_pass(p1);
    EXPECT(p1.sort() == p2.sort());
}

// The same transition written over its inputs in another order is the same
// step: S - state_update as x0 - x1 and as x1 - x0 over swapped inputs
TEST_CASE(reordered_step_inputs)
{
    auto ss = fshape({1, 2, 4, 8});
    auto ks = fshape({1, 2, 2, 2, 4});
    auto us = fshape({1, 2, 2, 4, 8});
    migraphx::program p1;
    {
        auto* mm           = p1.get_main_module();
        auto s0            = mm->add_parameter("s0", ss);
        auto output_matrix = mm->add_parameter("output_matrix", ks);
        auto state_update  = mm->add_parameter("state_update", us);
        auto output0 =
            mm->add_instruction(migraphx::make_op("dot"), chunk(*mm, output_matrix, 0), s0);
        auto s1 = add_pointwise(
            p1, "main:pointwise0", {s0, chunk(*mm, state_update, 0)}, single_pointwise("sub"));
        auto output1 =
            mm->add_instruction(migraphx::make_op("dot"), chunk(*mm, output_matrix, 1), s1);
        auto s2 = add_pointwise(
            p1, "main:pointwise1", {chunk(*mm, state_update, 1), s1}, [](auto* pm, const auto& xs) {
                return pm->add_instruction(migraphx::make_op("sub"), xs[1], xs[0]);
            });
        auto o = mm->add_instruction(migraphx::make_op("concat", {{"axis", 2}}), output0, output1);
        mm->add_return({o, s2});
    }
    run_pass(p1);

    migraphx::program p2;
    {
        auto* mm           = p2.get_main_module();
        auto s0            = mm->add_parameter("s0", ss);
        auto output_matrix = mm->add_parameter("output_matrix", ks);
        auto state_update  = mm->add_parameter("state_update", us);
        auto* step =
            create_pointwise_module(p2, "main:chunk_scan0:step", {s0, s0}, single_pointwise("sub"));
        auto op = migraphx::make_op(
            "gpu::chunk_scan",
            {{"output_shape", migraphx::to_value(fshape({1, 2, 4, 8}))},
             {"state_shape", migraphx::to_value(ss)},
             {"families",
              migraphx::to_value(std::vector<std::string>{"output_matrix", "state_update"})},
             {"chunks", 2},
             {"has_state", true}});
        auto scan = mm->add_instruction(op, {s0, output_matrix, state_update}, {step});
        mm->add_return({get_tuple_elem(*mm, scan, 0), get_tuple_elem(*mm, scan, 1)});
    }
    EXPECT(p1.sort() == p2.sort());
}

// Nor may the read swap its inputs between chunks: update_input - product
// and product - update_input
TEST_CASE(swapped_read_inputs_not_fused)
{
    auto ss = fshape({1, 2, 4, 4});
    auto ks = fshape({1, 2, 2, 4, 4});
    migraphx::program p1;
    {
        auto* mm           = p1.get_main_module();
        auto s0            = mm->add_parameter("s0", ss);
        auto read_matrix   = mm->add_parameter("read_matrix", ks);
        auto output_matrix = mm->add_parameter("output_matrix", ks);
        auto update_matrix = mm->add_parameter("update_matrix", ks);
        auto update_input  = mm->add_parameter("update_input", ks);
        auto s             = s0;
        std::vector<migraphx::instruction_ref> outs;
        for(std::size_t i = 0; i < 2; i++)
        {
            auto read_product =
                mm->add_instruction(migraphx::make_op("dot"), chunk(*mm, read_matrix, i), s);
            auto input = chunk(*mm, update_input, i);
            auto u =
                add_pointwise(p1,
                              "main:pointwise" + std::to_string(2 * i),
                              i == 0 ? std::vector<migraphx::instruction_ref>{input, read_product}
                                     : std::vector<migraphx::instruction_ref>{read_product, input},
                              single_pointwise("sub"));
            outs.push_back(
                mm->add_instruction(migraphx::make_op("dot"), chunk(*mm, output_matrix, i), s));
            auto state_update =
                mm->add_instruction(migraphx::make_op("dot"), chunk(*mm, update_matrix, i), u);
            s = add_pointwise(p1,
                              "main:pointwise" + std::to_string(2 * i + 1),
                              {s, state_update},
                              single_pointwise("add"));
        }
        auto o = mm->add_instruction(migraphx::make_op("concat", {{"axis", 2}}), outs);
        mm->add_return({o, s});
    }
    migraphx::program p2 = p1;
    run_pass(p1);
    EXPECT(p1.sort() == p2.sort());
}

// Nor the out pointwise: output_base - product and product - output_base
TEST_CASE(swapped_out_inputs_not_fused)
{
    auto ss = fshape({1, 2, 4, 4});
    auto ks = fshape({1, 2, 2, 4, 4});
    migraphx::program p1;
    {
        auto* mm           = p1.get_main_module();
        auto s0            = mm->add_parameter("s0", ss);
        auto output_matrix = mm->add_parameter("output_matrix", ks);
        auto state_update  = mm->add_parameter("state_update", ks);
        auto output_base   = mm->add_parameter("output_base", ks);
        auto s             = s0;
        std::vector<migraphx::instruction_ref> outs;
        for(std::size_t i = 0; i < 2; i++)
        {
            auto output_product =
                mm->add_instruction(migraphx::make_op("dot"), chunk(*mm, output_matrix, i), s);
            auto base = chunk(*mm, output_base, i);
            outs.push_back(
                add_pointwise(p1,
                              "main:pointwise" + std::to_string(2 * i),
                              i == 0 ? std::vector<migraphx::instruction_ref>{base, output_product}
                                     : std::vector<migraphx::instruction_ref>{output_product, base},
                              single_pointwise("sub")));
            s = add_pointwise(p1,
                              "main:pointwise" + std::to_string(2 * i + 1),
                              {s, chunk(*mm, state_update, i)},
                              single_pointwise("add"));
        }
        auto o = mm->add_instruction(migraphx::make_op("concat", {{"axis", 2}}), outs);
        mm->add_return({o, s});
    }
    migraphx::program p2 = p1;
    run_pass(p1);
    EXPECT(p1.sort() == p2.sort());
}

// A family operand may be any view of its chunk: the transposed keys stack
// as a transpose of the whole input
TEST_CASE(transposed_family)
{
    auto ss = fshape({1, 2, 4, 8});
    auto ks = fshape({1, 2, 2, 2, 4});
    auto vs = fshape({1, 2, 2, 2, 8});
    migraphx::program p1;
    {
        auto* mm = p1.get_main_module();
        auto s0  = mm->add_parameter("s0", ss);
        auto output_matrix = mm->add_parameter("output_matrix", ks);
        auto update_keys   = mm->add_parameter("update_keys", ks);
        auto update_input  = mm->add_parameter("update_input", vs);
        auto s   = s0;
        std::vector<migraphx::instruction_ref> outs;
        for(std::size_t i = 0; i < 2; i++)
        {
            outs.push_back(
                mm->add_instruction(migraphx::make_op("dot"), chunk(*mm, output_matrix, i), s));
            auto update_matrix =
                mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 1, 3, 2}}}),
                                    chunk(*mm, update_keys, i));
            auto state_update = mm->add_instruction(
                migraphx::make_op("dot"), update_matrix, chunk(*mm, update_input, i));
            s = add_pointwise(p1,
                              "main:pointwise" + std::to_string(i),
                              {s, state_update},
                              single_pointwise("add"));
        }
        auto o = mm->add_instruction(migraphx::make_op("concat", {{"axis", 2}}), outs);
        mm->add_return({o, s});
    }
    run_pass(p1);

    migraphx::program p2;
    {
        auto* mm = p2.get_main_module();
        auto s0  = mm->add_parameter("s0", ss);
        auto output_matrix = mm->add_parameter("output_matrix", ks);
        auto update_keys   = mm->add_parameter("update_keys", ks);
        auto update_input  = mm->add_parameter("update_input", vs);
        auto* step =
            create_pointwise_module(p2, "main:chunk_scan0:step", {s0, s0}, single_pointwise("add"));
        auto op = migraphx::make_op("gpu::chunk_scan",
                                    {{"output_shape", migraphx::to_value(fshape({1, 2, 4, 8}))},
                                     {"state_shape", migraphx::to_value(ss)},
                                     {"families",
                                      migraphx::to_value(std::vector<std::string>{
                                          "output_matrix", "update_matrix", "update_input"})},
                                     {"chunks", 2},
                                     {"has_state", true}});
        auto update_matrix = mm->add_instruction(
            migraphx::make_op("transpose", {{"permutation", {0, 1, 2, 4, 3}}}), update_keys);
        auto scan =
            mm->add_instruction(op, {s0, output_matrix, update_matrix, update_input}, {step});
        mm->add_return({get_tuple_elem(*mm, scan, 0), get_tuple_elem(*mm, scan, 1)});
    }
    EXPECT(p1.sort() == p2.sort());
}

// Views that only regroup the problem axes keep the matrices of the state:
// the families and the output are reshaped to and from the state's axes
TEST_CASE(state_regrouped_view)
{
    auto ss = fshape({1, 2, 4, 8});
    auto ks = fshape({2, 2, 2, 4});
    auto us = fshape({1, 2, 2, 4, 8});
    migraphx::program p1;
    {
        auto* mm = p1.get_main_module();
        auto s0  = mm->add_parameter("s0", ss);
        auto output_matrix = mm->add_parameter("output_matrix", ks);
        auto state_update  = mm->add_parameter("state_update", us);
        auto s   = s0;
        std::vector<migraphx::instruction_ref> outs;
        for(std::size_t i = 0; i < 2; i++)
        {
            auto output_matrix_i = mm->add_instruction(
                migraphx::make_op("slice", {{"axes", {1}}, {"starts", {i}}, {"ends", {i + 1}}}),
                output_matrix);
            auto sv =
                mm->add_instruction(migraphx::make_op("reshape", {{"dims", {2, 1, 4, 8}}}), s);
            outs.push_back(mm->add_instruction(migraphx::make_op("dot"), output_matrix_i, sv));
            s = add_pointwise(p1,
                              "main:pointwise" + std::to_string(i),
                              {s, chunk(*mm, state_update, i)},
                              single_pointwise("add"));
        }
        auto o = mm->add_instruction(migraphx::make_op("concat", {{"axis", 1}}), outs);
        mm->add_return({o, s});
    }
    run_pass(p1);

    migraphx::program p2;
    {
        auto* mm = p2.get_main_module();
        auto s0  = mm->add_parameter("s0", ss);
        auto output_matrix = mm->add_parameter("output_matrix", ks);
        auto state_update  = mm->add_parameter("state_update", us);
        auto* step =
            create_pointwise_module(p2, "main:chunk_scan0:step", {s0, s0}, single_pointwise("add"));
        auto op = migraphx::make_op(
            "gpu::chunk_scan",
            {{"output_shape", migraphx::to_value(fshape({1, 2, 4, 8}))},
             {"state_shape", migraphx::to_value(ss)},
             {"families",
              migraphx::to_value(std::vector<std::string>{"output_matrix", "state_update"})},
             {"chunks", 2},
             {"has_state", true}});
        auto output_matrices =
            mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0}}}), output_matrix);
        auto scan = mm->add_instruction(op, {s0, output_matrices, state_update}, {step});
        auto o    = mm->add_instruction(migraphx::make_op("reshape", {{"dims", {2, 2, 2, 8}}}),
                                        get_tuple_elem(*mm, scan, 0));
        mm->add_return({o, get_tuple_elem(*mm, scan, 1)});
    }
    EXPECT(p1.sort() == p2.sort());
}

// A dot reading the transposed state is not a read of the state the scan
// keeps, even when the state is square and every shape agrees
TEST_CASE(transposed_state_read_not_fused)
{
    auto ss = fshape({1, 2, 4, 4});
    auto ks = fshape({1, 2, 2, 4, 4});
    migraphx::program p1;
    {
        auto* mm = p1.get_main_module();
        auto s0  = mm->add_parameter("s0", ss);
        auto output_matrix = mm->add_parameter("output_matrix", ks);
        auto state_update  = mm->add_parameter("state_update", ks);
        auto s   = s0;
        std::vector<migraphx::instruction_ref> outs;
        for(std::size_t i = 0; i < 2; i++)
        {
            auto st = mm->add_instruction(
                migraphx::make_op("transpose", {{"permutation", {0, 1, 3, 2}}}), s);
            outs.push_back(
                mm->add_instruction(migraphx::make_op("dot"), chunk(*mm, output_matrix, i), st));
            s = add_pointwise(p1,
                              "main:pointwise" + std::to_string(i),
                              {s, chunk(*mm, state_update, i)},
                              single_pointwise("add"));
        }
        auto o = mm->add_instruction(migraphx::make_op("concat", {{"axis", 2}}), outs);
        mm->add_return({o, s});
    }
    migraphx::program p2 = p1;
    run_pass(p1);
    EXPECT(p1.sort() == p2.sort());
}

// Nor is a dot reading only some rows of the state
TEST_CASE(sliced_state_read_not_fused)
{
    auto ss = fshape({1, 2, 4, 8});
    auto ks = fshape({1, 2, 2, 2, 2});
    auto us = fshape({1, 2, 2, 4, 8});
    migraphx::program p1;
    {
        auto* mm = p1.get_main_module();
        auto s0  = mm->add_parameter("s0", ss);
        auto output_matrix = mm->add_parameter("output_matrix", ks);
        auto state_update  = mm->add_parameter("state_update", us);
        auto s   = s0;
        std::vector<migraphx::instruction_ref> outs;
        for(std::size_t i = 0; i < 2; i++)
        {
            auto top = mm->add_instruction(
                migraphx::make_op("slice", {{"axes", {2}}, {"starts", {0}}, {"ends", {2}}}), s);
            outs.push_back(
                mm->add_instruction(migraphx::make_op("dot"), chunk(*mm, output_matrix, i), top));
            s = add_pointwise(p1,
                              "main:pointwise" + std::to_string(i),
                              {s, chunk(*mm, state_update, i)},
                              single_pointwise("add"));
        }
        auto o = mm->add_instruction(migraphx::make_op("concat", {{"axis", 2}}), outs);
        mm->add_return({o, s});
    }
    migraphx::program p2 = p1;
    run_pass(p1);
    EXPECT(p1.sort() == p2.sort());
}

// The read feeding the transition must read the state's matrices as well
TEST_CASE(transposed_state_in_read_not_fused)
{
    auto ss = fshape({1, 2, 4, 4});
    auto ks = fshape({1, 2, 2, 4, 4});
    migraphx::program p1;
    {
        auto* mm = p1.get_main_module();
        auto s0  = mm->add_parameter("s0", ss);
        auto read_matrix   = mm->add_parameter("read_matrix", ks);
        auto output_matrix = mm->add_parameter("output_matrix", ks);
        auto update_matrix = mm->add_parameter("update_matrix", ks);
        auto update_input  = mm->add_parameter("update_input", ks);
        auto s   = s0;
        std::vector<migraphx::instruction_ref> outs;
        for(std::size_t i = 0; i < 2; i++)
        {
            auto st = mm->add_instruction(
                migraphx::make_op("transpose", {{"permutation", {0, 1, 3, 2}}}), s);
            auto read_product =
                mm->add_instruction(migraphx::make_op("dot"), chunk(*mm, read_matrix, i), st);
            auto u = add_pointwise(p1,
                                   "main:pointwise" + std::to_string(2 * i),
                                   {chunk(*mm, update_input, i), read_product},
                                   single_pointwise("sub"));
            outs.push_back(
                mm->add_instruction(migraphx::make_op("dot"), chunk(*mm, output_matrix, i), s));
            auto state_update =
                mm->add_instruction(migraphx::make_op("dot"), chunk(*mm, update_matrix, i), u);
            s = add_pointwise(p1,
                              "main:pointwise" + std::to_string(2 * i + 1),
                              {s, state_update},
                              single_pointwise("add"));
        }
        auto o = mm->add_instruction(migraphx::make_op("concat", {{"axis", 2}}), outs);
        mm->add_return({o, s});
    }
    migraphx::program p2 = p1;
    run_pass(p1);
    EXPECT(p1.sort() == p2.sort());
}

// The scan keeps u = read(update_input, read_matrix S) in the order the read
// produces it, so the
// read's product and the update's operand cannot be reordered on the way
TEST_CASE(transposed_read_product_not_fused)
{
    auto ss = fshape({1, 2, 4, 4});
    auto ks = fshape({1, 2, 2, 4, 4});
    migraphx::program p1;
    {
        auto* mm = p1.get_main_module();
        auto s0  = mm->add_parameter("s0", ss);
        auto read_matrix   = mm->add_parameter("read_matrix", ks);
        auto output_matrix = mm->add_parameter("output_matrix", ks);
        auto update_matrix = mm->add_parameter("update_matrix", ks);
        auto update_input  = mm->add_parameter("update_input", ks);
        auto s   = s0;
        std::vector<migraphx::instruction_ref> outs;
        for(std::size_t i = 0; i < 2; i++)
        {
            auto read_product =
                mm->add_instruction(migraphx::make_op("dot"), chunk(*mm, read_matrix, i), s);
            auto read_product_t = mm->add_instruction(
                migraphx::make_op("transpose", {{"permutation", {0, 1, 3, 2}}}), read_product);
            auto u = add_pointwise(p1,
                                   "main:pointwise" + std::to_string(2 * i),
                                   {chunk(*mm, update_input, i), read_product_t},
                                   single_pointwise("sub"));
            outs.push_back(
                mm->add_instruction(migraphx::make_op("dot"), chunk(*mm, output_matrix, i), s));
            auto state_update =
                mm->add_instruction(migraphx::make_op("dot"), chunk(*mm, update_matrix, i), u);
            s = add_pointwise(p1,
                              "main:pointwise" + std::to_string(2 * i + 1),
                              {s, state_update},
                              single_pointwise("add"));
        }
        auto o = mm->add_instruction(migraphx::make_op("concat", {{"axis", 2}}), outs);
        mm->add_return({o, s});
    }
    migraphx::program p2 = p1;
    run_pass(p1);
    EXPECT(p1.sort() == p2.sort());
}

TEST_CASE(transposed_update_operand_not_fused)
{
    auto ss = fshape({1, 2, 4, 4});
    auto ks = fshape({1, 2, 2, 4, 4});
    migraphx::program p1;
    {
        auto* mm = p1.get_main_module();
        auto s0  = mm->add_parameter("s0", ss);
        auto read_matrix   = mm->add_parameter("read_matrix", ks);
        auto output_matrix = mm->add_parameter("output_matrix", ks);
        auto update_matrix = mm->add_parameter("update_matrix", ks);
        auto update_input  = mm->add_parameter("update_input", ks);
        auto s   = s0;
        std::vector<migraphx::instruction_ref> outs;
        for(std::size_t i = 0; i < 2; i++)
        {
            auto read_product =
                mm->add_instruction(migraphx::make_op("dot"), chunk(*mm, read_matrix, i), s);
            auto u = add_pointwise(p1,
                                   "main:pointwise" + std::to_string(2 * i),
                                   {chunk(*mm, update_input, i), read_product},
                                   single_pointwise("sub"));
            outs.push_back(
                mm->add_instruction(migraphx::make_op("dot"), chunk(*mm, output_matrix, i), s));
            auto ut = mm->add_instruction(
                migraphx::make_op("transpose", {{"permutation", {0, 1, 3, 2}}}), u);
            auto state_update =
                mm->add_instruction(migraphx::make_op("dot"), chunk(*mm, update_matrix, i), ut);
            s = add_pointwise(p1,
                              "main:pointwise" + std::to_string(2 * i + 1),
                              {s, state_update},
                              single_pointwise("add"));
        }
        auto o = mm->add_instruction(migraphx::make_op("concat", {{"axis", 2}}), outs);
        mm->add_return({o, s});
    }
    migraphx::program p2 = p1;
    run_pass(p1);
    EXPECT(p1.sort() == p2.sort());
}

// The scan writes each chunk's rows in order, so a chunk output cannot be
// reordered before it joins the concat
TEST_CASE(transposed_output_not_fused)
{
    auto ss = fshape({1, 2, 4, 4});
    auto ks = fshape({1, 2, 2, 4, 4});
    migraphx::program p1;
    {
        auto* mm = p1.get_main_module();
        auto s0  = mm->add_parameter("s0", ss);
        auto output_matrix = mm->add_parameter("output_matrix", ks);
        auto state_update  = mm->add_parameter("state_update", ks);
        auto s   = s0;
        std::vector<migraphx::instruction_ref> outs;
        for(std::size_t i = 0; i < 2; i++)
        {
            auto output_product =
                mm->add_instruction(migraphx::make_op("dot"), chunk(*mm, output_matrix, i), s);
            outs.push_back(mm->add_instruction(
                migraphx::make_op("transpose", {{"permutation", {0, 1, 3, 2}}}), output_product));
            s = add_pointwise(p1,
                              "main:pointwise" + std::to_string(i),
                              {s, chunk(*mm, state_update, i)},
                              single_pointwise("add"));
        }
        auto o = mm->add_instruction(migraphx::make_op("concat", {{"axis", 2}}), outs);
        mm->add_return({o, s});
    }
    migraphx::program p2 = p1;
    run_pass(p1);
    EXPECT(p1.sort() == p2.sort());
}

// Nor can the chunks join along the columns, which interleaves their rows
TEST_CASE(column_concat_not_fused)
{
    auto ss = fshape({1, 2, 4, 8});
    auto ks = fshape({1, 2, 2, 2, 4});
    auto us = fshape({1, 2, 2, 4, 8});
    migraphx::program p1;
    {
        auto* mm = p1.get_main_module();
        auto s0  = mm->add_parameter("s0", ss);
        auto output_matrix = mm->add_parameter("output_matrix", ks);
        auto state_update  = mm->add_parameter("state_update", us);
        auto s   = s0;
        std::vector<migraphx::instruction_ref> outs;
        for(std::size_t i = 0; i < 2; i++)
        {
            outs.push_back(
                mm->add_instruction(migraphx::make_op("dot"), chunk(*mm, output_matrix, i), s));
            s = add_pointwise(p1,
                              "main:pointwise" + std::to_string(i),
                              {s, chunk(*mm, state_update, i)},
                              single_pointwise("add"));
        }
        auto o = mm->add_instruction(migraphx::make_op("concat", {{"axis", 3}}), outs);
        mm->add_return({o, s});
    }
    migraphx::program p2 = p1;
    run_pass(p1);
    EXPECT(p1.sort() == p2.sort());
}

// The parser's unrolled loop compiles to the scan kernel after the whole
// pipeline has simplified it
TEST_CASE(compile_gated_delta_loop)
{
    auto ss  = fshape({1, 2, 4, 8});
    auto ks  = fshape({1, 2, 2, 2, 4});
    auto kts = fshape({1, 2, 2, 4, 2});
    auto vs  = fshape({1, 2, 2, 2, 8});
    auto gs  = fshape({1, 2, 2, 1, 1});
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto s0  = mm->add_parameter("s0", ss);
    auto read_matrix   = mm->add_parameter("read_matrix", ks);
    auto output_matrix = mm->add_parameter("output_matrix", ks);
    auto update_matrix = mm->add_parameter("update_matrix", kts);
    auto update_input  = mm->add_parameter("update_input", vs);
    auto output_base   = mm->add_parameter("output_base", vs);
    auto decay         = mm->add_parameter("decay", gs);
    auto s   = s0;
    std::vector<migraphx::instruction_ref> outs;
    for(std::size_t i = 0; i < 2; i++)
    {
        auto read_product =
            mm->add_instruction(migraphx::make_op("dot"), chunk(*mm, read_matrix, i), s);
        auto u = mm->add_instruction(
            migraphx::make_op("sub"), chunk(*mm, update_input, i), read_product);
        auto output_product =
            mm->add_instruction(migraphx::make_op("dot"), chunk(*mm, output_matrix, i), s);
        outs.push_back(mm->add_instruction(
            migraphx::make_op("add"), chunk(*mm, output_base, i), output_product));
        auto state_update =
            mm->add_instruction(migraphx::make_op("dot"), chunk(*mm, update_matrix, i), u);
        auto decay_i = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", ss.lens()}}), chunk(*mm, decay, i));
        auto sg = mm->add_instruction(migraphx::make_op("mul"), s, decay_i);
        s       = mm->add_instruction(migraphx::make_op("add"), sg, state_update);
    }
    auto o = mm->add_instruction(migraphx::make_op("concat", {{"axis", 2}}), outs);
    mm->add_return({o, s});
    p.compile(migraphx::make_target("gpu"));

    auto kernels = get_kernel_names(p);
    EXPECT(std::any_of(kernels.begin(), kernels.end(), [](const auto& k) {
        return migraphx::contains(k, "chunk_scan");
    }));
    EXPECT(std::none_of(kernels.begin(), kernels.end(), [](const auto& k) {
        return migraphx::contains(k, "mlir");
    }));
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
