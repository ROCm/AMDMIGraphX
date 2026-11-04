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
#include <pointwise.hpp>

static void run_pass(migraphx::program& p)
{
    migraphx::run_passes(p, {migraphx::gpu::fuse_chunk_split{}, migraphx::dead_code_elimination{}});
}

static migraphx::shape fshape(std::vector<std::size_t> lens)
{
    return {migraphx::shape::float_type, std::move(lens)};
}

static migraphx::shape fview(std::vector<std::size_t> lens, std::vector<std::size_t> strides)
{
    return {migraphx::shape::float_type, std::move(lens), std::move(strides)};
}

// Chunk i of a batched per-chunk operand, the view the parser emits
static migraphx::instruction_ref
chunk(migraphx::module& m, migraphx::instruction_ref x, std::size_t i)
{
    auto s = m.add_instruction(
        migraphx::make_op("slice", {{"axes", {2}}, {"starts", {i}}, {"ends", {i + 1}}}), x);
    return m.add_instruction(migraphx::make_op("squeeze", {{"axes", {2}}}), s);
}

// S * gamma + upd
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

// The kernels launched by the program, without the output copies
static std::vector<std::string> get_kernel_names(migraphx::program& p)
{
    std::vector<std::string> result;
    auto im = migraphx::iterator_for(*p.get_main_module());
    migraphx::transform_if(
        im.begin(),
        im.end(),
        std::back_inserter(result),
        [](auto ins) {
            if(ins->name() != "gpu::code_object")
                return false;
            const auto& op = migraphx::any_cast<migraphx::gpu::code_object_op>(ins->get_operator());
            return op.symbol_name != "hip_copy_kernel";
        },
        [](auto ins) {
            return migraphx::any_cast<migraphx::gpu::code_object_op>(ins->get_operator())
                .symbol_name;
        });
    return result;
}

// Two chunks of two rows over two heads with a past state, the gated delta
// rule: u = u0 - w S, o = pu0 + q S, S = S * gamma + kt u
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
        auto w   = mm->add_parameter("w", ks);
        auto q   = mm->add_parameter("q", ks);
        auto kt  = mm->add_parameter("kt", kts);
        auto u0  = mm->add_parameter("u0", vs);
        auto pu0 = mm->add_parameter("pu0", vs);
        auto g   = mm->add_parameter("g", gs);
        auto s   = s0;
        std::vector<migraphx::instruction_ref> outs;
        for(std::size_t i = 0; i < 2; i++)
        {
            auto n  = std::to_string(3 * i);
            auto ws = mm->add_instruction(migraphx::make_op("dot"), chunk(*mm, w, i), s);
            auto u  = add_pointwise(
                p1, "main:pointwise" + n, {chunk(*mm, u0, i), ws}, single_pointwise("sub"));
            auto qs = mm->add_instruction(migraphx::make_op("dot"), chunk(*mm, q, i), s);
            outs.push_back(add_pointwise(p1,
                                         "main:pointwise" + std::to_string(3 * i + 1),
                                         {chunk(*mm, pu0, i), qs},
                                         single_pointwise("add")));
            auto upd = mm->add_instruction(migraphx::make_op("dot"), chunk(*mm, kt, i), u);
            auto gi  = mm->add_instruction(
                migraphx::make_op("multibroadcast", {{"out_lens", ss.lens()}}), chunk(*mm, g, i));
            s = add_pointwise(
                p1, "main:pointwise" + std::to_string(3 * i + 2), {s, gi, upd}, gated_step());
        }
        auto o = mm->add_instruction(migraphx::make_op("concat", {{"axis", 2}}), outs);
        mm->add_return({o, s});
    }
    run_pass(p1);

    migraphx::program p2;
    {
        auto* mm = p2.get_main_module();
        auto s0  = mm->add_parameter("s0", ss);
        auto w   = mm->add_parameter("w", ks);
        auto q   = mm->add_parameter("q", ks);
        auto kt  = mm->add_parameter("kt", kts);
        auto u0  = mm->add_parameter("u0", vs);
        auto pu0 = mm->add_parameter("pu0", vs);
        auto g   = mm->add_parameter("g", gs);
        auto* step =
            create_pointwise_module(p2, "main:chunk_scan0:step", {s0, s0, s0}, gated_step());
        auto* read =
            create_pointwise_module(p2, "main:chunk_scan0:read", {s0, s0}, single_pointwise("sub"));
        auto* out =
            create_pointwise_module(p2, "main:chunk_scan0:out", {s0, s0}, single_pointwise("add"));
        auto op   = migraphx::make_op("gpu::chunk_scan",
                                      {{"output_shape", migraphx::to_value(fshape({1, 2, 4, 8}))},
                                       {"state_shape", migraphx::to_value(ss)},
                                       {"views",
                                        migraphx::to_value(std::vector<migraphx::shape>{
                                            fview({1, 2, 2, 4}, {32, 16, 4, 1}),
                                            fview({1, 2, 2, 4}, {32, 16, 4, 1}),
                                            fview({1, 2, 4, 2}, {32, 16, 2, 1}),
                                            fview({1, 2, 2, 8}, {64, 32, 8, 1}),
                                            fview({1, 2, 4, 8}, {4, 2, 0, 0}),
                                            fview({1, 2, 2, 8}, {64, 32, 8, 1})})},
                                       {"offsets", {0, 0, 0, 0, 0, 0}},
                                       {"deltas", {8, 8, 8, 16, 1, 16}},
                                       {"chunks", 2},
                                       {"has_state", true},
                                       {"has_update_dot", true},
                                       {"has_read", true},
                                       {"has_gamma", true},
                                       {"has_out", true}});
        auto scan = mm->add_instruction(op, {s0, w, q, kt, u0, g, pu0}, {step, read, out});
        mm->add_return({get_tuple_elem(*mm, scan, 0), get_tuple_elem(*mm, scan, 1)});
    }
    EXPECT(p1.sort() == p2.sort());
}

// Three chunks without a past state under the delta rule: the first chunk
// emits its pu0 and S_1 = kt_0 u0_0 directly, so its w and q are absent
TEST_CASE(delta_zero_start)
{
    auto ss  = fshape({1, 2, 4, 8});
    auto ks  = fshape({1, 2, 3, 2, 4});
    auto kts = fshape({1, 2, 3, 4, 2});
    auto vs  = fshape({1, 2, 3, 2, 8});
    migraphx::program p1;
    {
        auto* mm = p1.get_main_module();
        auto w   = mm->add_parameter("w", ks);
        auto q   = mm->add_parameter("q", ks);
        auto kt  = mm->add_parameter("kt", kts);
        auto u0  = mm->add_parameter("u0", vs);
        auto pu0 = mm->add_parameter("pu0", vs);
        std::vector<migraphx::instruction_ref> outs{chunk(*mm, pu0, 0)};
        auto s =
            mm->add_instruction(migraphx::make_op("dot"), chunk(*mm, kt, 0), chunk(*mm, u0, 0));
        for(std::size_t i = 1; i < 3; i++)
        {
            auto ws = mm->add_instruction(migraphx::make_op("dot"), chunk(*mm, w, i), s);
            auto u  = add_pointwise(p1,
                                    "main:pointwise" + std::to_string(3 * i),
                                    {chunk(*mm, u0, i), ws},
                                    single_pointwise("sub"));
            auto qs = mm->add_instruction(migraphx::make_op("dot"), chunk(*mm, q, i), s);
            outs.push_back(add_pointwise(p1,
                                         "main:pointwise" + std::to_string(3 * i + 1),
                                         {chunk(*mm, pu0, i), qs},
                                         single_pointwise("add")));
            auto upd = mm->add_instruction(migraphx::make_op("dot"), chunk(*mm, kt, i), u);
            s        = add_pointwise(p1,
                                     "main:pointwise" + std::to_string(3 * i + 2),
                                     {s, upd},
                                     single_pointwise("add"));
        }
        auto o = mm->add_instruction(migraphx::make_op("concat", {{"axis", 2}}), outs);
        mm->add_return({o, s});
    }
    run_pass(p1);

    migraphx::program p2;
    {
        auto* mm = p2.get_main_module();
        auto w   = mm->add_parameter("w", ks);
        auto q   = mm->add_parameter("q", ks);
        auto kt  = mm->add_parameter("kt", kts);
        auto u0  = mm->add_parameter("u0", vs);
        auto pu0 = mm->add_parameter("pu0", vs);
        auto* step =
            create_pointwise_module(p2, "main:chunk_scan0:step", {u0, u0}, single_pointwise("add"));
        auto* read =
            create_pointwise_module(p2, "main:chunk_scan0:read", {u0, u0}, single_pointwise("sub"));
        auto* out =
            create_pointwise_module(p2, "main:chunk_scan0:out", {u0, u0}, single_pointwise("add"));
        auto op   = migraphx::make_op("gpu::chunk_scan",
                                      {{"output_shape", migraphx::to_value(fshape({1, 2, 6, 8}))},
                                       {"state_shape", migraphx::to_value(ss)},
                                       {"views",
                                        migraphx::to_value(std::vector<migraphx::shape>{
                                            fview({1, 2, 2, 4}, {48, 24, 4, 1}),
                                            fview({1, 2, 2, 4}, {48, 24, 4, 1}),
                                            fview({1, 2, 4, 2}, {48, 24, 2, 1}),
                                            fview({1, 2, 2, 8}, {96, 48, 8, 1}),
                                            fview({1, 2, 2, 8}, {96, 48, 8, 1})})},
                                       {"offsets", {0, 0, 0, 0, 0}},
                                       {"deltas", {8, 8, 8, 16, 16}},
                                       {"chunks", 3},
                                       {"has_state", false},
                                       {"has_update_dot", true},
                                       {"has_read", true},
                                       {"has_gamma", false},
                                       {"has_out", true}});
        auto scan = mm->add_instruction(op, {w, q, kt, u0, pu0}, {step, read, out});
        mm->add_return({get_tuple_elem(*mm, scan, 0), get_tuple_elem(*mm, scan, 1)});
    }
    EXPECT(p1.sort() == p2.sort());
}

// The update is a precomputed per-chunk view and the outputs are the bare
// reads q S: only the step module is needed
TEST_CASE(precomputed_update)
{
    auto ss = fshape({1, 2, 4, 8});
    auto ks = fshape({1, 2, 2, 2, 4});
    auto us = fshape({1, 2, 2, 4, 8});
    migraphx::program p1;
    {
        auto* mm = p1.get_main_module();
        auto s0  = mm->add_parameter("s0", ss);
        auto q   = mm->add_parameter("q", ks);
        auto upd = mm->add_parameter("upd", us);
        auto s   = s0;
        std::vector<migraphx::instruction_ref> outs;
        for(std::size_t i = 0; i < 2; i++)
        {
            outs.push_back(mm->add_instruction(migraphx::make_op("dot"), chunk(*mm, q, i), s));
            s = add_pointwise(p1,
                              "main:pointwise" + std::to_string(i),
                              {s, chunk(*mm, upd, i)},
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
        auto q   = mm->add_parameter("q", ks);
        auto upd = mm->add_parameter("upd", us);
        auto* step =
            create_pointwise_module(p2, "main:chunk_scan0:step", {s0, s0}, single_pointwise("add"));
        auto op = migraphx::make_op(
            "gpu::chunk_scan",
            {{"output_shape", migraphx::to_value(fshape({1, 2, 4, 8}))},
             {"state_shape", migraphx::to_value(ss)},
             {"views",
              migraphx::to_value(std::vector<migraphx::shape>{
                  fview({1, 2, 2, 4}, {32, 16, 4, 1}), fview({1, 2, 4, 8}, {128, 64, 8, 1})})},
             {"offsets", {0, 0}},
             {"deltas", {8, 32}},
             {"chunks", 2},
             {"has_state", true},
             {"has_update_dot", false},
             {"has_read", false},
             {"has_gamma", false},
             {"has_out", false}});
        auto scan = mm->add_instruction(op, {s0, q, upd}, {step});
        mm->add_return({get_tuple_elem(*mm, scan, 0), get_tuple_elem(*mm, scan, 1)});
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
        auto q   = mm->add_parameter("q", ks);
        auto q2  = mm->add_parameter("q2", ks);
        auto upd = mm->add_parameter("upd", us);
        auto s   = s0;
        std::vector<migraphx::instruction_ref> outs1;
        std::vector<migraphx::instruction_ref> outs2;
        for(std::size_t i = 0; i < 2; i++)
        {
            outs1.push_back(mm->add_instruction(migraphx::make_op("dot"), chunk(*mm, q, i), s));
            auto upd1 = mm->add_instruction(
                migraphx::make_op("slice",
                                  {{"axes", {2, 3}}, {"starts", {0, i}}, {"ends", {1, i + 1}}}),
                upd);
            s = add_pointwise(
                p1,
                "main:pointwise" + std::to_string(i),
                {s, mm->add_instruction(migraphx::make_op("squeeze", {{"axes", {2, 3}}}), upd1)},
                single_pointwise("add"));
        }
        for(std::size_t i = 0; i < 2; i++)
        {
            outs2.push_back(mm->add_instruction(migraphx::make_op("dot"), chunk(*mm, q2, i), s));
            auto upd2 = mm->add_instruction(
                migraphx::make_op("slice",
                                  {{"axes", {2, 3}}, {"starts", {1, i}}, {"ends", {2, i + 1}}}),
                upd);
            s = add_pointwise(
                p1,
                "main:pointwise" + std::to_string(2 + i),
                {s, mm->add_instruction(migraphx::make_op("squeeze", {{"axes", {2, 3}}}), upd2)},
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
        auto q   = mm->add_parameter("q", ks);
        auto q2  = mm->add_parameter("q2", ks);
        auto upd = mm->add_parameter("upd", us);
        auto* step0 =
            create_pointwise_module(p2, "main:chunk_scan0:step", {s0, s0}, single_pointwise("add"));
        auto* step1 =
            create_pointwise_module(p2, "main:chunk_scan1:step", {s0, s0}, single_pointwise("add"));
        auto scan_op = [&](std::size_t offset) {
            return migraphx::make_op(
                "gpu::chunk_scan",
                {{"output_shape", migraphx::to_value(fshape({1, 2, 4, 8}))},
                 {"state_shape", migraphx::to_value(ss)},
                 {"views",
                  migraphx::to_value(std::vector<migraphx::shape>{
                      fview({1, 2, 2, 4}, {32, 16, 4, 1}), fview({1, 2, 4, 8}, {256, 128, 8, 1})})},
                 {"offsets", {0, offset}},
                 {"deltas", {8, 32}},
                 {"chunks", 2},
                 {"has_state", true},
                 {"has_update_dot", false},
                 {"has_read", false},
                 {"has_gamma", false},
                 {"has_out", false}});
        };
        auto scan0 = mm->add_instruction(scan_op(0), {s0, q, upd}, {step0});
        auto s1    = get_tuple_elem(*mm, scan0, 1);
        auto scan1 = mm->add_instruction(scan_op(64), {s1, q2, upd}, {step1});
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
        auto q   = mm->add_parameter("q", ks);
        auto upd = mm->add_parameter("upd", us);
        auto qs0 = mm->add_instruction(migraphx::make_op("dot"), chunk(*mm, q, 0), s0);
        auto s1 =
            add_pointwise(p1, "main:pointwise0", {s0, chunk(*mm, upd, 0)}, single_pointwise("add"));
        auto qs1 = mm->add_instruction(migraphx::make_op("dot"), chunk(*mm, q, 1), s1);
        auto s2 =
            add_pointwise(p1, "main:pointwise1", {s1, chunk(*mm, upd, 1)}, single_pointwise("sub"));
        auto o = mm->add_instruction(migraphx::make_op("concat", {{"axis", 2}}), qs0, qs1);
        mm->add_return({o, s2});
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
    auto w   = mm->add_parameter("w", ks);
    auto q   = mm->add_parameter("q", ks);
    auto kt  = mm->add_parameter("kt", kts);
    auto u0  = mm->add_parameter("u0", vs);
    auto pu0 = mm->add_parameter("pu0", vs);
    auto g   = mm->add_parameter("g", gs);
    auto s   = s0;
    std::vector<migraphx::instruction_ref> outs;
    for(std::size_t i = 0; i < 2; i++)
    {
        auto ws = mm->add_instruction(migraphx::make_op("dot"), chunk(*mm, w, i), s);
        auto u  = mm->add_instruction(migraphx::make_op("sub"), chunk(*mm, u0, i), ws);
        auto qs = mm->add_instruction(migraphx::make_op("dot"), chunk(*mm, q, i), s);
        outs.push_back(mm->add_instruction(migraphx::make_op("add"), chunk(*mm, pu0, i), qs));
        auto upd = mm->add_instruction(migraphx::make_op("dot"), chunk(*mm, kt, i), u);
        auto gi  = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", ss.lens()}}), chunk(*mm, g, i));
        auto sg = mm->add_instruction(migraphx::make_op("mul"), s, gi);
        s       = mm->add_instruction(migraphx::make_op("add"), sg, upd);
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
