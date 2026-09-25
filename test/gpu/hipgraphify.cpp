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
#include <migraphx/gpu/hipgraphify.hpp>
#include <migraphx/gpu/hip.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/program.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/operation.hpp>
#include <migraphx/serialize.hpp>
#include <basic_ops.hpp>
#include <test.hpp>

// A stub with the quantized rocblas gemm's name. It has no compute, so it is
// not context-free and would be capturable; only the is_unsupported name check
// makes it a boundary -- which is exactly what the test pins.
struct quant_gemm_stub
{
    std::string name() const { return "gpu::quant_gemm"; }
    migraphx::shape compute_shape(std::vector<migraphx::shape> inputs) const
    {
        return inputs.front();
    }
};

// A boundary stub (gpu::gemm is in the unsupported set) that writes into the
// allocation passed as its last input and aliases it, like the lowered library
// ops do.
struct gemm_alias_stub
{
    std::string name() const { return "gpu::gemm"; }
    migraphx::shape compute_shape(std::vector<migraphx::shape> inputs) const
    {
        return inputs.back();
    }
    std::vector<std::size_t> output_alias(const std::vector<migraphx::shape>& shapes) const
    {
        return {shapes.size() - 1};
    }
};

// A capturable view: aliases its input, and with no compute it is not
// context-free, so is_capturable keeps it.
struct view_pass_op
{
    std::string name() const { return "view_pass"; }
    migraphx::shape compute_shape(std::vector<migraphx::shape> inputs) const
    {
        return inputs.front();
    }
    std::vector<std::size_t> output_alias(const std::vector<migraphx::shape>&) const { return {0}; }
};

// A context-free op that does not alias its input: it stands in for a host/ref
// op that cannot be captured into a HIP graph and so acts as a boundary.
struct host_op
{
    std::string name() const { return "host_op"; }
    migraphx::shape compute_shape(std::vector<migraphx::shape> inputs) const
    {
        if(inputs.empty())
            return {};
        return inputs.front();
    }
    migraphx::argument compute(const migraphx::shape&, std::vector<migraphx::argument> args) const
    {
        return args.empty() ? migraphx::argument{} : args.front();
    }
};

static void run_pass(migraphx::program& p, std::size_t min_size = 4)
{
    migraphx::run_passes(p,
                         {migraphx::gpu::hipgraphify{min_size}, migraphx::dead_code_elimination{}});
}

// Add a chain of n capturable (unary_pass) instructions starting from `start`.
static migraphx::instruction_ref
add_chain(migraphx::module& m, migraphx::instruction_ref start, std::size_t n)
{
    auto ins = start;
    for(std::size_t i = 0; i < n; ++i)
        ins = m.add_instruction(unary_pass_op{}, ins);
    return ins;
}

// A run of exactly min_partition_size capturable ops is extracted into one graph.
TEST_CASE(basic_capture)
{
    migraphx::program p1;
    {
        auto* mm = p1.get_main_module();
        auto x   = mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {4}});
        auto c   = add_chain(*mm, x, 4);
        mm->add_return({c});
    }
    run_pass(p1);

    migraphx::program p2;
    {
        auto* mm  = p2.get_main_module();
        auto x    = mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {4}});
        auto* sub = p2.create_module("main:hipgraph0");
        auto x0   = sub->add_parameter("x0", migraphx::shape{migraphx::shape::float_type, {4}});
        auto last = add_chain(*sub, x0, 4);
        sub->add_return({last});
        auto g = mm->add_instruction(
            migraphx::make_op("hip::graph", {{"replace_inputs", {0}}}), {x}, {sub});
        mm->add_return({g});
    }

    EXPECT(p1.sort() == p2.sort());
}

// A run shorter than min_partition_size is left untouched.
TEST_CASE(run_too_short)
{
    migraphx::program p1;
    {
        auto* mm = p1.get_main_module();
        auto x   = mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {4}});
        auto c   = add_chain(*mm, x, 3);
        mm->add_return({c});
    }
    run_pass(p1);

    migraphx::program p2;
    {
        auto* mm = p2.get_main_module();
        auto x   = mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {4}});
        auto c   = add_chain(*mm, x, 3);
        mm->add_return({c});
    }

    EXPECT(p1.sort() == p2.sort());
}

// A non-capturable `boundary` op between two chains splits the module into two
// runs, each captured into its own submodule.
static void check_boundary_split(const migraphx::operation& boundary)
{
    migraphx::program p1;
    {
        auto* mm = p1.get_main_module();
        auto x   = mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {4}});
        auto c   = add_chain(*mm, x, 4);
        auto b   = mm->add_instruction(boundary, c);
        auto d   = add_chain(*mm, b, 4);
        mm->add_return({d});
    }
    run_pass(p1);

    migraphx::program p2;
    {
        auto* mm   = p2.get_main_module();
        auto x     = mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {4}});
        auto* sub0 = p2.create_module("main:hipgraph0");
        auto x0    = sub0->add_parameter("x0", migraphx::shape{migraphx::shape::float_type, {4}});
        sub0->add_return({add_chain(*sub0, x0, 4)});
        auto g0 = mm->add_instruction(
            migraphx::make_op("hip::graph", {{"replace_inputs", {0}}}), {x}, {sub0});
        auto b     = mm->add_instruction(boundary, g0);
        auto* sub1 = p2.create_module("main:hipgraph1");
        auto y0    = sub1->add_parameter("x0", migraphx::shape{migraphx::shape::float_type, {4}});
        sub1->add_return({add_chain(*sub1, y0, 4)});
        auto g1 = mm->add_instruction(migraphx::make_op("hip::graph"), {b}, {sub1});
        mm->add_return({g1});
    }

    EXPECT(p1.sort() == p2.sort());
}

// A host-synchronizing op (hip::sync_stream) is a partition boundary.
TEST_CASE(split_by_sync) { check_boundary_split(migraphx::gpu::hip_sync_stream{}); }

// A context-free non-aliasing host op is also a boundary.
TEST_CASE(host_op_boundary) { check_boundary_split(host_op{}); }

// A quantized rocblas gemm is the same library call as gpu::gemm and is also a
// partition boundary.
TEST_CASE(split_by_quant_gemm) { check_boundary_split(quant_gemm_stub{}); }

// A long run is captured while a trailing run below the threshold is left alone.
TEST_CASE(mixed_long_short)
{
    migraphx::program p1;
    {
        auto* mm  = p1.get_main_module();
        auto x    = mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {4}});
        auto c    = add_chain(*mm, x, 5);
        auto sync = mm->add_instruction(migraphx::gpu::hip_sync_stream{}, c);
        auto d    = add_chain(*mm, sync, 2);
        mm->add_return({d});
    }
    run_pass(p1);

    migraphx::program p2;
    {
        auto* mm   = p2.get_main_module();
        auto x     = mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {4}});
        auto* sub0 = p2.create_module("main:hipgraph0");
        auto x0    = sub0->add_parameter("x0", migraphx::shape{migraphx::shape::float_type, {4}});
        sub0->add_return({add_chain(*sub0, x0, 5)});
        auto g0 = mm->add_instruction(
            migraphx::make_op("hip::graph", {{"replace_inputs", {0}}}), {x}, {sub0});
        auto sync = mm->add_instruction(migraphx::gpu::hip_sync_stream{}, g0);
        auto d    = add_chain(*mm, sync, 2);
        mm->add_return({d});
    }

    EXPECT(p1.sort() == p2.sort());
}

// min_partition_size is configurable: a run of 3 is captured when min is 2.
TEST_CASE(min_partition_size_config)
{
    migraphx::program p1;
    {
        auto* mm = p1.get_main_module();
        auto x   = mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {4}});
        auto c   = add_chain(*mm, x, 3);
        mm->add_return({c});
    }
    run_pass(p1, 2);

    migraphx::program p2;
    {
        auto* mm  = p2.get_main_module();
        auto x    = mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {4}});
        auto* sub = p2.create_module("main:hipgraph0");
        auto x0   = sub->add_parameter("x0", migraphx::shape{migraphx::shape::float_type, {4}});
        sub->add_return({add_chain(*sub, x0, 3)});
        auto g = mm->add_instruction(
            migraphx::make_op("hip::graph", {{"replace_inputs", {0}}}), {x}, {sub});
        mm->add_return({g});
    }

    EXPECT(p1.sort() == p2.sort());
}

// A run with two externally-used outputs produces a tuple plus get_tuple_elem.
TEST_CASE(multi_output)
{
    migraphx::program p1;
    {
        auto* mm = p1.get_main_module();
        auto x   = mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {4}});
        auto c0  = mm->add_instruction(unary_pass_op{}, x);
        auto c1  = mm->add_instruction(unary_pass_op{}, c0);
        auto c2  = mm->add_instruction(unary_pass_op{}, c1);
        auto c3  = mm->add_instruction(unary_pass_op{}, c2);
        auto s1  = mm->add_instruction(migraphx::gpu::hip_sync_stream{}, c1);
        auto s2  = mm->add_instruction(migraphx::gpu::hip_sync_stream{}, c3);
        mm->add_return({s1, s2});
    }
    run_pass(p1);

    migraphx::program p2;
    {
        auto* mm  = p2.get_main_module();
        auto x    = mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {4}});
        auto* sub = p2.create_module("main:hipgraph0");
        auto x0   = sub->add_parameter("x0", migraphx::shape{migraphx::shape::float_type, {4}});
        auto sc0  = sub->add_instruction(unary_pass_op{}, x0);
        auto sc1  = sub->add_instruction(unary_pass_op{}, sc0);
        auto sc2  = sub->add_instruction(unary_pass_op{}, sc1);
        auto sc3  = sub->add_instruction(unary_pass_op{}, sc2);
        sub->add_return({sc1, sc3});
        auto g = mm->add_instruction(
            migraphx::make_op("hip::graph", {{"replace_inputs", {0}}}), {x}, {sub});
        auto e0 = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), g);
        auto e1 = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 1}}), g);
        auto s1 = mm->add_instruction(migraphx::gpu::hip_sync_stream{}, e0);
        auto s2 = mm->add_instruction(migraphx::gpu::hip_sync_stream{}, e1);
        mm->add_return({s1, s2});
    }

    EXPECT(p1.sort() == p2.sort());
}

// Only the root module is partitioned; a pre-existing submodule is untouched.
TEST_CASE(root_only)
{
    migraphx::program p1;
    {
        auto* mm   = p1.get_main_module();
        auto x     = mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {4}});
        auto* body = p1.create_module("body");
        auto x0    = body->add_parameter("x0", migraphx::shape{migraphx::shape::float_type, {4}});
        body->add_return({add_chain(*body, x0, 4)});
        auto r = mm->add_instruction(mod_pass_op{}, {x}, {body});
        mm->add_return({r});
    }
    run_pass(p1);

    migraphx::program p2;
    {
        auto* mm   = p2.get_main_module();
        auto x     = mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {4}});
        auto* body = p2.create_module("body");
        auto x0    = body->add_parameter("x0", migraphx::shape{migraphx::shape::float_type, {4}});
        body->add_return({add_chain(*body, x0, 4)});
        auto r = mm->add_instruction(mod_pass_op{}, {x}, {body});
        mm->add_return({r});
    }

    EXPECT(p1.sort() == p2.sort());
}

// Applying the pass twice is the same as applying it once: its own hip::graph
// product is a partition boundary, so a second run has nothing to wrap.
TEST_CASE(idempotent)
{
    migraphx::program p1;
    {
        auto* mm = p1.get_main_module();
        auto x   = mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {4}});
        auto c   = add_chain(*mm, x, 4);
        mm->add_return({c});
    }
    migraphx::program p2 = p1;
    run_pass(p1);
    run_pass(p1);
    run_pass(p2);

    EXPECT(p1.sort() == p2.sort());
}

// One external output writes into an allocation while another is not
// allocation-backed: the pass only captures a run whose outputs are all
// allocation-backed or all not, so the mixed run is left uncaptured.
TEST_CASE(mixed_output_backing)
{
    migraphx::shape s{migraphx::shape::float_type, {4}};
    migraphx::program p1;
    {
        auto* mm = p1.get_main_module();
        auto x   = mm->add_parameter("x", s);
        auto c1  = mm->add_instruction(unary_pass_op{}, x);
        auto c2  = mm->add_instruction(unary_pass_op{}, c1);
        auto a = mm->add_instruction(migraphx::make_op("hip::allocate", {{"shape", to_value(s)}}));
        auto v = mm->add_instruction(view_pass_op{}, a);
        auto sync1 = mm->add_instruction(migraphx::gpu::hip_sync_stream{}, c2);
        auto sync2 = mm->add_instruction(migraphx::gpu::hip_sync_stream{}, v);
        mm->add_return({sync1, sync2});
    }
    migraphx::program p2 = p1;
    run_pass(p1);

    EXPECT(p1.sort() == p2.sort());
}

// An external output whose alias root is an allocation outside the run, reached
// only through views, cannot become an input of the hip::graph op (so no alias
// index could refer to it); the run is left uncaptured.
TEST_CASE(output_root_outside_run)
{
    migraphx::shape s{migraphx::shape::float_type, {4}};
    migraphx::program p1;
    {
        auto* mm = p1.get_main_module();
        auto x   = mm->add_parameter("x", s);
        auto a1 = mm->add_instruction(migraphx::make_op("hip::allocate", {{"shape", to_value(s)}}));
        auto g1 = mm->add_instruction(gemm_alias_stub{}, x, a1);
        auto v  = g1;
        for(std::size_t i = 0; i < 4; ++i)
            v = mm->add_instruction(view_pass_op{}, v);
        auto sync = mm->add_instruction(migraphx::gpu::hip_sync_stream{}, v);
        mm->add_return({sync});
    }
    migraphx::program p2 = p1;
    run_pass(p1);

    EXPECT(p1.sort() == p2.sort());
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
