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
#include <basic_ops.hpp>
#include <test.hpp>

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

static const migraphx::shape s{migraphx::shape::float_type, {4}};

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
        auto x   = mm->add_parameter("x", s);
        auto c   = add_chain(*mm, x, 4);
        mm->add_return({c});
    }
    run_pass(p1);

    migraphx::program p2;
    {
        auto* mm  = p2.get_main_module();
        auto x    = mm->add_parameter("x", s);
        auto* sub = p2.create_module("main:hipgraph0");
        auto x0   = sub->add_parameter("x0", s);
        auto last = add_chain(*sub, x0, 4);
        sub->add_return({last});
        auto g = mm->add_instruction(migraphx::make_op("hip::graph"), {x}, {sub});
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
        auto x   = mm->add_parameter("x", s);
        auto c   = add_chain(*mm, x, 3);
        mm->add_return({c});
    }
    run_pass(p1);

    migraphx::program p2;
    {
        auto* mm = p2.get_main_module();
        auto x   = mm->add_parameter("x", s);
        auto c   = add_chain(*mm, x, 3);
        mm->add_return({c});
    }

    EXPECT(p1.sort() == p2.sort());
}

// A host-synchronizing op (hip::sync_stream) splits the module into two runs,
// each captured into its own submodule.
TEST_CASE(split_by_sync)
{
    migraphx::program p1;
    {
        auto* mm  = p1.get_main_module();
        auto x    = mm->add_parameter("x", s);
        auto c    = add_chain(*mm, x, 4);
        auto sync = mm->add_instruction(migraphx::gpu::hip_sync_stream{}, c);
        auto d    = add_chain(*mm, sync, 4);
        mm->add_return({d});
    }
    run_pass(p1);

    migraphx::program p2;
    {
        auto* mm   = p2.get_main_module();
        auto x     = mm->add_parameter("x", s);
        auto* sub0 = p2.create_module("main:hipgraph0");
        auto x0    = sub0->add_parameter("x0", s);
        sub0->add_return({add_chain(*sub0, x0, 4)});
        auto g0    = mm->add_instruction(migraphx::make_op("hip::graph"), {x}, {sub0});
        auto sync  = mm->add_instruction(migraphx::gpu::hip_sync_stream{}, g0);
        auto* sub1 = p2.create_module("main:hipgraph1");
        auto y0    = sub1->add_parameter("x0", s);
        sub1->add_return({add_chain(*sub1, y0, 4)});
        auto g1 = mm->add_instruction(migraphx::make_op("hip::graph"), {sync}, {sub1});
        mm->add_return({g1});
    }

    EXPECT(p1.sort() == p2.sort());
}

// A context-free non-aliasing host op is also a boundary.
TEST_CASE(host_op_boundary)
{
    migraphx::program p1;
    {
        auto* mm = p1.get_main_module();
        auto x   = mm->add_parameter("x", s);
        auto c   = add_chain(*mm, x, 4);
        auto h   = mm->add_instruction(host_op{}, c);
        auto d   = add_chain(*mm, h, 4);
        mm->add_return({d});
    }
    run_pass(p1);

    migraphx::program p2;
    {
        auto* mm   = p2.get_main_module();
        auto x     = mm->add_parameter("x", s);
        auto* sub0 = p2.create_module("main:hipgraph0");
        auto x0    = sub0->add_parameter("x0", s);
        sub0->add_return({add_chain(*sub0, x0, 4)});
        auto g0    = mm->add_instruction(migraphx::make_op("hip::graph"), {x}, {sub0});
        auto h     = mm->add_instruction(host_op{}, g0);
        auto* sub1 = p2.create_module("main:hipgraph1");
        auto y0    = sub1->add_parameter("x0", s);
        sub1->add_return({add_chain(*sub1, y0, 4)});
        auto g1 = mm->add_instruction(migraphx::make_op("hip::graph"), {h}, {sub1});
        mm->add_return({g1});
    }

    EXPECT(p1.sort() == p2.sort());
}

// A long run is captured while a trailing run below the threshold is left alone.
TEST_CASE(mixed_long_short)
{
    migraphx::program p1;
    {
        auto* mm  = p1.get_main_module();
        auto x    = mm->add_parameter("x", s);
        auto c    = add_chain(*mm, x, 5);
        auto sync = mm->add_instruction(migraphx::gpu::hip_sync_stream{}, c);
        auto d    = add_chain(*mm, sync, 2);
        mm->add_return({d});
    }
    run_pass(p1);

    migraphx::program p2;
    {
        auto* mm   = p2.get_main_module();
        auto x     = mm->add_parameter("x", s);
        auto* sub0 = p2.create_module("main:hipgraph0");
        auto x0    = sub0->add_parameter("x0", s);
        sub0->add_return({add_chain(*sub0, x0, 5)});
        auto g0   = mm->add_instruction(migraphx::make_op("hip::graph"), {x}, {sub0});
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
        auto x   = mm->add_parameter("x", s);
        auto c   = add_chain(*mm, x, 3);
        mm->add_return({c});
    }
    run_pass(p1, 2);

    migraphx::program p2;
    {
        auto* mm  = p2.get_main_module();
        auto x    = mm->add_parameter("x", s);
        auto* sub = p2.create_module("main:hipgraph0");
        auto x0   = sub->add_parameter("x0", s);
        sub->add_return({add_chain(*sub, x0, 3)});
        auto g = mm->add_instruction(migraphx::make_op("hip::graph"), {x}, {sub});
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
        auto x   = mm->add_parameter("x", s);
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
        auto x    = mm->add_parameter("x", s);
        auto* sub = p2.create_module("main:hipgraph0");
        auto x0   = sub->add_parameter("x0", s);
        auto sc0  = sub->add_instruction(unary_pass_op{}, x0);
        auto sc1  = sub->add_instruction(unary_pass_op{}, sc0);
        auto sc2  = sub->add_instruction(unary_pass_op{}, sc1);
        auto sc3  = sub->add_instruction(unary_pass_op{}, sc2);
        sub->add_return({sc1, sc3});
        auto g  = mm->add_instruction(migraphx::make_op("hip::graph"), {x}, {sub});
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
        auto x     = mm->add_parameter("x", s);
        auto* body = p1.create_module("body");
        auto x0    = body->add_parameter("x0", s);
        body->add_return({add_chain(*body, x0, 4)});
        auto r = mm->add_instruction(mod_pass_op{}, {x}, {body});
        mm->add_return({r});
    }
    run_pass(p1);

    migraphx::program p2;
    {
        auto* mm   = p2.get_main_module();
        auto x     = mm->add_parameter("x", s);
        auto* body = p2.create_module("body");
        auto x0    = body->add_parameter("x0", s);
        body->add_return({add_chain(*body, x0, 4)});
        auto r = mm->add_instruction(mod_pass_op{}, {x}, {body});
        mm->add_return({r});
    }

    EXPECT(p1.sort() == p2.sort());
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
