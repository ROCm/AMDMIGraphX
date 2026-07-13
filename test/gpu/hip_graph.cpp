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
#include <migraphx/program.hpp>
#include <migraphx/module.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/register_target.hpp>
#include <migraphx/verify.hpp>
#include <migraphx/gpu/hip.hpp>
#include <test.hpp>
#include <vector>

static migraphx::instruction_ref
add_dot(migraphx::module& m, migraphx::instruction_ref x, const migraphx::shape& s)
{
    std::vector<float> w(s.elements(), 0.0f);
    // A mostly-diagonal weight keeps the repeated products bounded while still
    // mixing the elements.
    for(std::size_t r = 0; r < s.lens()[0]; ++r)
        for(std::size_t c = 0; c < s.lens()[1]; ++c)
            w[r * s.lens()[1] + c] = (r == c) ? 0.5f : 0.05f;
    return m.add_instruction(migraphx::make_op("dot"), x, m.add_literal({s, w}));
}

// Compile `p` for the gpu without offload copy (so it reads/writes
// caller-provided gpu buffers), confirm it captured a hip::graph, then evaluate
// it with input buffers that genuinely move between runs and check every run
// against the ref target. Without correct pointer rebinding the second and third
// runs would replay stale addresses.
static void check_rebind(const migraphx::program& p, const migraphx::shape& s)
{
    auto p_gpu = p;
    auto p_ref = p;

    migraphx::compile_options options;
    options.offload_copy                 = false;
    options.backend_options["hip_graph"] = true;
    p_gpu.compile(migraphx::make_target("gpu"), options);
    p_ref.compile(migraphx::make_target("ref"));

    bool captured = false;
    for(auto ins : migraphx::iterator_for(*p_gpu.get_main_module()))
        captured |= (ins->name() == "hip::graph");
    EXPECT(captured);

    auto gpu_shapes = p_gpu.get_parameter_shapes();
    // Keep stable buffers for every parameter except x so only x's address
    // varies, and hold two distinct x buffers alive at once so they really get
    // different addresses (reusing one freed buffer would alias the same address
    // and never exercise the rebinding path).
    migraphx::parameter_map base;
    for(auto&& [name, ps] : gpu_shapes)
        if(name != "x")
            base[name] = migraphx::gpu::allocate_gpu(ps);
    auto x1 = migraphx::gpu::to_gpu(migraphx::generate_argument(s, 1));
    auto x2 = migraphx::gpu::to_gpu(migraphx::generate_argument(s, 2));
    EXPECT(x1.data() != x2.data());

    auto eval_gpu = [&](const migraphx::argument& x) {
        migraphx::parameter_map m = base;
        m["x"]                    = x;
        return migraphx::gpu::from_gpu(p_gpu.eval(m).back()).to_vector<float>();
    };
    auto eval_ref = [&](unsigned long seed) {
        return p_ref.eval({{"x", migraphx::generate_argument(s, seed)}}).back().to_vector<float>();
    };

    auto ref1 = eval_ref(1);
    auto ref2 = eval_ref(2);

    auto first  = eval_gpu(x1); // captures the graph with x1's address
    auto change = eval_gpu(x2); // x2 has a different address -> graph is re-bound
    auto rebind = eval_gpu(x1); // back to x1's address       -> graph is re-bound
    auto stable = eval_gpu(x1); // same address again         -> replay, no re-bind

    EXPECT(migraphx::verify::verify_rms_range(first, ref1));
    EXPECT(migraphx::verify::verify_rms_range(change, ref2));
    EXPECT(migraphx::verify::verify_rms_range(rebind, ref1));
    EXPECT(migraphx::verify::verify_rms_range(stable, ref1));
}

// A chain of matmuls lowers to several code-object kernels with packed argument
// buffers, so a moved input is patched directly into the executable graph.
TEST_CASE(rebind_kernel_only)
{
    migraphx::shape s{migraphx::shape::float_type, {8, 8}};
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto x   = mm->add_parameter("x", s);
    auto cur = x;
    for(int i = 0; i < 6; ++i)
        cur = add_dot(*mm, cur, s);
    mm->add_return({cur});
    check_rebind(p, s);
}

// Convolutions exercise a different op and library path (MLIR, or MIOpen when
// MLIR is disabled) than the matmul chain.
TEST_CASE(rebind_convolution)
{
    migraphx::shape s{migraphx::shape::float_type, {1, 4, 8, 8}};
    migraphx::shape ws{migraphx::shape::float_type, {4, 4, 3, 3}};
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto x   = mm->add_parameter("x", s);
    auto cur = x;
    for(int i = 0; i < 4; ++i)
    {
        std::vector<float> w(ws.elements(), 0.05f);
        cur = mm->add_instruction(
            migraphx::make_op("convolution", {{"padding", {1, 1}}, {"stride", {1, 1}}}),
            cur,
            mm->add_literal({ws, w}));
    }
    mm->add_return({cur});
    check_rebind(p, s);
}

// With offload copy the kernels consume fixed internal gpu allocations (the host
// parameter is bridged by an excluded copy), so the captured graph has no movable
// inputs and is replayed without ever inspecting its nodes.
TEST_CASE(offload_copy_no_rebind)
{
    migraphx::shape s{migraphx::shape::float_type, {8, 8}};
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto x   = mm->add_parameter("x", s);
    auto cur = x;
    for(int i = 0; i < 6; ++i)
        cur = add_dot(*mm, cur, s);
    mm->add_return({cur});

    auto p_gpu = p;
    auto p_ref = p;
    migraphx::compile_options options;
    options.offload_copy                 = true;
    options.backend_options["hip_graph"] = true;
    p_gpu.compile(migraphx::make_target("gpu"), options);
    p_ref.compile(migraphx::make_target("ref"));

    bool captured = false;
    for(auto ins : migraphx::iterator_for(*p_gpu.get_main_module()))
        captured |= (ins->name() == "hip::graph");
    EXPECT(captured);

    auto run_gpu = [&](unsigned long seed) {
        return p_gpu.eval({{"x", migraphx::generate_argument(s, seed)}}).back().to_vector<float>();
    };
    auto run_ref = [&](unsigned long seed) {
        return p_ref.eval({{"x", migraphx::generate_argument(s, seed)}}).back().to_vector<float>();
    };
    EXPECT(migraphx::verify::verify_rms_range(run_gpu(1), run_ref(1)));
    EXPECT(migraphx::verify::verify_rms_range(run_gpu(2), run_ref(2)));
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
