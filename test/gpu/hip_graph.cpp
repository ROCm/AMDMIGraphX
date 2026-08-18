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
#include <migraphx/algorithm.hpp>
#include <migraphx/program.hpp>
#include <migraphx/module.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/load_save.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/register_target.hpp>
#include <migraphx/verify.hpp>
#include <migraphx/gpu/hip.hpp>
#include <test.hpp>
#include <algorithm>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

// One layer of the kernel chain: a pointwise multiply followed by a reverse.
// Both are jit-compiled into code-object kernels with packed argument buffers
// (so their pointer slots can be patched on rebind), and the reverse keeps the
// pointwise ops from fusing so the chain stays several kernels long while
// mixing the elements between layers.
static migraphx::instruction_ref add_layer(migraphx::module& m,
                                           migraphx::instruction_ref x,
                                           const migraphx::shape& s,
                                           std::int64_t axis)
{
    // Weights below one keep the repeated products bounded.
    std::vector<float> w(s.elements());
    std::generate(w.begin(), w.end(), [n = 0]() mutable { return 0.5f + 0.01f * (n++ % 8); });
    auto scaled = m.add_instruction(migraphx::make_op("mul"), x, m.add_literal({s, w}));
    return m.add_instruction(migraphx::make_op("reverse", {{"axes", {axis}}}), scaled);
}

// A chain of layers over `s`, alternating the reversed axis.
static migraphx::instruction_ref
add_layers(migraphx::module& m, migraphx::instruction_ref x, const migraphx::shape& s, int n)
{
    auto cur = x;
    for(int i = 0; i < n; ++i)
        cur = add_layer(m, cur, s, i % 2);
    return cur;
}

// True when the compiled program contains a hip::graph instruction.
static bool captured_hip_graph(const migraphx::program& p)
{
    auto instructions = migraphx::iterator_for(*p.get_main_module());
    return std::any_of(instructions.begin(), instructions.end(), [](auto ins) {
        return ins->name() == "hip::graph";
    });
}

// Compile copies of `p` for the gpu (with hip graphs enabled) and for the ref
// target, confirming the gpu program captured a hip::graph.
static std::pair<migraphx::program, migraphx::program> compile_gpu_ref(const migraphx::program& p,
                                                                       bool offload_copy)
{
    auto p_gpu = p;
    auto p_ref = p;
    migraphx::compile_options options;
    options.offload_copy                 = offload_copy;
    options.backend_options["hip_graph"] = true;
    p_gpu.compile(migraphx::make_target("gpu"), options);
    p_ref.compile(migraphx::make_target("ref"));
    EXPECT(captured_hip_graph(p_gpu));
    return {std::move(p_gpu), std::move(p_ref)};
}

// Compile `p` for the gpu without offload copy (so it reads/writes
// caller-provided gpu buffers), confirm it captured a hip::graph, then evaluate
// it with input buffers that genuinely move between runs and check every run
// against the ref target. Without correct pointer rebinding the second and third
// runs would replay stale addresses. With `round_trip` the compiled program is
// serialized and reloaded first: the binding must not depend on state that
// serialization drops (the parameter-order field).
static void
check_rebind(const migraphx::program& p, const migraphx::shape& s, bool round_trip = false)
{
    auto [p_gpu, p_ref] = compile_gpu_ref(p, false);
    if(round_trip)
    {
        p_gpu = migraphx::load_buffer(migraphx::save_buffer(p_gpu));
        EXPECT(captured_hip_graph(p_gpu));
    }

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

    auto eval_gpu = [&, &p_gpu = p_gpu](const migraphx::argument& x) {
        migraphx::parameter_map m = base;
        m["x"]                    = x;
        return migraphx::gpu::from_gpu(p_gpu.eval(m).back()).to_vector<float>();
    };
    auto eval_ref = [&, &p_ref = p_ref](unsigned long seed) {
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

// A chain of pointwise/reverse layers lowers to several code-object kernels with
// packed argument buffers, so a moved input is patched directly into the
// executable graph.
TEST_CASE(rebind_kernel_only)
{
    migraphx::shape s{migraphx::shape::float_type, {8, 8}};
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto x   = mm->add_parameter("x", s);
    mm->add_return({add_layers(*mm, x, s, 3)});
    check_rebind(p, s);
}

// Convolutions exercise a different op and library path (MLIR, or MIOpen when
// MLIR is disabled) than the pointwise/reverse chain used by the other tests.
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
    mm->add_return({add_layers(*mm, x, s, 3)});

    auto [p_gpu, p_ref] = compile_gpu_ref(p, true);

    auto run_gpu = [&, &p_gpu = p_gpu](unsigned long seed) {
        return p_gpu.eval({{"x", migraphx::generate_argument(s, seed)}}).back().to_vector<float>();
    };
    auto run_ref = [&, &p_ref = p_ref](unsigned long seed) {
        return p_ref.eval({{"x", migraphx::generate_argument(s, seed)}}).back().to_vector<float>();
    };
    EXPECT(migraphx::verify::verify_rms_range(run_gpu(1), run_ref(1)));
    EXPECT(migraphx::verify::verify_rms_range(run_gpu(2), run_ref(2)));
}

// The submodule's parameter binding must survive serialization: the binding is
// by name order, while the parameter-order field that get_parameter_names()
// reflects is dropped by save/load. A reloaded program that bound positionally would pair
// arguments with the wrong parameters and rebind the wrong kernel slots.
TEST_CASE(rebind_save_load)
{
    migraphx::shape s{migraphx::shape::float_type, {8, 8}};
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto x   = mm->add_parameter("x", s);
    mm->add_return({add_layers(*mm, x, s, 3)});
    check_rebind(p, s, true);
}

// The kernels consume a slice of x rather than x itself, so the captured
// pointer slots carry a nonzero within-leaf offset that must be re-applied on
// top of the moved buffer's base address when x rebinds.
TEST_CASE(rebind_sliced_input)
{
    migraphx::shape xs{migraphx::shape::float_type, {16, 8}};
    migraphx::shape s{migraphx::shape::float_type, {8, 8}};
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto x   = mm->add_parameter("x", xs);
    auto sl  = mm->add_instruction(
        migraphx::make_op("slice", {{"axes", {0}}, {"starts", {8}}, {"ends", {16}}}), x);
    mm->add_return({add_layers(*mm, sl, s, 3)});
    check_rebind(p, xs);
}

// The parameter names of the output buffers replace_allocate creates when
// offload copy is disabled.
static std::vector<std::string> output_param_names(const migraphx::program& p)
{
    auto param_shapes = p.get_parameter_shapes();
    std::vector<std::string> names;
    migraphx::transform_if(
        param_shapes.begin(),
        param_shapes.end(),
        std::back_inserter(names),
        [](const auto& param) { return migraphx::contains(param.first, "#output_"); },
        [](const auto& param) { return param.first; });
    std::sort(names.begin(), names.end());
    return names;
}

// Both the input and the output buffer move between runs. The output parameter
// is inserted right before the kernel that writes it, which splits the run
// there: the graph's rebinding and the uncaptured trailing kernel must both
// follow the moved buffers for the returned values to stay correct.
TEST_CASE(rebind_output_buffer)
{
    migraphx::shape s{migraphx::shape::float_type, {8, 8}};
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto x   = mm->add_parameter("x", s);
    mm->add_return({add_layers(*mm, x, s, 3)});

    auto [p_gpu, p_ref] = compile_gpu_ref(p, false);

    auto out_names = output_param_names(p_gpu);
    EXPECT(out_names.size() == 1);
    auto gpu_shapes = p_gpu.get_parameter_shapes();
    migraphx::parameter_map base;
    for(auto&& [name, ps] : gpu_shapes)
    {
        if(name != "x" and name != out_names.front())
            base[name] = migraphx::gpu::allocate_gpu(ps);
    }

    auto x1   = migraphx::gpu::to_gpu(migraphx::generate_argument(s, 1));
    auto x2   = migraphx::gpu::to_gpu(migraphx::generate_argument(s, 2));
    auto out1 = migraphx::gpu::allocate_gpu(gpu_shapes.at(out_names.front()));
    auto out2 = migraphx::gpu::allocate_gpu(gpu_shapes.at(out_names.front()));

    auto eval_gpu = [&, &p_gpu = p_gpu](const migraphx::argument& x_arg,
                                        const migraphx::argument& out_arg) {
        migraphx::parameter_map m = base;
        m["x"]                    = x_arg;
        m[out_names.front()]      = out_arg;
        return migraphx::gpu::from_gpu(p_gpu.eval(m).back()).to_vector<float>();
    };
    auto eval_ref = [&, &p_ref = p_ref](unsigned long seed) {
        return p_ref.eval({{"x", migraphx::generate_argument(s, seed)}}).back().to_vector<float>();
    };

    auto ref1 = eval_ref(1);
    auto ref2 = eval_ref(2);
    // A stale cached result would still view out1 on the second run and so
    // return run-1 values.
    EXPECT(migraphx::verify::verify_rms_range(eval_gpu(x1, out1), ref1));
    EXPECT(migraphx::verify::verify_rms_range(eval_gpu(x2, out2), ref2));
    EXPECT(migraphx::verify::verify_rms_range(eval_gpu(x1, out1), ref1));
}

// Two returned values captured as one graph: the first output also feeds the
// rest of the chain, and with offload copy both outputs are written into
// allocations kept in the parent (the aliases path), so the op returns a tuple
// unpacked with get_tuple_elem. Two evals with different data check the tuple
// result across capture and replay. (Without offload copy this program is
// deliberately left uncaptured: the interleaved output parameters split the
// runs and mix parameter-backed with allocation-backed outputs.)
TEST_CASE(multi_output_tuple)
{
    migraphx::shape s{migraphx::shape::float_type, {8, 8}};
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto x   = mm->add_parameter("x", s);
    auto c1  = add_layers(*mm, x, s, 1);
    auto c3  = add_layers(*mm, c1, s, 2);
    mm->add_return({c1, c3});

    auto [p_gpu, p_ref] = compile_gpu_ref(p, true);

    auto run2 = [&](migraphx::program& prog, unsigned long seed) {
        auto results = prog.eval({{"x", migraphx::generate_argument(s, seed)}});
        EXPECT(results.size() == 2);
        return std::make_pair(results[0].to_vector<float>(), results[1].to_vector<float>());
    };

    auto g1 = run2(p_gpu, 1);
    auto r1 = run2(p_ref, 1);
    auto g2 = run2(p_gpu, 2);
    auto r2 = run2(p_ref, 2);
    EXPECT(migraphx::verify::verify_rms_range(g1.first, r1.first));
    EXPECT(migraphx::verify::verify_rms_range(g1.second, r1.second));
    EXPECT(migraphx::verify::verify_rms_range(g2.first, r2.first));
    EXPECT(migraphx::verify::verify_rms_range(g2.second, r2.second));
}

// The first run binds one buffer to both x and y; pointer slots cannot be
// attributed unambiguously to overlapping inputs, so when the buffers become
// distinct the op must re-record instead of patching y's kernel slots with x's
// new address. This also pins the re-record path itself, which patchable
// (all-code-object) programs never take.
TEST_CASE(rebind_aliased_inputs)
{
    migraphx::shape s{migraphx::shape::float_type, {8, 8}};
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto x   = mm->add_parameter("x", s);
    auto y   = mm->add_parameter("y", s);
    auto lx  = add_layers(*mm, x, s, 2);
    auto sum = mm->add_instruction(migraphx::make_op("add"), lx, y);
    mm->add_return({add_layers(*mm, sum, s, 2)});

    auto [p_gpu, p_ref] = compile_gpu_ref(p, false);

    auto gpu_shapes = p_gpu.get_parameter_shapes();
    migraphx::parameter_map base;
    for(auto&& [name, ps] : gpu_shapes)
    {
        if(name != "x" and name != "y")
            base[name] = migraphx::gpu::allocate_gpu(ps);
    }

    auto v1 = migraphx::gpu::to_gpu(migraphx::generate_argument(s, 1));
    auto v2 = migraphx::gpu::to_gpu(migraphx::generate_argument(s, 2));
    auto v3 = migraphx::gpu::to_gpu(migraphx::generate_argument(s, 3));

    auto eval_gpu = [&, &p_gpu = p_gpu](const migraphx::argument& x_arg,
                                        const migraphx::argument& y_arg) {
        migraphx::parameter_map m = base;
        m["x"]                    = x_arg;
        m["y"]                    = y_arg;
        return migraphx::gpu::from_gpu(p_gpu.eval(m).back()).to_vector<float>();
    };
    auto eval_ref = [&, &p_ref = p_ref](unsigned long xseed, unsigned long yseed) {
        return p_ref
            .eval({{"x", migraphx::generate_argument(s, xseed)},
                   {"y", migraphx::generate_argument(s, yseed)}})
            .back()
            .to_vector<float>();
    };

    EXPECT(migraphx::verify::verify_rms_range(eval_gpu(v1, v1), eval_ref(1, 1)));
    EXPECT(migraphx::verify::verify_rms_range(eval_gpu(v2, v3), eval_ref(2, 3)));
    EXPECT(migraphx::verify::verify_rms_range(eval_gpu(v1, v1), eval_ref(1, 1)));
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
