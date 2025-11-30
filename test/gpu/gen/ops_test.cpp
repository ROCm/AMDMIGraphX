/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2025 Advanced Micro Devices, Inc. All rights reserved.
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
#include <test.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/module.hpp>
#include <migraphx/program.hpp>
#include <migraphx/shape.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/gpu/compile_hip_code_object.hpp>
#include <migraphx/gpu/compile_hip.hpp>
#include <migraphx/gpu/kernel.hpp>
#include <migraphx/gpu/device_name.hpp>
#include <migraphx/gpu/gen/codegen.hpp>
#include <migraphx/gpu/hip.hpp>
#include <hip/hip_runtime.h>

// Helper to compile and run a gen kernel that uses MIGRAPHX_CHECK
static void run_gen_kernel(const std::string& src,
                           const std::string& kernel_name,
                           std::size_t global = 1,
                           std::size_t local  = 1)
{
    migraphx::gpu::context ctx;
    migraphx::gpu::hip_compile_options options;
    options.global      = global;
    options.local       = local;
    options.kernel_name = kernel_name;
    options.emplace_param("-Wno-float-equal");

    auto binary = migraphx::gpu::compile_hip_raw(ctx, src, options);
    migraphx::gpu::kernel k{binary, kernel_name};
    k.launch(nullptr, global, local)();
    CHECK(hipDeviceSynchronize() == hipSuccess);
}

TEST_CASE(test_tile_region_op)
{
    auto op = migraphx::make_op(
        "gpu::gen::tile_region",
        {{"tile_dims", std::vector<std::size_t>{32, 64}}, {"axis", std::size_t{1}}});
    EXPECT(op.name() == "gpu::gen::tile_region");

    // Test compute_shape - tile_region takes (tensor, workgroup_id)
    auto tensor_shape =
        migraphx::shape{migraphx::shape::float_type, {128, 256, 512}, {131072, 512, 1}};
    auto wg_id_shape = migraphx::shape{migraphx::shape::uint64_type};
    auto s           = op.compute_shape({tensor_shape, wg_id_shape});
    EXPECT(s.type() == migraphx::shape::float_type);
    // Output shape: dims before axis=1 become 1, tile_dims replace remaining dims
    EXPECT(s.lens() == std::vector<std::size_t>{1, 32, 64});
}

TEST_CASE(test_lane_id_op)
{
    auto op = migraphx::make_op("gpu::gen::lane_id");
    EXPECT(op.name() == "gpu::gen::lane_id");

    auto attrs = op.attributes();
    EXPECT(attrs.contains("point_op"));
    EXPECT(attrs["point_op"].to<std::string>() == "idx.local_wave()");

    auto s = op.compute_shape({});
    EXPECT(s.type() == migraphx::shape::uint64_type);
}

TEST_CASE(test_local_id_op)
{
    auto op = migraphx::make_op("gpu::gen::local_id");
    EXPECT(op.name() == "gpu::gen::local_id");

    auto attrs = op.attributes();
    EXPECT(attrs.contains("point_op"));
    EXPECT(attrs["point_op"].to<std::string>() == "idx.local");

    auto s = op.compute_shape({});
    EXPECT(s.type() == migraphx::shape::uint64_type);
}

TEST_CASE(test_global_id_op)
{
    auto op = migraphx::make_op("gpu::gen::global_id");
    EXPECT(op.name() == "gpu::gen::global_id");

    auto attrs = op.attributes();
    EXPECT(attrs.contains("point_op"));
    EXPECT(attrs["point_op"].to<std::string>() == "idx.global");

    auto s = op.compute_shape({});
    EXPECT(s.type() == migraphx::shape::uint64_type);
}

TEST_CASE(test_workgroup_id_op)
{
    auto op = migraphx::make_op("gpu::gen::workgroup_id");
    EXPECT(op.name() == "gpu::gen::workgroup_id");

    auto attrs = op.attributes();
    EXPECT(attrs.contains("point_op"));
    EXPECT(attrs["point_op"].to<std::string>() == "idx.group");

    auto s = op.compute_shape({});
    EXPECT(s.type() == migraphx::shape::uint64_type);
}

TEST_CASE(test_workgroup_size_op)
{
    auto op = migraphx::make_op("gpu::gen::workgroup_size");
    EXPECT(op.name() == "gpu::gen::workgroup_size");

    auto attrs = op.attributes();
    EXPECT(attrs.contains("point_op"));
    EXPECT(attrs["point_op"].to<std::string>() == "idx.nlocal()");

    auto s = op.compute_shape({});
    EXPECT(s.type() == migraphx::shape::uint64_type);
}

TEST_CASE(test_barrier_op)
{
    auto op = migraphx::make_op("gpu::gen::barrier");
    EXPECT(op.name() == "gpu::gen::barrier");

    auto attrs = op.attributes();
    EXPECT(attrs.contains("point_op"));
    // (void) prefix indicates this returns void and should not create a variable
    EXPECT(attrs["point_op"].to<std::string>() == "(void)__syncthreads()");
}

TEST_CASE(test_check_op)
{
    auto op = migraphx::make_op("gpu::gen::check");
    EXPECT(op.name() == "gpu::gen::check");

    auto attrs = op.attributes();
    EXPECT(attrs.contains("point_op"));
    // (void) prefix indicates this returns void and should not create a variable
    EXPECT(attrs["point_op"].to<std::string>() == "(void)MIGRAPHX_CHECK(${0})");

    // check returns an empty shape (no output, just side effect)
    auto s = op.compute_shape({migraphx::shape{migraphx::shape::bool_type}});
    EXPECT(s.lens().empty());
}

TEST_CASE(test_gen_pointwise_op)
{
    auto op = migraphx::make_op("gpu::gen::pointwise");
    EXPECT(op.name() == "gpu::gen::pointwise");
}

TEST_CASE(test_vector_load_op)
{
    auto op = migraphx::make_op("gpu::gen::vector_load", {{"size", std::size_t{4}}});
    EXPECT(op.name() == "gpu::gen::vector_load");

    // compute_shape: takes (tensor, index), returns vector of `size` elements
    auto tensor_shape = migraphx::shape{migraphx::shape::float_type, {64, 64}};
    auto index_shape  = migraphx::shape{migraphx::shape::uint64_type};
    auto s            = op.compute_shape({tensor_shape, index_shape});
    EXPECT(s.type() == migraphx::shape::float_type);
    EXPECT(s.lens() == std::vector<std::size_t>{4});
}

TEST_CASE(test_vector_store_op)
{
    auto op = migraphx::make_op("gpu::gen::vector_store", {{"size", std::size_t{4}}});
    EXPECT(op.name() == "gpu::gen::vector_store");

    // compute_shape: takes (tensor, index, data), returns empty (side effect)
    auto tensor_shape = migraphx::shape{migraphx::shape::float_type, {64, 64}};
    auto index_shape  = migraphx::shape{migraphx::shape::uint64_type};
    auto data_shape   = migraphx::shape{migraphx::shape::float_type, {4}};
    auto s            = op.compute_shape({tensor_shape, index_shape, data_shape});
    EXPECT(s.lens().empty());
}

TEST_CASE(test_copy_op)
{
    // Test copy with default schedule
    auto op = migraphx::make_op("gpu::gen::copy");
    EXPECT(op.name() == "gpu::gen::copy");

    // compute_shape: takes (src, dst), returns dst shape
    auto src_shape = migraphx::shape{migraphx::shape::float_type, {64, 64}};
    auto dst_shape = migraphx::shape{migraphx::shape::float_type, {64, 64}};
    auto s         = op.compute_shape({src_shape, dst_shape});
    EXPECT(s == dst_shape);
}

TEST_CASE(test_copy_op_with_schedule)
{
    // Test copy with per_block schedule
    auto op = migraphx::make_op("gpu::gen::copy", {{"schedule", std::string("per_block")}});
    EXPECT(op.name() == "gpu::gen::copy");

    auto v = op.to_value();
    EXPECT(v.at("schedule").to<std::string>() == "per_block");
}

// Runtime tests - compile and run on GPU using MIGRAPHX_CHECK

TEST_CASE(test_global_id_runtime)
{
    // Test that global_id == 0 when launching 1 thread
    std::string src = R"(
#include <migraphx/kernels/index.hpp>
#include <migraphx/kernels/debug.hpp>

using namespace migraphx;

extern "C" __global__ void test_global_id() {
    auto idx = make_index();
    MIGRAPHX_CHECK(idx.global == 0);
}
)";
    run_gen_kernel(src, "test_global_id");
}

TEST_CASE(test_local_id_runtime)
{
    // Test that local_id == 0 when launching 1 thread
    std::string src = R"(
#include <migraphx/kernels/index.hpp>
#include <migraphx/kernels/debug.hpp>

using namespace migraphx;

extern "C" __global__ void test_local_id() {
    auto idx = make_index();
    MIGRAPHX_CHECK(idx.local == 0);
}
)";
    run_gen_kernel(src, "test_local_id");
}

TEST_CASE(test_workgroup_id_runtime)
{
    // Test that workgroup_id == 0 when launching 1 workgroup
    std::string src = R"(
#include <migraphx/kernels/index.hpp>
#include <migraphx/kernels/debug.hpp>

using namespace migraphx;

extern "C" __global__ void test_workgroup_id() {
    auto idx = make_index();
    MIGRAPHX_CHECK(idx.group == 0);
}
)";
    run_gen_kernel(src, "test_workgroup_id");
}

TEST_CASE(test_workgroup_size_runtime)
{
    // Test that nlocal() == 64 when launching with local=64
    std::string src = R"(
#include <migraphx/kernels/index.hpp>
#include <migraphx/kernels/debug.hpp>

using namespace migraphx;

extern "C" __global__ void test_workgroup_size() {
    auto idx = make_index();
    MIGRAPHX_CHECK(idx.nlocal() == 64);
}
)";
    run_gen_kernel(src, "test_workgroup_size", 64, 64);
}

TEST_CASE(test_lane_id_runtime)
{
    // Test that lane_id (local_wave) is in valid range [0, wavefront_size)
    std::string src = R"(
#include <migraphx/kernels/index.hpp>
#include <migraphx/kernels/debug.hpp>

using namespace migraphx;

extern "C" __global__ void test_lane_id() {
    auto idx = make_index();
    // local_wave() returns lane within wavefront
    MIGRAPHX_CHECK(idx.local_wave() < idx.nlocal_wave());
}
)";
    run_gen_kernel(src, "test_lane_id", 64, 64);
}

TEST_CASE(test_multiple_threads_global_id)
{
    // Test that each thread has unique global_id < nglobal
    std::string src = R"(
#include <migraphx/kernels/index.hpp>
#include <migraphx/kernels/debug.hpp>

using namespace migraphx;

extern "C" __global__ void test_multi_global() {
    auto idx = make_index();
    MIGRAPHX_CHECK(idx.global < idx.nglobal());
}
)";
    run_gen_kernel(src, "test_multi_global", 256, 64);
}

TEST_CASE(test_compile_gen_simple)
{
    // Create a simple gen IR program that verifies global_id is valid
    migraphx::program p;
    auto* mm = p.get_main_module();

    // Add a parameter (dummy tensor for args.hpp generation)
    auto x = mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {64}});

    // Add global_id
    auto gid = mm->add_instruction(migraphx::make_op("gpu::gen::global_id"));

    // We need to return something
    mm->add_return({x});

    // Compile and run
    migraphx::gpu::context ctx;
    auto code_op = migraphx::gpu::gen::compile_gen(ctx, p, "test_compile_gen_simple");

    // Verify we got a valid code object operation
    EXPECT(code_op.name() == "gpu::code_object");

    (void)gid;
}

TEST_CASE(test_compile_gen_vector_load_store)
{
    // Create a program that loads and stores a vector
    migraphx::program p;
    auto* mm = p.get_main_module();

    // Parameters: input tensor, output tensor
    auto x = mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {64}});
    auto y = mm->add_parameter("y", migraphx::shape{migraphx::shape::float_type, {64}});

    // Get global ID
    auto gid = mm->add_instruction(migraphx::make_op("gpu::gen::global_id"));

    // Load a vector from x
    auto load = mm->add_instruction(
        migraphx::make_op("gpu::gen::vector_load", {{"size", std::size_t{4}}}), x, gid);

    // Store to y
    mm->add_instruction(
        migraphx::make_op("gpu::gen::vector_store", {{"size", std::size_t{4}}}), y, gid, load);

    mm->add_return({y});

    // Compile
    migraphx::gpu::context ctx;
    auto code_op = migraphx::gpu::gen::compile_gen(ctx, p, "test_vector_kernel");

    // Verify we got a valid code object operation
    EXPECT(code_op.name() == "gpu::code_object");
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
