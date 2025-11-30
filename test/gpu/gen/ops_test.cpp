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
#include <migraphx/shape.hpp>

TEST_CASE(test_tile_region_op)
{
    auto op = migraphx::make_op(
        "gpu::gen::tile_region",
        {{"tile_dims", std::vector<std::size_t>{32, 64}}, {"axis", std::size_t{1}}});
    EXPECT(op.name() == "gpu::gen::tile_region");

    auto attrs = op.attributes();
    EXPECT(attrs.contains("point_op"));

    // Test compute_shape
    auto s = op.compute_shape({migraphx::shape{migraphx::shape::float_type, {128, 256}}});
    EXPECT(s.type() == migraphx::shape::float_type);
    EXPECT(s.lens() == std::vector<std::size_t>{128, 256});
}

TEST_CASE(test_lane_id_op)
{
    auto op = migraphx::make_op("gpu::gen::lane_id");
    EXPECT(op.name() == "gpu::gen::lane_id");

    auto attrs = op.attributes();
    EXPECT(attrs.contains("point_op"));
    EXPECT(attrs["point_op"].to<std::string>() == "__lane_id()");

    auto s = op.compute_shape({});
    EXPECT(s.type() == migraphx::shape::uint32_type);
}

TEST_CASE(test_local_id_op)
{
    auto op = migraphx::make_op("gpu::gen::local_id", {{"dim", std::size_t{0}}});
    EXPECT(op.name() == "gpu::gen::local_id");

    auto attrs = op.attributes();
    EXPECT(attrs.contains("point_op"));
    EXPECT(attrs["point_op"].to<std::string>() == "threadIdx.x");

    auto s = op.compute_shape({});
    EXPECT(s.type() == migraphx::shape::uint32_type);
}

TEST_CASE(test_local_id_dim_y)
{
    auto op    = migraphx::make_op("gpu::gen::local_id", {{"dim", std::size_t{1}}});
    auto attrs = op.attributes();
    EXPECT(attrs["point_op"].to<std::string>() == "threadIdx.y");
}

TEST_CASE(test_local_id_dim_z)
{
    auto op    = migraphx::make_op("gpu::gen::local_id", {{"dim", std::size_t{2}}});
    auto attrs = op.attributes();
    EXPECT(attrs["point_op"].to<std::string>() == "threadIdx.z");
}

TEST_CASE(test_global_id_op)
{
    auto op = migraphx::make_op("gpu::gen::global_id", {{"dim", std::size_t{0}}});
    EXPECT(op.name() == "gpu::gen::global_id");

    auto attrs = op.attributes();
    EXPECT(attrs.contains("point_op"));
    EXPECT(attrs["point_op"].to<std::string>() == "idx.global");

    auto s = op.compute_shape({});
    EXPECT(s.type() == migraphx::shape::uint64_type);
}

TEST_CASE(test_workgroup_id_op)
{
    auto op = migraphx::make_op("gpu::gen::workgroup_id", {{"dim", std::size_t{0}}});
    EXPECT(op.name() == "gpu::gen::workgroup_id");

    auto attrs = op.attributes();
    EXPECT(attrs.contains("point_op"));
    EXPECT(attrs["point_op"].to<std::string>() == "blockIdx.x");

    auto s = op.compute_shape({});
    EXPECT(s.type() == migraphx::shape::uint32_type);
}

TEST_CASE(test_workgroup_size_op)
{
    auto op = migraphx::make_op("gpu::gen::workgroup_size", {{"dim", std::size_t{1}}});
    EXPECT(op.name() == "gpu::gen::workgroup_size");

    auto attrs = op.attributes();
    EXPECT(attrs.contains("point_op"));
    EXPECT(attrs["point_op"].to<std::string>() == "blockDim.y");

    auto s = op.compute_shape({});
    EXPECT(s.type() == migraphx::shape::uint32_type);
}

TEST_CASE(test_barrier_op)
{
    auto op = migraphx::make_op("gpu::gen::barrier");
    EXPECT(op.name() == "gpu::gen::barrier");

    auto attrs = op.attributes();
    EXPECT(attrs.contains("point_op"));
    EXPECT(attrs["point_op"].to<std::string>() == "__syncthreads()");
}

TEST_CASE(test_gen_pointwise_op)
{
    auto op = migraphx::make_op("gpu::gen::pointwise");
    EXPECT(op.name() == "gpu::gen::pointwise");
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
