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
// code_object_op's cluster_size/warp_specialize accessors, modeled on lds_carveout.cpp. These do
// not need a GPU.
#include <migraphx/gpu/code_object_op.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/serialize.hpp>
#include <migraphx/stringutils.hpp>
#include <sstream>
#include <test.hpp>

namespace {
std::string print(const migraphx::gpu::code_object_op& op)
{
    std::stringstream ss;
    ss << op;
    return ss.str();
}
} // namespace

TEST_CASE(cluster_size_default)
{
    migraphx::gpu::code_object_op op;
    EXPECT(op.get_cluster_size() == 1);
    EXPECT(not migraphx::contains(print(op), "cluster_size"));
}

// 1 (no cluster) is the default and records nothing, matching lds_carveout's -1
TEST_CASE(cluster_size_one_records_nothing)
{
    migraphx::gpu::code_object_op op;
    op.set_cluster_size(1);
    EXPECT(op.get_cluster_size() == 1);
}

TEST_CASE(cluster_size_set)
{
    migraphx::gpu::code_object_op op;
    op.set_cluster_size(4);
    EXPECT(op.get_cluster_size() == 4);
    EXPECT(migraphx::contains(print(op), "cluster_size"));
}

TEST_CASE(cluster_size_round_trip)
{
    migraphx::gpu::code_object_op op;
    op.symbol_name = "kernel";
    op.set_cluster_size(8);
    auto loaded = migraphx::from_value<migraphx::gpu::code_object_op>(migraphx::to_value(op));
    EXPECT(loaded.get_cluster_size() == 8);
    EXPECT(loaded.symbol_name == "kernel");

    migraphx::gpu::code_object_op unset;
    unset.symbol_name = "kernel";
    auto loaded_unset =
        migraphx::from_value<migraphx::gpu::code_object_op>(migraphx::to_value(unset));
    EXPECT(loaded_unset.get_cluster_size() == 1);
}

TEST_CASE(warp_specialize_default)
{
    migraphx::gpu::code_object_op op;
    EXPECT(not op.get_warp_specialize());
    EXPECT(not migraphx::contains(print(op), "warp_specialize"));
}

// false is the default and records nothing
TEST_CASE(warp_specialize_false_records_nothing)
{
    migraphx::gpu::code_object_op op;
    op.set_warp_specialize(false);
    EXPECT(not op.get_warp_specialize());
}

TEST_CASE(warp_specialize_set)
{
    migraphx::gpu::code_object_op op;
    op.set_warp_specialize(true);
    EXPECT(op.get_warp_specialize());
    EXPECT(migraphx::contains(print(op), "warp_specialize"));
}

TEST_CASE(warp_specialize_round_trip)
{
    migraphx::gpu::code_object_op op;
    op.symbol_name = "kernel";
    op.set_warp_specialize(true);
    auto loaded = migraphx::from_value<migraphx::gpu::code_object_op>(migraphx::to_value(op));
    EXPECT(loaded.get_warp_specialize());

    migraphx::gpu::code_object_op unset;
    unset.symbol_name = "kernel";
    auto loaded_unset =
        migraphx::from_value<migraphx::gpu::code_object_op>(migraphx::to_value(unset));
    EXPECT(not loaded_unset.get_warp_specialize());
}

// Both annotations coexist and survive a round trip together, e.g. a clustered warp-specialized
// kernel with a carveout too.
TEST_CASE(cluster_size_and_warp_specialize_together)
{
    migraphx::gpu::code_object_op op;
    op.set_cluster_size(2);
    op.set_warp_specialize(true);
    op.set_lds_carveout(100);
    auto loaded = migraphx::from_value<migraphx::gpu::code_object_op>(migraphx::to_value(op));
    EXPECT(loaded.get_cluster_size() == 2);
    EXPECT(loaded.get_warp_specialize());
    EXPECT(loaded.get_lds_carveout() == 100);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
