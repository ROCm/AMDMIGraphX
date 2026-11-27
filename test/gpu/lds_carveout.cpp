/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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
#include <migraphx/compile_options.hpp>
#include <migraphx/gpu/code_object_op.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/module.hpp>
#include <migraphx/program.hpp>
#include <migraphx/register_target.hpp>
#include <migraphx/serialize.hpp>
#include <migraphx/stringutils.hpp>
#include <algorithm>
#include <sstream>
#include <string>
#include <vector>
#include <test.hpp>

namespace {

std::string print(const migraphx::gpu::code_object_op& op)
{
    std::stringstream ss;
    ss << op;
    return ss.str();
}

} // namespace

// The carveout is kept on the op, and nothing is recorded when it is unset
TEST_CASE(code_object_op_default)
{
    migraphx::gpu::code_object_op op;
    EXPECT(op.get_lds_carveout() == -1);
    EXPECT(not migraphx::contains(print(op), "lds_carveout"));
}

TEST_CASE(code_object_op_unset_value_records_nothing)
{
    migraphx::gpu::code_object_op op;
    op.set_lds_carveout(-1);
    EXPECT(op.get_lds_carveout() == -1);
}

TEST_CASE(code_object_op_set)
{
    migraphx::gpu::code_object_op op;
    op.set_lds_carveout(100);
    EXPECT(op.get_lds_carveout() == 100);
    EXPECT(migraphx::contains(print(op), "lds_carveout"));
}

// 0 is a real request (all of the storage as L1), not the same as unset
TEST_CASE(code_object_op_zero_is_not_unset)
{
    migraphx::gpu::code_object_op op;
    op.set_lds_carveout(0);
    EXPECT(op.get_lds_carveout() == 0);
}

// The value is what is saved in and loaded from a program, and a program without it loads as
// unset
TEST_CASE(code_object_op_round_trip)
{
    migraphx::gpu::code_object_op op;
    op.symbol_name = "kernel";
    op.set_lds_carveout(50);
    auto loaded = migraphx::from_value<migraphx::gpu::code_object_op>(migraphx::to_value(op));
    EXPECT(loaded.get_lds_carveout() == 50);
    EXPECT(loaded.symbol_name == "kernel");

    migraphx::gpu::code_object_op unset;
    unset.symbol_name = "kernel";
    auto loaded_unset =
        migraphx::from_value<migraphx::gpu::code_object_op>(migraphx::to_value(unset));
    EXPECT(loaded_unset.get_lds_carveout() == -1);
}

namespace {

migraphx::program compile_conv(const migraphx::value& carveout)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto x   = mm->add_parameter("x", {migraphx::shape::float_type, {1, 8, 8, 8}});
    auto w   = mm->add_parameter("w", {migraphx::shape::float_type, {8, 8, 3, 3}});
    mm->add_return({mm->add_instruction(migraphx::make_op("convolution"), x, w)});

    migraphx::compile_options options;
    if(not carveout.is_null())
        options.backend_options["lds_carveout"] = carveout;
    p.compile(migraphx::make_target("gpu"), options);
    return p;
}

struct kernel_carveout
{
    std::string symbol_name;
    int carveout;
};

std::vector<kernel_carveout> carveouts(const migraphx::program& p)
{
    std::vector<kernel_carveout> result;
    for(const auto& ins : *p.get_main_module())
    {
        if(ins.name() != "gpu::code_object")
            continue;
        const auto& op = migraphx::any_cast<migraphx::gpu::code_object_op>(ins.get_operator());
        result.push_back({op.symbol_name, op.get_lds_carveout()});
    }
    return result;
}

bool is_mlir_kernel(const kernel_carveout& k)
{
    return migraphx::starts_with(k.symbol_name, "mlir");
}

} // namespace

// Leaving lds_carveout unset means no request, on every device: every kernel stays -1.
// MIGraphX does not infer a default from the device; a real value is only ever set when the
// caller explicitly requests one.
void expect_unset_carveouts(const std::vector<kernel_carveout>& ks)
{
    EXPECT(not ks.empty());
    EXPECT(std::all_of(ks.begin(), ks.end(), [](const auto& k) { return k.carveout == -1; }));
}

TEST_CASE(compile_without_setting)
{
    expect_unset_carveouts(carveouts(compile_conv(migraphx::value{})));
}

TEST_CASE(compile_with_unset_value) { expect_unset_carveouts(carveouts(compile_conv(-1))); }

// An explicitly requested, in-range value is forwarded unconditionally; MIGraphX does not throw
// based on the device. Only MLIR kernels take the setting — rocMLIR applies it where the device
// has a configurable split and ignores it elsewhere.
TEST_CASE(compile_with_setting)
{
    auto ks = carveouts(compile_conv(100));
    EXPECT(not ks.empty());
    const auto non_mlir_unset = std::all_of(
        ks.begin(), ks.end(), [](const auto& k) { return is_mlir_kernel(k) or k.carveout == -1; });
    EXPECT(non_mlir_unset);
    const auto mlir_count = std::count_if(ks.begin(), ks.end(), is_mlir_kernel);
    EXPECT(mlir_count > 0);
    EXPECT(std::all_of(ks.begin(), ks.end(), [](const auto& k) {
        return not is_mlir_kernel(k) or k.carveout == 100;
    }));
}

TEST_CASE(invalid_setting)
{
    EXPECT(test::throws([] { compile_conv(101); }));
    EXPECT(test::throws([] { compile_conv(-2); }));
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
