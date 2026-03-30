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
#include <migraphx/gpu/context.hpp>
#include <migraphx/gpu/lowering.hpp>
#include <migraphx/auto_contiguous.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/program.hpp>
#include <migraphx/make_op.hpp>
#include <test.hpp>

static void run_lowering(migraphx::module& m, bool offload_copy = false)
{
    auto ctx = migraphx::gpu::context{};
    migraphx::run_passes(m,
                         {migraphx::auto_contiguous{},
                          migraphx::gpu::lowering{&ctx, offload_copy},
                          migraphx::dead_code_elimination{}});
}

// After lowering, a slice with runtime inputs should have hip::copy_from_gpu
// and hip::sync_stream inserted for the metadata inputs.
TEST_CASE(dyn_slice_lowering_runtime_inputs)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape data_s{migraphx::shape::float_type, {{2, 4}, {2, 4}, {3, 8}}};
    migraphx::shape idx_s{migraphx::shape::int32_type, {1}};

    auto data   = mm->add_parameter("data", data_s);
    auto starts = mm->add_parameter("starts", idx_s);
    auto ends   = mm->add_parameter("ends", idx_s);
    auto sl     = mm->add_instruction(
        migraphx::make_op("slice", {{"axes", {2}}}), data, starts, ends);
    mm->add_return({sl});

    run_lowering(*mm);

    bool has_copy_from_gpu = false;
    bool has_sync_stream   = false;
    for(auto ins : migraphx::iterator_for(*mm))
    {
        if(ins->name() == "hip::copy_from_gpu")
            has_copy_from_gpu = true;
        if(ins->name() == "hip::sync_stream")
            has_sync_stream = true;
    }
    EXPECT(has_copy_from_gpu);
    EXPECT(has_sync_stream);
}

// A slice with only 1 input (all attributes inline) should not be modified
// by the dynamic slice lowering.
TEST_CASE(dyn_slice_lowering_single_input)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape data_s{migraphx::shape::float_type, {2, 2, 4}};

    auto data = mm->add_parameter("data", data_s);
    auto sl   = mm->add_instruction(
        migraphx::make_op("slice", {{"axes", {2}}, {"starts", {0}}, {"ends", {2}}}), data);
    mm->add_return({sl});

    run_lowering(*mm);

    bool has_copy_from_gpu = false;
    bool has_sync_stream   = false;
    for(auto ins : migraphx::iterator_for(*mm))
    {
        if(ins->name() == "hip::copy_from_gpu")
            has_copy_from_gpu = true;
        if(ins->name() == "hip::sync_stream")
            has_sync_stream = true;
    }
    EXPECT(not has_copy_from_gpu);
    EXPECT(not has_sync_stream);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
