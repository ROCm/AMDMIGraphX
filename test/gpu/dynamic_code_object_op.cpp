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

#include <cstdlib>
#include <migraphx/gpu/lowering.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/operation.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/module.hpp>
#include <migraphx/register_target.hpp>
#include <test.hpp>
#include <pointwise.hpp>

static void run_lowering(migraphx::program& p, bool offload_copy = false)
{
    auto ctx = migraphx::gpu::context{};
    migraphx::run_passes(*p.get_main_module(), {migraphx::gpu::lowering{&ctx, offload_copy}});
}

TEST_CASE(dynamic_code_object_op)
{
    migraphx::shape s{migraphx::shape::float_type, {{1, 3}, {2, 4}, {6, 6}}};
    migraphx::program p1;
    auto* mm = p1.get_main_module();
    auto a   = mm->add_parameter("a", s);
    auto b   = mm->add_parameter("b", s);

    auto pw               = add_pointwise(p1, "main:pointwise0", {a, b}, single_pointwise("add"));
    auto pw_module_inputs = pw->module_inputs();

    mm->add_return({pw});

    run_lowering(p1);

    bool found = false;
    for(auto ins : iterator_for(*p1.get_main_module()))
    {
        if(ins->name() == "gpu::dynamic_code_object_op")
        {
            found = true;
            EXPECT(ins->module_inputs() == pw_module_inputs);
        }
    }
    EXPECT(found);
}

TEST_CASE(dynamic_code_object_zero_output)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto x   = mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {1, 1, 2, 2}});
    auto rois =
        mm->add_parameter("rois", migraphx::shape{migraphx::shape::float_type, {{0, 2}, {4, 4}}});
    auto batch_ind =
        mm->add_parameter("batch_ind",
                          migraphx::shape{migraphx::shape::int64_type,
                                          std::vector<migraphx::shape::dynamic_dimension>{{0, 2}}});
    auto r = mm->add_instruction(migraphx::make_op("roialign",
                                                   {{"output_height", int64_t{1}},
                                                    {"output_width", int64_t{1}},
                                                    {"sampling_ratio", int64_t{1}}}),
                                 x,
                                 rois,
                                 batch_ind);
    mm->add_return({r});

    auto target = migraphx::make_target("gpu");
    p.compile(target);

    std::vector<float> x_data       = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> roi_data     = {0.0f, 0.0f, 1.0f, 1.0f};
    std::vector<int64_t> batch_data = {0};
    migraphx::parameter_map params;
    params["x"] = target.copy_to(
        migraphx::argument{{migraphx::shape::float_type, {1, 1, 2, 2}}, x_data.data()});
    params["rois"] =
        target.copy_to(migraphx::argument{{migraphx::shape::float_type, {1, 4}}, roi_data.data()});
    params["batch_ind"] =
        target.copy_to(migraphx::argument{{migraphx::shape::int64_type, {1}}, batch_data.data()});
    for(const auto& [name, s] : p.get_parameter_shapes())
    {
        if(params.count(name) > 0)
            continue;
        auto allocation_shape = s.dynamic() ? migraphx::shape{s.type(), s.max_lens()} : s;
        params[name]          = target.allocate(allocation_shape);
    }

    auto result = target.copy_from(p.eval(params).back());
    EXPECT(result.get_shape() == migraphx::shape{migraphx::shape::float_type, {1, 1, 1, 1}});
    EXPECT(result.to_vector<float>() == std::vector<float>{1.0f});

    params["rois"]      = target.allocate({migraphx::shape::float_type, {0, 4}});
    params["batch_ind"] = target.allocate({migraphx::shape::int64_type, {0}});

    auto empty_result = p.eval(params).back();
    EXPECT(empty_result.get_shape() == migraphx::shape{migraphx::shape::float_type, {0, 1, 1, 1}});
    EXPECT(empty_result.get_shape().elements() == 0);
}

int main(int argc, const char* argv[])
{
#ifdef _WIN32
    _putenv_s("MIGRAPHX_ENABLE_FULL_DYNAMIC", "1");
#else
    setenv("MIGRAPHX_ENABLE_FULL_DYNAMIC", "1", 1);
#endif
    test::run(argc, argv);
}
