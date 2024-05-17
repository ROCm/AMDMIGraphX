/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2023 Advanced Micro Devices, Inc. All rights reserved.
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

#include <onnx_test.hpp>

TEST_CASE(reversesequence_time_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();

    int batch_axis = 1;
    int time_axis  = 0;

    migraphx::shape sx{migraphx::shape::float_type, {4, 4}};
    auto input = mm->add_parameter("x", sx);

    int batch_size                     = sx.lens()[batch_axis];
    int time_size                      = sx.lens()[time_axis];
    std::vector<int64_t> sequence_lens = {4, 3, 2, 1};

    auto add_slice =
        [&mm, &input, batch_axis, time_axis](int b_start, int b_end, int t_start, int t_end) {
            return mm->add_instruction(migraphx::make_op("slice",
                                                         {{"axes", {batch_axis, time_axis}},
                                                          {"starts", {b_start, t_start}},
                                                          {"ends", {b_end, t_end}}}),
                                       input);
        };

    migraphx::instruction_ref ret;
    for(int b = 0; b < batch_size - 1; ++b)
    {
        auto s0 = add_slice(b, b + 1, 0, sequence_lens[b]);
        s0      = mm->add_instruction(migraphx::make_op("reverse", {{"axes", {time_axis}}}), s0);
        if(sequence_lens[b] < time_size)
        {
            auto s1 = add_slice(b, b + 1, sequence_lens[b], time_size);
            s0 = mm->add_instruction(migraphx::make_op("concat", {{"axis", time_axis}}), s0, s1);
        }
        if(b == 0)
        {
            ret = s0;
        }
        else
        {
            ret = mm->add_instruction(migraphx::make_op("concat", {{"axis", batch_axis}}), ret, s0);
        }
    }
    auto s0 = add_slice(batch_size - 1, batch_size, 0, time_size);
    ret     = mm->add_instruction(migraphx::make_op("concat", {{"axis", batch_axis}}), ret, s0);
    mm->add_return({ret});

    auto prog = read_onnx("reversesequence_time_test.onnx");
    EXPECT(p == prog);
}
