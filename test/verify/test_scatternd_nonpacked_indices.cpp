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
#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/shape.hpp>

// Regression test: scatternd must honor non-packed strides on its indices
// input. The vision_encoder pattern produces indices via
//     unsqueeze[axes=1] -> transpose[{0,2,1}] -> slice -> concat
// which leaves the indices tensor with shape {N, k} but non-packed strides
// {1, N}. A naive kernel that reads two contiguous int64 entries per row will
// pull the wrong values and direct every thread to the same output cell.
template <migraphx::shape::type_t DType>
struct test_scatternd_nonpacked_indices : verify_program<test_scatternd_nonpacked_indices<DType>>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm   = p.get_main_module();
        auto itype = migraphx::shape::int64_type;

        constexpr std::size_t n = 8;
        constexpr std::size_t k = 2;
        migraphx::shape src_idx_shape{itype, {k, n}};
        std::vector<int64_t> src_idx_vec{0, 0, 0, 0, 0, 0, 0, 0, 7, 6, 5, 4, 3, 2, 1, 0};
        auto src_idx = mm->add_literal(migraphx::literal{src_idx_shape, src_idx_vec});
        auto indices =
            mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0}}}), src_idx);

        migraphx::shape data_shape{DType, {1, n}};
        auto data = mm->add_parameter("data", data_shape);

        migraphx::shape upd_shape{DType, {n}};
        auto updates = mm->add_parameter("update", upd_shape);

        auto scatternd =
            mm->add_instruction(migraphx::make_op("scatternd_none"), data, indices, updates);
        mm->add_return({scatternd});

        return p;
    }
};

template struct test_scatternd_nonpacked_indices<migraphx::shape::float_type>;
