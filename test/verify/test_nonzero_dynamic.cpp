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
#include <migraphx/generate.hpp>
#include <migraphx/make_op.hpp>

// The GPU kernel bakes the input lengths into its code object, so a dynamic input takes the host
// ref fallback in gpu lowering. Two non-fixed dimensions keep split_single_dyn_dim from
// specializing the module first, which is what leaves a dynamic shape for that fallback to catch.
// Run below the maximum in both dimensions so the padding in the indices output is exercised too.
template <migraphx::shape::type_t DType>
struct test_nonzero_dynamic : verify_program<test_nonzero_dynamic<DType>>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape s{DType, {{1, 4}, {1, 4}}};
        auto x       = mm->add_parameter("data", s);
        auto nz      = mm->add_instruction(migraphx::make_op("nonzero"), x);
        auto indices = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), nz);
        auto num_nonzero =
            mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 1}}), nz);
        mm->add_return({indices, num_nonzero});

        return p;
    }

    std::unordered_map<std::string, migraphx::shape> get_test_dims() const
    {
        return {{"data", migraphx::shape{DType, {2, 3}}}};
    }
};

template struct test_nonzero_dynamic<migraphx::shape::bool_type>;
template struct test_nonzero_dynamic<migraphx::shape::float_type>;
