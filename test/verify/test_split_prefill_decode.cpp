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
#include <migraphx/make_op.hpp>
#include <migraphx/program.hpp>
#include <migraphx/sym.hpp>

namespace {

migraphx::program create_split_program()
{
    using dd             = migraphx::shape::dynamic_dimension;
    auto sequence_length = migraphx::sym::var("sequence_length", {1, 4});

    migraphx::program p;
    auto* mm = p.get_main_module();
    auto data =
        mm->add_parameter("data",
                          migraphx::shape{migraphx::shape::float_type,
                                          {dd{migraphx::sym::lit(2)}, dd{sequence_length}}});
    auto result = mm->add_instruction(migraphx::make_op("relu"), data);
    mm->add_return({result});
    return p;
}

} // namespace

struct test_split_prefill_decode_decode : verify_program<test_split_prefill_decode_decode>
{
    migraphx::program create_program() const { return create_split_program(); }

    std::unordered_map<std::string, migraphx::shape> get_test_dims() const
    {
        return {{"data", {migraphx::shape::float_type, {2, 1}}}};
    }
};

struct test_split_prefill_decode_prefill : verify_program<test_split_prefill_decode_prefill>
{
    migraphx::program create_program() const { return create_split_program(); }

    std::unordered_map<std::string, migraphx::shape> get_test_dims() const
    {
        return {{"data", {migraphx::shape::float_type, {2, 4}}}};
    }
};
