/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
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
 *
 */
/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2025 Advanced Micro Devices, Inc. All rights
 * reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to do so, subject to the
 * following conditions:
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
#include <migraphx/literal.hpp>

struct test_gather_flatten_stride_grid : verify_program<test_gather_flatten_stride_grid>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();

        auto data    = mm->add_parameter("X", {migraphx::shape::float_type, {1, 3, 16, 16}});
        auto flatten = mm->add_instruction(migraphx::make_op("reshape", {{"dims", {768}}}), data);

        migraphx::shape indices_shape{migraphx::shape::int32_type, {1, 3, 4, 4}};
        std::vector<int32_t> indices = {17,  21,  25,  29,  81,  85,  89,  93,  145, 149, 153, 157,
                                        209, 213, 217, 221, 273, 277, 281, 285, 337, 341, 345, 349,
                                        401, 405, 409, 413, 465, 469, 473, 477, 529, 533, 537, 541,
                                        593, 597, 601, 605, 657, 661, 665, 669, 721, 725, 729, 733};
        auto indices_literal         = mm->add_literal(migraphx::literal{indices_shape, indices});

        auto gather = mm->add_instruction(migraphx::make_op("gather"), flatten, indices_literal);
        mm->add_return({gather});

        return p;
    }
};
