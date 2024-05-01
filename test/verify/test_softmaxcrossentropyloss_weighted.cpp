/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2024 Advanced Micro Devices, Inc. All rights reserved.
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

template <const char* Mode, migraphx::shape::type_t ScoreType, migraphx::shape::type_t LabelType>
struct test_softmaxcrossentropyloss_weighted
    : verify_program<test_softmaxcrossentropyloss_weighted<Mode, ScoreType, LabelType>>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape s{ScoreType, {4, 100}};
        migraphx::shape l{LabelType, {4}};
        migraphx::shape w{ScoreType, {4}};
        auto scores  = mm->add_parameter("scores", s);
        auto labels  = mm->add_parameter("labels", l);
        auto weights = mm->add_parameter("weights", w);
        mm->add_instruction(migraphx::make_op("softmaxcrossentropyloss", {{"reduction", Mode}}),
                            scores,
                            labels,
                            weights);

        return p;
    }
};

static const char mean[] = "mean";
static const char none[] = "none";
static const char sum[]  = "sum";

template struct test_softmaxcrossentropyloss_weighted<none,
                                                      migraphx::shape::float_type,
                                                      migraphx::shape::int32_type>;
template struct test_softmaxcrossentropyloss_weighted<none,
                                                      migraphx::shape::double_type,
                                                      migraphx::shape::int32_type>;
template struct test_softmaxcrossentropyloss_weighted<none,
                                                      migraphx::shape::half_type,
                                                      migraphx::shape::int32_type>;
template struct test_softmaxcrossentropyloss_weighted<none,
                                                      migraphx::shape::half_type,
                                                      migraphx::shape::int64_type>;
template struct test_softmaxcrossentropyloss_weighted<none,
                                                      migraphx::shape::float_type,
                                                      migraphx::shape::int64_type>;
template struct test_softmaxcrossentropyloss_weighted<none,
                                                      migraphx::shape::double_type,
                                                      migraphx::shape::int64_type>;
template struct test_softmaxcrossentropyloss_weighted<mean,
                                                      migraphx::shape::float_type,
                                                      migraphx::shape::int32_type>;
template struct test_softmaxcrossentropyloss_weighted<mean,
                                                      migraphx::shape::double_type,
                                                      migraphx::shape::int32_type>;
template struct test_softmaxcrossentropyloss_weighted<mean,
                                                      migraphx::shape::half_type,
                                                      migraphx::shape::int32_type>;
template struct test_softmaxcrossentropyloss_weighted<mean,
                                                      migraphx::shape::half_type,
                                                      migraphx::shape::int64_type>;
template struct test_softmaxcrossentropyloss_weighted<mean,
                                                      migraphx::shape::float_type,
                                                      migraphx::shape::int64_type>;
template struct test_softmaxcrossentropyloss_weighted<mean,
                                                      migraphx::shape::double_type,
                                                      migraphx::shape::int64_type>;
template struct test_softmaxcrossentropyloss_weighted<sum,
                                                      migraphx::shape::float_type,
                                                      migraphx::shape::int32_type>;
template struct test_softmaxcrossentropyloss_weighted<sum,
                                                      migraphx::shape::double_type,
                                                      migraphx::shape::int32_type>;
template struct test_softmaxcrossentropyloss_weighted<sum,
                                                      migraphx::shape::half_type,
                                                      migraphx::shape::int32_type>;
template struct test_softmaxcrossentropyloss_weighted<sum,
                                                      migraphx::shape::half_type,
                                                      migraphx::shape::int64_type>;
template struct test_softmaxcrossentropyloss_weighted<sum,
                                                      migraphx::shape::float_type,
                                                      migraphx::shape::int64_type>;
template struct test_softmaxcrossentropyloss_weighted<sum,
                                                      migraphx::shape::double_type,
                                                      migraphx::shape::int64_type>;
