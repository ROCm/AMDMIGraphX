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
#include <migraphx/instruction.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/program.hpp>
#include <migraphx/register_target.hpp>
#include <migraphx/verify.hpp>
#include <random>

#include <test.hpp>

TEST_CASE(multinomial_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();

    size_t sample_size = 100000;
    float seed         = 0.0f;
    std::mt19937 gen(seed);
    std::uniform_real_distribution<> dis(0.0, 1.0);
    std::vector<float> rand_samples(sample_size);
    std::generate(rand_samples.begin(), rand_samples.end(), [&]() { return dis(gen); });
    migraphx::shape rs{migraphx::shape::float_type, {1, sample_size}};
    auto rs_lit = mm->add_literal(migraphx::literal{rs, rand_samples});

    migraphx::shape s{migraphx::shape::float_type, {1, 5}};
    std::vector<int> dist{15, 25, 15, 25, 20};
    std::vector<float> data(5);
    std::transform(dist.begin(), dist.end(), data.begin(), [&](auto d) { return std::log(d); });
    auto input = mm->add_literal(migraphx::literal(s, data));

    auto maxes = mm->add_instruction(migraphx::make_op("reduce_max", {{"axes", {1}}}), input);
    auto mb_maxes =
        mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {1, 5}}}), maxes);
    auto cdf = mm->add_instruction(migraphx::make_op("sub"), input, mb_maxes);
    cdf      = mm->add_instruction(migraphx::make_op("exp"), cdf);
    cdf      = mm->add_instruction(
        migraphx::make_op("prefix_scan_sum", {{"axis", 1}, {"exclusive", false}}), cdf);

    mm->add_instruction(migraphx::make_op("multinomial"), cdf, rs_lit);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<int32_t> result_vec(sample_size);
    result.visit([&](auto output) { result_vec.assign(output.begin(), output.end()); });

    std::vector<int> res_dist(5, 0);
    for(const auto& r : result_vec)
        res_dist[r]++;
    auto dist_sum     = std::accumulate(dist.begin(), dist.end(), 0);
    auto res_dist_sum = std::accumulate(res_dist.begin(), res_dist.end(), 0);
    std::vector<float> norm(5);
    std::vector<float> res_norm(5);
    std::transform(dist.begin(), dist.end(), norm.begin(), [&](auto n) {
        return static_cast<double>(n) / dist_sum;
    });
    std::transform(res_dist.begin(), res_dist.end(), res_norm.begin(), [&](auto n) {
        return static_cast<double>(n) / res_dist_sum;
    });
    EXPECT(migraphx::verify::verify_range_with_threshold(
        res_norm, migraphx::verify::expected{norm}, migraphx::verify::threshold{0.01}));
}
