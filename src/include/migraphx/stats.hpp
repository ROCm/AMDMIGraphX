/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2025 Advanced Micro Devices, Inc. All rights reserved.
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
#ifndef MIGRAPHX_GUARD_MIGRAPHX_STATS_HPP
#define MIGRAPHX_GUARD_MIGRAPHX_STATS_HPP

#include <migraphx/float_equal.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

double common_average(const std::vector<double>& v)
{
    assert(std::is_sorted(v.begin(), v.end()));

    std::size_t n = v.size() / 4;
    double total  = std::accumulate(v.begin() + n, v.end() - n, 0.0);
    return total / std::distance(v.begin() + n, v.end() - n);
}

double mean(const std::vector<double>& v)
{
    double total = std::accumulate(v.begin(), v.end(), 0.0);
    return total / v.size();
}

double median(const std::vector<double>& v)
{
    size_t mid = v.size() / 2;
    if(v.size() % 2 == 0)
    {
        return (v[mid - 1] + v[mid]) / 2.0;
    }
    else
    {
        return v[mid];
    }
}

std::vector<double> abs_dev(const std::vector<double>& v)
{
    assert(std::is_sorted(v.begin(), v.end()));

    double median_v = median(v);
    std::vector<double> abs_dev(v.size());

    std::transform(v.begin(), v.end(), std::back_inserter(abs_dev), [&](double x) {
        return std::abs(x - median_v);
    });

    return abs_dev;
}

std::vector<double> modified_z_scores(const std::vector<double>& v)
{
    assert(std::is_sorted(v.begin(), v.end()));

    double median_v               = median(v);
    std::vector<double> abs_dev_v = abs_dev(v);
    double mad_v                  = median(abs_dev_v);

    // if MAD == 0, then unable to calculate modified z-score
    if(float_equal(mad_v, 0.0))
    {
        return std::vector<double>(v.size(), 0.0);
    }

    std::vector<double> mod_z_scores(v.size());
    std::transform(v.begin(), v.end(), std::back_inserter(mod_z_scores), [&](double x) {
        return 0.6745 * (x - median_v) / mad_v;
    });
    return mod_z_scores;
}

double mod_z_average(const std::vector<double>& v, double z_threshold)
{
    assert(std::is_sorted(v.begin(), v.end()));

    std::vector<double> mod_z_scores = modified_z_scores(v);
    std::vector<double> filtered_v;

    // if the modified z-score at the particular index is within the threshold,
    // add the value at that index to the filtered dataset
    for(size_t i = 0; i < v.size(); i++)
    {
        if(mod_z_scores[i] <= z_threshold)
            filtered_v.push_back(v[i]);
    }
    std::cout << "num outliers removed: " << v.size() - filtered_v.size() << std::endl;
    return mean(filtered_v);
}

double percentile(const std::vector<double>& v, double percentile)
{
    assert(std::is_sorted(v.begin(), v.end()));

    size_t index = (percentile * (v.size() - 1));
    return v[index];
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
#endif // MIGRAPHX_GUARD_MIGRAPHX_STATS_HPP
