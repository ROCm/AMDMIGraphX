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
#pragma once

#include <chrono>
#include <iostream>
#include <numeric>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

// For debugging the timing of each part
class Timer
{
    public:
    explicit Timer(const std::string& tag_, bool verbose_ = false) : tag(tag_), verbose(verbose_)
    {
        std::cout << "Timer " << tag << " created." << std::endl;
    }
    void add(const std::chrono::duration<double, std::milli>& in)
    {
        std::thread::id id = std::this_thread::get_id();
        count[id] += 1;
        total[id] += in;
        if(verbose)
            measurements[id].emplace_back(in);
    }
    ~Timer()
    {
        auto total_accum = std::accumulate(
            std::begin(total),
            std::end(total),
            0,
            [](int64_t value,
               std::pair<std::thread::id, std::chrono::duration<double, std::milli>> p) {
                return value + p.second.count();
            });

        auto count_accum = std::accumulate(
            std::begin(count),
            std::end(count),
            0,
            [](size_t value, std::pair<std::thread::id, size_t> p) { return value + p.second; });

        std::cout << "Timer " << tag << " reports " << (double)total_accum / count_accum
                  << " ms per call for " << count_accum << " times." << std::endl;
        if(verbose)
        {
            std::cout << "  Measurements=[";
            for(const auto& m : measurements)
            {
                std::cout << " Thread " << m.first << ":  {";
                for(const auto& d : m.second)
                {
                    std::cout << d.count() << ",";
                }

                std::cout << "},";
            }
            std::cout << "]" << std::endl;
        }
    }

    private:
    std::string tag;
    bool verbose;
    std::unordered_map<std::thread::id, std::chrono::duration<double, std::milli>> total;
    std::unordered_map<std::thread::id, std::vector<std::chrono::duration<double, std::milli>>>
        measurements;
    std::unordered_map<std::thread::id, size_t> count;
};
