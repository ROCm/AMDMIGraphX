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
#include <migraphx/register_target.hpp>
#include <migraphx/target.hpp>
#include <thread>
#include "test.hpp"

TEST_CASE(make_target)
{
    for(const auto& name : migraphx::get_targets())
    {
        auto t = migraphx::make_target(name);
        CHECK(t.name() == name);
    }
}

TEST_CASE(make_invalid_target)
{
    EXPECT(test::throws([&] { migraphx::make_target("mi100"); }));
}

TEST_CASE(targets)
{
    // GCC doesn't load libmigraphx_ref unless necesssary even though it is linked to the test.
    // Force it to load by making ref target
#if defined(__GNUC__) && !defined(__clang__)
    auto ref_target = migraphx::make_target("ref");
#endif
    auto ts = migraphx::get_targets();
    EXPECT(ts.size() >= 1);
}

TEST_CASE(concurrent_targets)
{
    std::vector<std::thread> threads;
#ifdef HAVE_GPU
    std::string target_name = "gpu";
#elif defined(HAVE_CPU)
    std::string target_name = "cpu";
#elif defined(HAVE_FPGA)
    std::string target_name = "fpga";
#else
    std::string target_name = "ref";
#endif

    auto n_threads = std::thread::hardware_concurrency() * 4;

    for(auto i = 0u; i < n_threads; i++)
    {
        auto thread_body = [&target_name]() {

            // remove all existing targets, if any
            std::vector<std::string> target_list = migraphx::get_targets();
            for(auto tt : target_list)
                migraphx::unregister_target(tt);
                
            auto ref_target = migraphx::make_target(target_name);
            migraphx::register_target(ref_target);
            EXPECT(test::throws([&] { ref_target = migraphx::make_target("xyz"); }));

            migraphx::get_targets();
        };

        threads.emplace_back(thread_body);
    }
    for(auto& tt : threads)
        tt.join();
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
