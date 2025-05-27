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
#include <migraphx/env.hpp>
#include <migraphx/ranges.hpp>
#include <cstdlib>
#include <mutex>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

static auto access_envs()
{
    static std::map<std::string, std::string> obj;
    static std::mutex m;
    return [&](auto f) {
        std::lock_guard<std::mutex> lock(m);
        return f(obj);
    };
}

bool enabled(const char* name)
{
    auto e = env(name);
    if(e.empty() or not contains({"1", "enable", "enabled", "yes", "true"}, e.front()))
        return false;
    access_envs()([&](auto& m) { m[name] = e.front(); });
    return true;
}

bool disabled(const char* name) { return not enabled(name); }

std::size_t value_of(const char* name, std::size_t fallback)
{
    auto e = env(name);
    if(e.empty())
        return fallback;
    access_envs()([&](auto& m) { m[name] = e.front(); });
    return std::stoul(e.front());
}

std::string string_value_of(const char* name, std::string fallback)
{
    auto e = env(name);
    if(e.empty())
        return fallback;
    auto rv         = e.front();
    access_envs()([&](auto& m) { m[name] = rv; });
    return rv;
}

std::vector<std::string> env(const char* name)
{
    auto* p = std::getenv(name);
    if(p == nullptr)
        return {};
    return {{p}};
}

std::map<std::string, std::string> get_all_envs()
{
    return access_envs()([&](auto& m) { return m; });
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
