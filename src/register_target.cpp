/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2022 Advanced Micro Devices, Inc. All rights reserved.
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
#include <string>
#include <unordered_map>
#include <migraphx/register_target.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/dynamic_loader.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

void store_target_lib(const dynamic_loader& lib)
{
    static std::vector<dynamic_loader> target_loader;
    target_loader.emplace_back(lib);
}

std::unordered_map<std::string, target>& target_map()
{
    static std::unordered_map<std::string, target> m; // NOLINT
    return m;
}

void register_target_init() { (void)target_map(); }

void unregister_target(const std::string& name)
{
    assert(target_map().count(name));
    target_map().erase(name);
}

void register_target(const target& t) { target_map()[t.name()] = t; }

target make_target(const std::string& name)
{
    if(not contains(target_map(), name))
    {
#ifdef _WIN32
        std::string target_name = "migraphx_" + name + ".dll";
#else
        std::string target_name = "libmigraphx_" + name + ".so";
#endif
        store_target_lib(dynamic_loader(target_name));
    }
    const auto it = target_map().find(name);
    if(it == target_map().end())
    {
        MIGRAPHX_THROW("Requested target '" + name + "' is not loaded or not supported");
    }
    return it->second;
}

std::vector<std::string> get_targets()
{
    std::vector<std::string> result;
    std::transform(target_map().begin(),
                   target_map().end(),
                   std::back_inserter(result),
                   [&](auto&& p) { return p.first; });
    std::sort(result.begin(), result.end());
    return result;
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
