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
#include <mutex>
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
    static std::mutex mutex;
    std::unique_lock<std::mutex> lock(mutex);

    target_loader.emplace_back(lib);
}

/**
 * Returns a singleton map of targets and names.
 */
std::unordered_map<std::string, target>& target_map()
{
    static std::unordered_map<std::string, target> m; // NOLINT
    return m;
}

/**
 * Returns a singleton mutex used by the various register_target methods.
 */
std::mutex& target_mutex()
{
    static std::mutex m; // NOLINT
    return m;
}

void register_target_init() { (void)target_map(); }

void unregister_target(const std::string& name)
{
    std::unique_lock<std::mutex> lock(target_mutex());
    assert(target_map().count(name));
    target_map().erase(name);
}

/**
 * Insert a target name in the target_map; thread safe.
 */
void register_target(const target& t)
{
    std::unique_lock<std::mutex> lock(target_mutex());
    target_map()[t.name()] = t;
}

/**
 * Search for a target by name in the target_map; thread-safe.
 */
migraphx::optional<target> find_target(const std::string& name)
{
    // search for match or return none
    std::unique_lock<std::mutex> lock(target_mutex());
    const auto it = target_map().find(name);

    if(it == target_map().end())
        return nullopt;
    return it->second;
}

/**
 * Get a target by name.  Load target library and register target if needed.
 * Thread safe.
 */
target make_target(const std::string& name)
{
    //   no lock required here
    auto t = find_target(name);
    if(t == nullopt)
    {
        std::string target_name = "libmigraphx_" + name + ".so";
        // register_target is called by this
        store_target_lib(dynamic_loader(target_name));
        t = find_target(name);
    }
    // at this point we should always have a target

    return *t;
}

/**
 * Get list of names of registered targets.
 */
std::vector<std::string> get_targets()
{
    std::unique_lock<std::mutex> lock(target_mutex());
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
