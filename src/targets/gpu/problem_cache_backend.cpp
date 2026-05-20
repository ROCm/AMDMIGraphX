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
 *
 */
#include <migraphx/gpu/problem_cache_backend.hpp>
#include <migraphx/gpu/json_cache_backend.hpp>
#include <migraphx/logger.hpp>
#include <stdexcept>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

// --- parse_device_key ---

cache_device_key parse_device_key(const std::string& s)
{
    if(s.empty())
        return {};

    // Format: "gpu_arch|cu_count|wavefront_size"
    auto first_pipe = s.find('|');
    if(first_pipe == std::string::npos)
        return {};
    auto second_pipe = s.find('|', first_pipe + 1);
    if(second_pipe == std::string::npos)
        return {};

    cache_device_key dk;
    dk.gpu_arch = s.substr(0, first_pipe);
    try
    {
        dk.cu_count = std::stoi(s.substr(first_pipe + 1, second_pipe - first_pipe - 1));
        dk.wavefront_size = std::stoi(s.substr(second_pipe + 1));
    }
    catch(...)
    {
        return {};
    }
    return dk;
}

// --- Factory functions ---

problem_cache_backend make_cache_backend(const std::string& type)
{
    if(type == "json")
        return problem_cache_backend{json_cache_backend{}};
    throw std::runtime_error("Problem cache backend not available: " + type);
}

problem_cache_backend make_default_cache_backend()
{
    return problem_cache_backend{json_cache_backend{}};
}

problem_cache_backend
make_cache_backend_with_fallback(const std::string& explicit_backend)
{
    if(!explicit_backend.empty() && explicit_backend != "json")
    {
        log::warn() << "Unknown cache backend '" << explicit_backend
                    << "'. Falling back to JSON.\n";
    }
    return problem_cache_backend{json_cache_backend{}};
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
