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
 */
#ifndef MIGRAPHX_GUARD_RTGLIB_COMPILE_OPTIONS_HPP
#define MIGRAPHX_GUARD_RTGLIB_COMPILE_OPTIONS_HPP

#include <migraphx/config.hpp>
#include <migraphx/tracer.hpp>
#include <migraphx/value.hpp>
#include <string>
#include <unordered_map>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct compile_options
{
    /**
     * Have MIGX allocate memory for parameters and add instructions
     * to copy parameters and output to/from an offload device like a GPU.
     */
    bool offload_copy = false;

    bool fast_math       = true;
    bool exhaustive_tune = false;

    /**
     * Backend-specific options keyed by name. Targets can read these to
     * configure compilation in a way that is opaque to the core engine.
     */
    std::unordered_map<std::string, value> backend_options;

    tracer trace{};
};

/**
 * Merge the backend options from an object value into the compile options.
 * Each top-level key of the object becomes an entry in backend_options.
 */
inline void set_backend_options(compile_options& options, const value& v)
{
    if(not v.is_object())
        MIGRAPHX_THROW("set_backend_options expects an object value");
    for(const auto& opt : v)
        options.backend_options[opt.get_key()] = opt.without_key();
}

/**
 * Read a backend option by name, converting it to `To`. Returns `default_value`
 * when the option is not present.
 */
template <class To>
To get_backend_option(const compile_options& options,
                      const std::string& name,
                      const To& default_value)
{
    auto it = options.backend_options.find(name);
    if(it == options.backend_options.end())
        return default_value;
    return it->second.to<To>();
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
