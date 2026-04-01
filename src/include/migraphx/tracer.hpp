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
#ifndef MIGRAPHX_GUARD_RTGLIB_TRACER_HPP
#define MIGRAPHX_GUARD_RTGLIB_TRACER_HPP

#include <sstream>
#include <migraphx/functional.hpp>
#include <migraphx/config.hpp>
#include <migraphx/logger.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct tracer
{
    tracer() {}

    explicit tracer(bool enable) : enabled_(enable)
    {
        if(enabled_ and not log::is_enabled(log::severity::trace) and
           not log::is_severity_explicit())
            log::set_severity(log::severity::trace);
    }

    bool enabled() const { return enabled_; }

    template <class... Ts>
    void operator()(const Ts&... xs) const
    {
        if(enabled_)
        {
            std::ostringstream ss;
            swallow{ss << xs...};
            log::trace() << ss.str();
        }
    }

    private:
    bool enabled_ = false;
};

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
