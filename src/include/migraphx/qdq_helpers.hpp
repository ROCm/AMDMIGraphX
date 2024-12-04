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

#ifndef MIGRAPHX_GUARD_RTGLIB_QDQ_HELPERS_HPP
#define MIGRAPHX_GUARD_RTGLIB_QDQ_HELPERS_HPP

#include <migraphx/config.hpp>
#include <migraphx/matcher.hpp>
#include <migraphx/make_op.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

namespace {

template <class... Ms>
auto skip_post_dq_ops(Ms... ms)
{
    return match::skip(match::name(
        "broadcast", "multibroadcast", "contiguous", "transpose", "reshape", "convert"))(ms...);
}

static std::unordered_set<std::string> get_quantizable_op_names()
{
    static std::unordered_set<std::string> s = {"convolution", "dot"};
    return s;
}

static auto dequantizelinear_op(const std::string& scale, const std::string& zp)
{
    return match::name("dequantizelinear")(
        match::arg(1)(match::skip_broadcasts(match::is_constant().bind(scale))),
        match::arg(2)(match::skip_broadcasts(match::is_constant().bind(zp))));
}
} // namespace
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
