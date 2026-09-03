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
#ifndef MIGRAPHX_GUARD_RTGLIB_SPLIT_SEQ_LEN_HPP
#define MIGRAPHX_GUARD_RTGLIB_SPLIT_SEQ_LEN_HPP

#include <string>
#include <migraphx/pass_manager.hpp>
#include <migraphx/config.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

/**
 * Split a kv-cache model with a symbolic sequence-length dimension into two static
 * submodules selected at run time: one for decode (the minimum sequence length) and one
 * for prefill (the maximum). Prefill inputs are zero-padded up to the maximum with
 * fixed_pad and the padded output rows are trimmed back off with dyn_slice, so any
 * sequence length in the interval runs on one of the two static graphs.
 */
struct MIGRAPHX_EXPORT split_seq_len
{
    std::string name() const { return "split_seq_len"; }
    void apply(module_pass_manager&) const;
};

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
