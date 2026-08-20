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
#ifndef MIGRAPHX_GUARD_RTGLIB_FP_TO_DOUBLE_HPP
#define MIGRAPHX_GUARD_RTGLIB_FP_TO_DOUBLE_HPP

#include <set>
#include <string>
#include <migraphx/config.hpp>
#include <migraphx/fp8_types.hpp>
#include <migraphx/shape.hpp>
#include <migraphx/pass_manager.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct module;

/**
 * Convert floating point values to double precision. Also removes fake quantization (q/dq pairs),
 * which would otherwise keep reintroducing the narrow type the conversion is meant to remove.
 *
 * fp4x2 is excluded because it is not computable: shape::visit throws for it, so a convert to or
 * from it cannot be evaluated.
 */
struct MIGRAPHX_EXPORT fp_to_double
{
    std::set<shape::type_t> convert_fp_types = [] {
        auto types = fp8_types{}.get();
        types.insert(
            {shape::type_t::half_type, shape::type_t::float_type, shape::type_t::bf16_type});
        return types;
    }();
    std::string name() const { return "fp_to_double"; }
    void apply(module_pass_manager& mpm) const;
};

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
