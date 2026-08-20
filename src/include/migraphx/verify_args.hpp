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
#ifndef MIGRAPHX_GUARD_RTGLIB_VERIFY_ARGS_HPP
#define MIGRAPHX_GUARD_RTGLIB_VERIFY_ARGS_HPP

#include <migraphx/verify.hpp>
#include <migraphx/argument.hpp>
#include <migraphx/config.hpp>
#include <migraphx/optional.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

/// Multiple of the type's epsilon used as the default rms bound.
inline constexpr std::size_t default_tolerance = 80;

/// Baseline elementwise tolerances for a data type.
/// A single rounding step in a low precision type can exceed the fp32 defaults.
MIGRAPHX_EXPORT verify::tolerance tolerance_for_type(shape::type_t type);

/// Tolerance applied when a caller does not supply one: elementwise bounds from the precision
/// class, and an rms bound scaled from the type's epsilon.
MIGRAPHX_EXPORT verify::tolerance
default_tolerance_for(shape::type_t type, std::size_t rms_multiplier = default_tolerance);

MIGRAPHX_EXPORT bool verify_args(const std::string& name,
                                 const argument& target_arg,
                                 const verify::expected<argument>& ref_arg,
                                 verify::tolerance);

/// Verify against `tols`, or against `default_tolerance_for` the target's type when unset.
MIGRAPHX_EXPORT bool verify_args_with_tolerance(const std::string& name,
                                                const argument& target_arg,
                                                const verify::expected<argument>& ref_arg,
                                                optional<verify::tolerance> tols = nullopt);

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
