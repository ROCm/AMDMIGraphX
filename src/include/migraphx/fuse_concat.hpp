/*
 * The MIT License (MIT)
 *
<<<<<<<< HEAD:src/include/migraphx/op/scatter_min.hpp
 * Copyright (c) 2015-2023 Advanced Micro Devices, Inc. All rights reserved.
========
 * Copyright (c) 2015-2024 Advanced Micro Devices, Inc. All rights reserved.
>>>>>>>> 9941849ca4ffe76c66866b4af74bebfe40eb3c22:src/include/migraphx/fuse_concat.hpp
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
<<<<<<<< HEAD:src/include/migraphx/op/scatter_min.hpp
#ifndef MIGRAPHX_GUARD_OPERATORS_SCATTER_ELEMENTS_MIN_HPP
#define MIGRAPHX_GUARD_OPERATORS_SCATTER_ELEMENTS_MIN_HPP

#include <migraphx/op/scatter_op.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

struct scatter_min : public scatter_op<scatter_min>
{
    auto reduction() const
    {
        return [](auto& x, const auto& y) { x = std::min(x, y); };
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
========
#ifndef MIGRAPHX_GUARD_MIGRAPHX_FUSE_CONCAT_HPP
#define MIGRAPHX_GUARD_MIGRAPHX_FUSE_CONCAT_HPP

#include <migraphx/config.hpp>
#include <string>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct module_pass_manager;

struct MIGRAPHX_EXPORT fuse_concat
{
    std::string name() const { return "fuse_concat"; }
    void apply(module_pass_manager& mpm) const;
};

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
#endif // MIGRAPHX_GUARD_MIGRAPHX_FUSE_CONCAT_HPP
>>>>>>>> 9941849ca4ffe76c66866b4af74bebfe40eb3c22:src/include/migraphx/fuse_concat.hpp
