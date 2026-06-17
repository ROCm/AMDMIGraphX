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
#ifndef MIGRAPHX_GUARD_OPERATORS_EXTERNAL_WEIGHT_HPP
#define MIGRAPHX_GUARD_OPERATORS_EXTERNAL_WEIGHT_HPP

#include <migraphx/check_shapes.hpp>
#include <migraphx/config.hpp>
#include <migraphx/shape.hpp>
#include <migraphx/argument.hpp>
#include <migraphx/context.hpp>
#include <migraphx/errors.hpp>
#include <string>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

/**
 * A constant tensor whose bytes live in an external file rather than inside the
 * program. It is a leaf op (no inputs) carrying the tensor shape and the
 * location/offset/length needed to read the raw bytes from disk.
 *
 * A producer (e.g. the ONNX parser with keep_weights_external) records weight
 * references directly in the IR so a compiled template can later be baked into a
 * self-contained program by load_external_weights, which replaces each
 * external_weight with a literal.
 */
struct external_weight
{
    shape s;
    std::string location = "";
    std::size_t offset    = 0;
    std::size_t length    = 0;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.s, "shape"),
                    f(self.location, "location"),
                    f(self.offset, "offset"),
                    f(self.length, "length"));
    }

    std::string name() const { return "external_weight"; }

    shape compute_shape(const std::vector<shape>& inputs) const
    {
        migraphx::check_shapes{inputs, *this}.has(0);
        return s;
    }

    // Taking a context (like @param / @literal) marks the op as not
    // context-free, so propagate_constant won't try to evaluate it.
    argument compute(context&, const shape&, const std::vector<argument>&) const
    {
        MIGRAPHX_THROW("EXTERNAL_WEIGHT: cannot evaluate an unresolved external weight; the "
                       "weights must be loaded (see load_external_weights) before running");
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
