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
#include <migraphx/fp8_ocp_to_nanoo.hpp>
#include <migraphx/matcher.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
    
struct match_fp8ocp_dq_convert_to_fp8nanoo
{
    /**
     * Match dequantizelinear instructions.
     * Bind the scale and zero_point inputs.
     */
    static auto dequantizelinear_op(const std::string& scale, const std::string& zp)
    {
        return match::name("dequantizelinear")(
            match::arg(1)(match::skip_broadcasts(match::is_constant().bind(scale))),
            match::arg(2)(match::skip_broadcasts(match::is_constant().bind(zp))));
    }

    auto matcher() const
    {
        return dequantizelinear_op("scale", "zp");
    }

    void fp8_ocp_to_nanoo::apply(module_pass_manager& mpm) const
    {
        // Check if input is a quantizelinear instruction.
        // Change how the quantizelinear works if it is by changing the last convert
        // to where instructions into a bit_cast instruction.
        // 
        // if input is a parameter just add the where instructions and bit_cast.
        //
        // Multiply the scale of the dequantizelinear by 2.
    }

};
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
