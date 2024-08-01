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

#include <migraphx/matcher.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/register_op.hpp>
#include <migraphx/gpu/fuse_ck.hpp>
#include <migraphx/gpu/gemm_softmax_gemm.hpp>
#include <migraphx/gpu/device_name.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
struct module;

namespace gpu {

struct ck_gemm
{
    operation op = make_op("dot");

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.op, "op"));
    }

    std::string name() const { return "gpu::ck_gemm"; }

    void check_gemm_shape(const shape& s) const
    {
        if(not contains(range(s.strides().rbegin(), s.strides().rbegin() + 3), 1))
            MIGRAPHX_THROW("Invalid shape for ck_gemm");
    }

    shape compute_shape(std::vector<shape> inputs, const std::vector<module_ref>& mods) const
    {
        check_shapes{inputs, *this}.same_ndims();
        if(inputs.size() < 2)
            MIGRAPHX_THROW(name() + ": should have at least two inputs.");
        auto a = inputs[0];
        auto b = inputs[1];
        for(const auto& input : inputs)
            check_gemm_shape(input);
        auto r = op.compute_shape({a, b});
        if(mods.empty())
            return r;
        return r.with_type(mods.front()->get_output_shapes().front().type());
    }

    static bool is_ck_supported_type(shape::type_t t)
    {
        return contains({shape::half_type, shape::int8_type, shape::int32_type}, t);
    }
};
MIGRAPHX_REGISTER_OP(ck_gemm);

struct ck_gemm_softmax_gemm : gemm_softmax_gemm
{
    std::string name() const { return "gpu::ck_gemm_softmax_gemm"; }
};
MIGRAPHX_REGISTER_OP(ck_gemm_softmax_gemm);

void fuse_ck::apply(module_pass_manager& mpm) const
{
    match::find_matches(mpm, find_ck_gemm{});
}

} // namespace gpu

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
