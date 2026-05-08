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
#ifndef MIGRAPHX_GUARD_AMDMIGRAPHX_MLSS_MHA_OP_HPP
#define MIGRAPHX_GUARD_AMDMIGRAPHX_MLSS_MHA_OP_HPP

#include <migraphx/config.hpp>
#include <migraphx/value.hpp>
#include <migraphx/argument.hpp>
#include <migraphx/functional.hpp>
#include <migraphx/gpu/kernel.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

struct context;

struct mlss_mha_op
{
    value::binary code_object{};
    std::string symbol_name{};
    std::size_t global = 0;
    std::size_t local  = 0;
    float scale        = 1.0f;
    // `k` is not reflected — it is re-created in finalize() from code_object/symbol_name
    kernel k{};

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.code_object, "code_object"),
                    f(self.symbol_name, "symbol_name"),
                    f(self.global, "global"),
                    f(self.local, "local"),
                    f(self.scale, "scale"));
    }

#ifdef MIGRAPHX_HAS_MLSS_HEADERS
    static mlss_mha_op make_gfx1201_fp16_packed_qkv(float scale, std::size_t global, std::size_t local);
#endif

    std::string name() const { return "gpu::mlss_mha"; }
    shape compute_shape(std::vector<shape> inputs) const;
    argument compute(context& ctx, const shape& output_shape, const std::vector<argument>& args) const;
    void finalize(context&, const shape&, const std::vector<shape>&);
    std::vector<std::size_t> output_alias(const std::vector<shape>& shapes) const
    {
        return {shapes.size() - 1};
    }
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
