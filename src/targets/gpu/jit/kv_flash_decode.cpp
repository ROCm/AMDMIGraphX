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
#include <migraphx/gpu/compiler.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/gpu/compile_hip.hpp>
#include <migraphx/gpu/compile_hip_code_object.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/stringutils.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

// NOLINTNEXTLINE
static const char* const kv_flash_decode_splitk_kernel = R"__migraphx__(
#include <args.hpp>
#include <migraphx/kernels/kv_flash_decode.hpp>

namespace migraphx {

extern "C" {

MIGRAPHX_GLOBAL void ${kernel}(${params})
{
    transform_args(make_tensors())(${args})([](auto... xs) {
        kv_flash_decode_splitk<${q_heads}, ${kv_heads}, ${groups}>(xs..., ${scale});
    });
}

}

} // namespace migraphx

)__migraphx__";

// NOLINTNEXTLINE
static const char* const kv_flash_decode_reduce_kernel = R"__migraphx__(
#include <args.hpp>
#include <migraphx/kernels/kv_flash_decode.hpp>

namespace migraphx {

extern "C" {

MIGRAPHX_GLOBAL void ${kernel}(${params})
{
    transform_args(make_tensors())(${args})([](auto... xs) {
        kv_flash_decode_reduce(xs...);
    });
}

}

} // namespace migraphx

)__migraphx__";

struct kv_flash_decode_splitk_compiler : compiler<kv_flash_decode_splitk_compiler>
{
    std::vector<std::string> names() const { return {"gpu::kv_flash_decode_splitk"}; }

    operation compile_op(context& ctx, const std::vector<shape>& inputs, const value& v) const
    {
        auto q_heads            = v.at("q_heads").to<std::size_t>();
        auto kv_heads           = v.at("kv_heads").to<std::size_t>();
        auto groups             = v.at("groups").to<std::size_t>();
        const auto& k_shape     = inputs.at(1);
        auto batch              = k_shape.lens().at(0);
        const std::size_t block = 256;

        hip_compile_options options;
        options.set_launch_params(v, batch * kv_heads * groups * block, block);
        options.inputs      = inputs;
        options.output      = inputs.back();
        options.kernel_name = "kv_flash_decode_splitk_kernel";

        auto src = interpolate_string(kv_flash_decode_splitk_kernel,
                                      {{"params", enum_params(inputs.size(), "void * private_p")},
                                       {"args", enum_params(inputs.size(), "private_p")},
                                       {"kernel", options.kernel_name},
                                       {"q_heads", std::to_string(q_heads)},
                                       {"kv_heads", std::to_string(kv_heads)},
                                       {"groups", std::to_string(groups)},
                                       {"scale", std::to_string(v.at("scale").to<float>()) + "f"}});
        return compile_hip_code_object(ctx, src, options);
    }

    compiler_replace compile(context& ctx, instruction_ref ins, const operation& op) const
    {
        return compile_op(ctx, to_shapes(ins->inputs()), op.to_value());
    }
};

struct kv_flash_decode_reduce_compiler : compiler<kv_flash_decode_reduce_compiler>
{
    std::vector<std::string> names() const { return {"gpu::kv_flash_decode_reduce"}; }

    operation compile_op(context& ctx, const std::vector<shape>& inputs, const value& v) const
    {
        const auto& p_shape = inputs.front();
        auto batch          = p_shape.lens().at(0);
        auto q_heads        = p_shape.lens().at(1);
        auto d              = p_shape.lens().at(3) - 1;

        hip_compile_options options;
        options.set_launch_params(v, batch * q_heads * d, d);
        options.inputs      = inputs;
        options.output      = inputs.back();
        options.kernel_name = "kv_flash_decode_reduce_kernel";

        auto src = interpolate_string(kv_flash_decode_reduce_kernel,
                                      {{"params", enum_params(inputs.size(), "void * private_p")},
                                       {"args", enum_params(inputs.size(), "private_p")},
                                       {"kernel", options.kernel_name}});
        return compile_hip_code_object(ctx, src, options);
    }

    compiler_replace compile(context& ctx, instruction_ref ins, const operation& op) const
    {
        return compile_op(ctx, to_shapes(ins->inputs()), op.to_value());
    }
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
