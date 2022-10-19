/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2022 Advanced Micro Devices, Inc. All rights reserved.
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
#include <fstream>
#include <filesystem>
#include <migraphx/gpu/compiler.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/gpu/context.hpp>

#include <migraphx/gpu/compile_hip_code_object.hpp>
#include <migraphx/gpu/compile_hip.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/env.hpp>
#include <migraphx/reduce_dims.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/eliminate_common_subexpression.hpp>
#include <migraphx/module.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/env.hpp>
#include <migraphx/file_buffer.hpp>

std::vector<std::string>&
get_instance(std::size_t i, const std::function<bool(const std::vector<std::string>&)>& pred);

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

namespace gpu {

MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_LOG_CK_GEMM);
MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_CK_TUNING);

// NOLINTNEXTLINE
static const char* const ck_gemm_kernel = R"__migraphx__(
#include <args.hpp>
#include <migraphx/kernels/ck_gemm.hpp>

namespace migraphx {

extern "C" {

__global__ void ck_gemm_kernel(void* a_p, void* b_p, void* c_p)
{
    make_tensors()(a_p, b_p, c_p)([&](auto a, auto b, auto c) {
        ck_gemm<CK_DeviceGemmMultipleD<${instance}>>(a, b, c);
    });
}

}

} // namespace migraphx

)__migraphx__";

static std::size_t int_div_ceil(std::size_t x, std::size_t y) { return (x + y - 1) / y; }

static std::size_t block_size_index = 15;

static std::size_t padding_index = 13;

static std::size_t get_block_size(const std::vector<std::string>& s)
{
    return std::stoull(s[block_size_index]);
}

static std::size_t get_grid_size(const std::vector<std::string>& s, std::size_t m, std::size_t n)
{
    auto mpb = std::stoull(s[block_size_index + 1]);
    auto npb = std::stoull(s[block_size_index + 2]);
    return int_div_ceil(m, mpb) * int_div_ceil(n, npb);
}

static void set_padding(std::vector<std::string>& s, const std::string p) { s[padding_index] = p; }

template <class F, class Action>
auto action_decorate(F f, Action action)
{
    return [=](auto&&... xs) {
        action();
        f(std::forward<decltype(xs)>(xs)...);
    };
}

using tuning_entry = std::pair<std::vector<shape>, size_t>;
static std::vector<tuning_entry> read_tuning(const std::string& s)
{
    if(not fs::exists(s))
        return {};
    return from_value<std::vector<tuning_entry>>(from_json_string(read_string(s)));
}

static std::size_t get_tuning_for(const std::vector<shape>& inputs)
{
    static auto tuning = read_tuning(string_value_of(MIGRAPHX_CK_TUNING{}, ""));
    if(tuning.empty())
        std::cout << "*********** Warning: No CK tuning!" << std::endl;
    auto it = std::find_if(
        tuning.begin(), tuning.end(), [&](const auto& p) { return p.first == inputs; });
    if(it == tuning.end())
    {
        std::cout << "*********** Warning: CK tuning missing for config!" << std::endl;
        return 4;
    }
    return it->second;
}

struct ck_gemm_compiler : compiler<ck_gemm_compiler>
{
    static std::string get_layout(const shape& s)
    {
        return s.transposed() ? "ck::tensor_layout::gemm::ColumnMajor"
                              : "ck::tensor_layout::gemm::RowMajor";
    }

    static std::string get_type(const shape& s)
    {
        if(s.type() == shape::half_type)
            return "ck::half_t";
        return shape::cpp_type(s.type());
    }

    std::vector<std::string> names() const { return {"ck_gemm", "gpu::ck_gemm"}; }

    operation compile_op(context& /* ctx */, const std::vector<shape>& inputs, const value& v) const
    {
        auto a_shape = inputs[0];
        auto b_shape = inputs[1];
        auto c_shape = inputs[2];

        auto m = c_shape.lens().front();
        auto n = c_shape.lens().back();
        auto k = a_shape.lens().back();

        auto i         = v.get("tuning_val", get_tuning_for(inputs));
        auto& instance = get_instance(i, [&](const auto& x) -> bool {
            return get_layout(a_shape) == x[0] and get_layout(b_shape) == x[1] and
                   get_layout(c_shape) == x[3] and get_type(a_shape) == x[4] and
                   get_type(b_shape) == x[5] and get_type(c_shape) == x[9];
        });

        const bool pad_m = m % 8;
        const bool pad_n = n % 8;
        const bool pad_k = k % 8;

        if(pad_m or pad_n or pad_k)
        {
            std::string padding_t = "ck::tensor_operation::device::GemmSpecialization::";
            padding_t += pad_m ? "M" : "";
            padding_t += pad_n ? "N" : "";
            padding_t += pad_k ? "K" : "";
            padding_t += "Padding";
            set_padding(instance, padding_t);
        }

        hip_compile_options options;
        auto block_size = get_block_size(instance);
        auto grid_size  = get_grid_size(instance, m, n);
        options.set_launch_params(v, grid_size * block_size, block_size);
        options.inputs         = inputs;
        options.output         = c_shape;
        options.kernel_name    = "ck_gemm_kernel";
        options.virtual_inputs = inputs;

        auto src = interpolate_string(ck_gemm_kernel, {{"instance", join_strings(instance, ",")}});

        return compile_hip_code_object(src, options);
    }

    compiler_replace compile(context& ctx, instruction_ref ins, const operation& op) const
    {
        auto shapes = to_shapes(ins->inputs());
        return action_decorate(replace(compile_op(ctx, shapes, op.to_value())), [=] {
            if(enabled(MIGRAPHX_LOG_CK_GEMM{}))
                std::cout << "ck_gemm: " << to_json_string(to_value(shapes)) << std::endl;
        });
    }
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
