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
#include <migraphx/gpu/compile_hip_code_object.hpp>
#include <migraphx/gpu/compile_hip.hpp>
#include <migraphx/gpu/compile_gen.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/reduce_dims.hpp>
#include <migraphx/float_equal.hpp>
#include <migraphx/op/pad.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

using namespace migraphx::gpu::gen; // NOLINT

static const char* const pad_kernel = R"__migraphx__(
#include <migraphx/kernels/pad.hpp>
#include <migraphx/kernels/index.hpp>
#include <migraphx/kernels/ops.hpp>
#include <args.hpp>

namespace migraphx {

extern "C" {
MIGRAPHX_GLOBAL void pad_kernel(void* input_p, void* output_p) 
{
    auto offsets = diff_ints<${offsets}>{};
    auto idx     = make_index();
    make_tensors()(input_p, output_p)([&](auto input, auto output) {
        pad(idx, offsets, input, output, ${pad_val}, ${pad_mode});
    });
}
    
}

} // namespace migraphx

)__migraphx__";

static const char* const pad_border_kernel = R"__migraphx__(
#include <migraphx/kernels/pad.hpp>
#include <migraphx/kernels/index.hpp>
#include <migraphx/kernels/ops.hpp>
#include <args.hpp>

namespace migraphx {

extern "C" {
MIGRAPHX_GLOBAL void pad_border_kernel(void* input_p, void* output_p) 
{
    auto offsets = diff_ints<${offsets}>{};
    auto idx     = make_index();
    make_tensors()(input_p, output_p)([&](auto input, auto output) {
        pad_border(idx, offsets, input, output, ${pad_val}, ${pad_mode});
    });
}
    
}

} // namespace migraphx

)__migraphx__";

// Merge the dims the input and output can share and return the reduced shapes along with the
// offsets of the input inside the output
static std::pair<std::vector<shape>, std::vector<std::size_t>>
reduce_pad_dims(const std::vector<shape>& inputs, const value& v)
{
    auto padding    = v.at("pads").to_vector<int64_t>();
    auto input_lens = inputs.front().lens();
    std::vector<size_t> offsets(input_lens.size());
    std::copy(padding.begin(), padding.begin() + offsets.size(), offsets.begin());

    auto offset_lens = input_lens;
    std::transform(input_lens.begin(),
                   input_lens.end(),
                   offsets.begin(),
                   offset_lens.begin(),
                   [&](auto input, auto offset) { return input + offset; });

    auto vinputs = inputs;
    vinputs.push_back(inputs.front().with_lens(offset_lens));
    auto rinputs = reduce_dims(normalize_permutation(vinputs));

    auto rinput_lens  = rinputs.front().lens();
    auto roffset_lens = rinputs.back().lens();
    std::vector<size_t> roffsets(roffset_lens.size());
    std::transform(rinput_lens.begin(),
                   rinput_lens.end(),
                   roffset_lens.begin(),
                   roffsets.begin(),
                   [](auto input, auto offset_dim) { return offset_dim - input; });
    rinputs.pop_back();
    return {rinputs, roffsets};
}

static std::string pad_value_string(const value& v)
{
    auto pad_val = v.get("value", 0.f);
    if(float_equal(pad_val, std::numeric_limits<float>::lowest()))
        return "lowest{}";
    if(float_equal(pad_val, std::numeric_limits<float>::max()))
        return "highest{}";
    if(auto nonfinite = nonfinite_cpp_literal(pad_val))
        return *nonfinite;
    return to_string(pad_val);
}

static std::string pad_mode_string(const value& v)
{
    auto pad_mode_val = v.get("mode", static_cast<int>(op::pad::constant_pad));
    if(pad_mode_val == op::pad::reflect_pad)
        return "migraphx::pad_reflect{}";
    if(pad_mode_val == op::pad::edge_pad)
        return "migraphx::pad_edge{}";
    return "migraphx::pad_constant{}";
}

static hip_src make_pad_src(const value& v, hip_compile_options options, const char* src)
{
    auto [rinputs, roffsets] = reduce_pad_dims(options.inputs, v);
    options.virtual_inputs   = rinputs;
    auto content             = interpolate_string(src,
                                                  {{"pad_val", pad_value_string(v)},
                                                   {"offsets", to_string_range(roffsets)},
                                                   {"pad_mode", pad_mode_string(v)}});
    return {content, options};
}

struct pad_compiler : compiler<pad_compiler>
{
    std::vector<std::string> names() const { return {"pad"}; }

    hip_src make_src(const context& ctx, const std::vector<shape>& inputs, const value& v) const
    {
        hip_compile_options options;
        options.inputs      = inputs;
        options.output      = inputs.back();
        options.kernel_name = "pad_kernel";
        options.set_launch_params(v, compute_global_for(ctx, inputs.back().elements()));
        return make_pad_src(v, options, pad_kernel);
    }

    operation compile_op(context& ctx, const std::vector<shape>& inputs, const value& v) const
    {
        return compile_hip_code_object(ctx, make_src(ctx, inputs, v));
    }

    compiler_replace compile(context& ctx, instruction_ref ins, const operation& op) const
    {
        return compile_op(ctx, to_shapes(ins->inputs()), op.to_value());
    }

    std::string
    compile_key(const context& ctx, instruction_ref ins, const operation& op, const value&) const
    {
        return hip_compile_key(ctx, make_src(ctx, to_shapes(ins->inputs()), op.to_value()));
    }
};

struct pad_border_compiler : compiler<pad_border_compiler>
{
    std::vector<std::string> names() const { return {"pad_border"}; }

    hip_src make_src(const context& ctx, const std::vector<shape>& inputs, const value& v) const
    {
        hip_compile_options options;
        options.inputs      = inputs;
        options.output      = inputs.back();
        options.kernel_name = "pad_border_kernel";
        // Only the padding region is written
        auto nborder = inputs.back().elements() - inputs.front().elements();
        options.set_launch_params(v, compute_global_for(ctx, nborder));
        return make_pad_src(v, options, pad_border_kernel);
    }

    operation compile_op(context& ctx, const std::vector<shape>& inputs, const value& v) const
    {
        return compile_hip_code_object(ctx, make_src(ctx, inputs, v));
    }

    compiler_replace compile(context& ctx, instruction_ref ins, const operation& op) const
    {
        return compile_op(ctx, to_shapes(ins->inputs()), op.to_value());
    }

    std::string
    compile_key(const context& ctx, instruction_ref ins, const operation& op, const value&) const
    {
        return hip_compile_key(ctx, make_src(ctx, to_shapes(ins->inputs()), op.to_value()));
    }
};
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
