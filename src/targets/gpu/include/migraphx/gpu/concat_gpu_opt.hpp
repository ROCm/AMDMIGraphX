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
#ifndef MIGRAPHX_GUARD_RTGLIB_CONCAT_GPU_OPT_HPP
#define MIGRAPHX_GUARD_RTGLIB_CONCAT_GPU_OPT_HPP

#include <migraphx/op/concat.hpp>
#include <migraphx/operation.hpp>
#include <migraphx/serialize.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/gpu/allocation_model.hpp>

namespace migraphx {
namespace gpu {

struct concat_gpu_optimization
{
    std::string allocate() const { return "hip::allocate"; }
    optional<migraphx::op::concat> get_concat(const migraphx::operation& op) const
    {
        if(op.name() != "gpu::precompile_op")
            return nullopt;
        auto r = from_value<operation>(op.to_value().at("op"));
        if(r.name() == "concat")
            return any_cast<migraphx::op::concat>(r);
        return nullopt;
    }
    bool supports_non_packed_output(instruction_ref ins, std::size_t) const
    {
        return ins->name() == "gpu::precompile_op";
    }
    // JIT kernels are compiled from the output buffer they are given, so a
    // static slice layout can be pinned. rocMLIR convolution kernels keep the
    // layout baked into the submodule: pinning a concat slice makes that
    // buffer disagree with the kernel, and the precompile then fails or hangs.
    static bool compiler_honors_output_shape(const operation& op)
    {
        const auto name = op.name();
        if(name == "quant_convolution" or name == "convolution")
            return false;
        if(name != "gpu::mlir_op")
            return true;
        const auto wrapped = from_value<operation>(op.to_value().at("op"));
        return wrapped.name() != "quant_convolution" and wrapped.name() != "convolution";
    }
    bool supports_output_shape(instruction_ref ins, const shape& s) const
    {
        if(ins->name() != "gpu::precompile_op" or s.dynamic())
            return false;
        const auto inner = from_value<operation>(ins->get_operator().to_value().at("op"));
        return compiler_honors_output_shape(inner);
    }
    void set_output_shape(instruction_ref ins, const shape& s) const
    {
        auto v            = ins->get_operator().to_value();
        v["output_shape"] = to_value(s);
        ins->replace(make_op("gpu::precompile_op", v));
    }
    bool supports_non_packed_input(instruction_ref ins, std::size_t axis) const
    {
        if(is_context_free(ins->get_operator()) or
           contains({"gpu::precompile_op", "gpu::contiguous", "hip::copy"}, ins->name()))
            return true;
        if(ins->name() == "gpu::hipblaslt_op" or contains(ins->name(), "gemm"))
            return axis < ins->get_shape().ndim() - 2;
        return false;
    }
    gpu_allocation_model allocation() const { return gpu_allocation_model{}; }
};

} // namespace gpu

} // namespace migraphx

#endif
