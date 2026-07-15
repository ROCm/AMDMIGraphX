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
#include <migraphx/gpu/code_object_op.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/gpu/pack_args.hpp>
#include <migraphx/gpu/hip.hpp>
#include <migraphx/register_op.hpp>
#include <migraphx/pmr/vector.hpp>
#include <algorithm>
#include <cstring>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

MIGRAPHX_REGISTER_OP(code_object_op);

shape code_object_op::compute_shape(std::vector<shape> inputs) const
{
    std::transform(inputs.begin(), inputs.end(), inputs.begin(), [](const shape& s) {
        return s.normalize_standard();
    });
    auto einputs = expected_inputs;
    std::transform(einputs.begin(), einputs.end(), einputs.begin(), [](const shape& s) {
        return s.normalize_standard();
    });
    if(not migraphx::equal(
           flatten_tuple_shapes(einputs), flatten_tuple_shapes(inputs), &shape::is_compatible))
        MIGRAPHX_THROW("Input shapes have changed: [" + to_string_range(einputs) + "] -> [" +
                       to_string_range(inputs) + "]");
    auto output_buffer_shape = inputs.at(get_output_arg(inputs.size()));
    if(not shape::is_compatible(output_buffer_shape, output))
        MIGRAPHX_THROW("Output buffer [" + to_string(output_buffer_shape) +
                       "] doesn't match the expected output shape from the kernel [" +
                       to_string(output) + "]");
    return output;
}

static bool needs_flatten(const std::vector<argument>& args)
{
    return std::any_of(args.begin(), args.end(), [&](const argument& arg) {
        return arg.get_shape().type() == shape::tuple_type;
    });
}

template <class F>
static void visit_flatten_args(const std::vector<argument>& args, F f)
{
    if(needs_flatten(args))
        f(flatten(args));
    else
        f(args);
}

static std::vector<argument> ensure_gpu_kernel_args(const std::vector<argument>& args,
                                                    pmr::vector<argument>& temps)
{
    std::vector<argument> gpu_args;
    gpu_args.reserve(args.size());
    for(const auto& arg : args)
    {
        if(arg.data() != nullptr and not is_gpu_device_ptr(arg.data()))
        {
            temps.push_back(to_gpu(arg));
            gpu_args.push_back(temps.back());
        }
        else
        {
            gpu_args.push_back(arg);
        }
    }
    return gpu_args;
}

argument
code_object_op::compute(context& ctx, const shape&, const std::vector<argument>& args) const
{
#if MIGRAPHX_HAS_PMR
    std::array<char, 256> temp_storage;
    std::pmr::monotonic_buffer_resource temp_resource{temp_storage.data(), temp_storage.size()};
    pmr::vector<argument> temp_gpu_args(&temp_resource);
#else
    pmr::vector<argument> temp_gpu_args;
#endif
    auto gpu_args = ensure_gpu_kernel_args(args, temp_gpu_args);
    if(not packed_kernargs.empty())
    {
        // Fast path: copy pre-packed buffer, patch pointer slots, launch.
        // Back the buffer with a monotonic_buffer_resource so it lives on the
        // stack when the payload fits, and falls back to heap allocation only
        // when it doesn't.
        auto sz = packed_kernargs.size();
#if MIGRAPHX_HAS_PMR
        alignas(8) std::array<char, 512> storage;
        std::pmr::monotonic_buffer_resource resource{storage.data(), storage.size()};
        pmr::vector<char> buf(sz, &resource);
#else
        pmr::vector<char> buf(sz);
#endif
        std::copy(packed_kernargs.begin(), packed_kernargs.end(), buf.begin());

        for(const auto& [arg_idx, byte_offset] : runtime_arg_offsets)
        {
            auto ptr           = reinterpret_cast<uint64_t>(gpu_args[arg_idx].data());
            const auto* p_byte = reinterpret_cast<const char*>(&ptr);
            std::copy(p_byte, p_byte + sizeof(uint64_t), buf.begin() + byte_offset);
        }

        auto [start, stop] = ctx.get_perf_events();
        k.launch(ctx.get_stream().get(), global, local, buf.data(), sz, start, stop);
    }
    else
    {
#if MIGRAPHX_HAS_PMR
        std::array<char, 256> storage;
        std::pmr::monotonic_buffer_resource resource{storage.data(), storage.size()};
        pmr::vector<void*> kargs(&resource);
#else
        pmr::vector<void*> kargs;
#endif
        visit_flatten_args(gpu_args, [&](const auto& fargs) {
            kargs.reserve(fargs.size());
            std::transform(fargs.begin(),
                           fargs.end(),
                           std::back_inserter(kargs),
                           [](const argument& a) { return a.data(); });
        });
        auto [start, stop] = ctx.get_perf_events();
        k.launch(ctx.get_stream().get(), global, local, kargs, start, stop);
    }
    return gpu_args[get_output_arg(gpu_args.size())];
}
void code_object_op::finalize(context&, const shape&, const std::vector<shape>&)
{
    assert(not code_object.empty());
    k = kernel(code_object, symbol_name);

    if(kernel_args.empty())
        return;

    // Represent each entry as a kernel_argument_value and let pack_args do
    // the padding/insertion. Pointer slots become 8 zero bytes with align=8;
    // their byte offsets are replayed below for compute()-time patching.
    std::vector<kernel_argument_value> flat;
    flat.reserve(kernel_args.size());
    std::vector<bool> is_pointer;
    is_pointer.reserve(kernel_args.size());

    for(const auto& [idx, v] : kernel_args)
    {
        if(v.data.empty())
        {
            kernel_argument_value slot;
            slot.align = 8;
            slot.data.assign(8, 0);
            flat.push_back(std::move(slot));
            is_pointer.push_back(true);
        }
        else
        {
            flat.push_back(v);
            is_pointer.push_back(false);
        }
    }

    packed_kernargs = pack_args(flat);

    // Replay the same alignment math pack_args uses to recover pointer-slot
    // offsets in the packed buffer.
    runtime_arg_offsets.clear();
    std::size_t arg_counter = 0;
    std::size_t pos         = 0;
    for(std::size_t i = 0; i < flat.size(); ++i)
    {
        auto align = flat[i].align;
        pos += (align - (pos % align)) % align;
        if(is_pointer[i])
            runtime_arg_offsets.emplace_back(arg_counter++, pos);
        pos += flat[i].data.size();
    }
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
