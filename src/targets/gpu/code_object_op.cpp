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
#include <migraphx/register_op.hpp>
#include <migraphx/pmr/vector.hpp>
#include <any>
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
    if(not migraphx::equal(flatten(einputs), flatten(inputs), &shape::is_compatible))
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

// Convert a value::binary kernel arg into a kernel_argument.
// The binary blob's size (1, 4, or 8) determines the ABI type.
// We memcpy into a std::any-held scalar so the pointer stays valid for pack_args.
static void push_karg(std::vector<kernel_argument>& kargs,
                      std::vector<std::any>& storage,
                      const value& v)
{
    const auto& bin = v.get_binary();
    auto sz         = bin.size();
    switch(sz)
    {
    case 1:
    {
        auto& ref = storage.emplace_back(bin[0]);
        kargs.emplace_back(*std::any_cast<uint8_t>(&ref));
        break;
    }
    case 4:
    {
        uint32_t tmp;
        std::memcpy(&tmp, bin.data(), 4);
        auto& ref = storage.emplace_back(tmp);
        kargs.emplace_back(*std::any_cast<uint32_t>(&ref));
        break;
    }
    case 8:
    {
        uint64_t tmp;
        std::memcpy(&tmp, bin.data(), 8);
        auto& ref = storage.emplace_back(tmp);
        kargs.emplace_back(*std::any_cast<uint64_t>(&ref));
        break;
    }
    default: MIGRAPHX_THROW("push_karg: unsupported size " + std::to_string(sz));
    }
}

argument
code_object_op::compute(context& ctx, const shape&, const std::vector<argument>& args) const
{
    if(not kernel_args.empty())
    {
        // Patch buffer pointers into a mutable copy of kernel_args
        auto local_args = kernel_args;
        for(const auto& [karg_idx, arg_idx] : runtime_arg_indices)
        {
            auto ptr = reinterpret_cast<uint64_t>(args[arg_idx].data());
            value::binary b(sizeof(uint64_t));
            std::memcpy(b.data(), &ptr, sizeof(uint64_t));
            local_args[karg_idx] = value(std::move(b));
        }

        // Build kernel_argument vector in index order.
        // std::any storage keeps cast-back values alive for pack_args.
        std::vector<kernel_argument> kargs;
        std::vector<std::any> storage;
        kargs.reserve(local_args.size());
        storage.reserve(local_args.size());

        for(auto& [idx, v] : local_args)
        {
            push_karg(kargs, storage, v);
        }

        auto [start, stop] = ctx.get_perf_events();
        k.launch(ctx.get_stream().get(), global, local, kargs, start, stop);
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
        visit_flatten_args(args, [&](const auto& fargs) {
            kargs.reserve(fargs.size());
            std::transform(fargs.begin(),
                           fargs.end(),
                           std::back_inserter(kargs),
                           [](const argument& a) { return a.data(); });
        });
        auto [start, stop] = ctx.get_perf_events();
        k.launch(ctx.get_stream().get(), global, local, kargs, start, stop);
    }
    return args[get_output_arg(args.size())];
}
void code_object_op::finalize(context&, const shape&, const std::vector<shape>&)
{
    assert(not code_object.empty());
    k = kernel(code_object, symbol_name);
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
