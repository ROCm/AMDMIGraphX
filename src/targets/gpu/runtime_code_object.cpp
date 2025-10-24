/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2025 Advanced Micro Devices, Inc. All rights reserved.
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
#include <migraphx/gpu/runtime_code_object.hpp>
#include <migraphx/gpu/compile_gen.hpp>
#include <migraphx/gpu/compile_hip_code_object.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/register_op.hpp>
#include <migraphx/pmr/vector.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

MIGRAPHX_REGISTER_OP(runtime_code_object);

shape runtime_code_object::compute_shape(std::vector<shape> inputs) const
{
    // std::transform(inputs.begin(), inputs.end(), inputs.begin(), [](const shape& s) {
    //     return s.normalize_standard();
    // });
    // auto einputs = expected_inputs;
    // std::transform(einputs.begin(), einputs.end(), einputs.begin(), [](const shape& s) {
    //     return s.normalize_standard();
    // });
    // if(not migraphx::equal(flatten(einputs), flatten(inputs), &shape::is_compatible))
    //     MIGRAPHX_THROW("Input shapes have changed: [" + to_string_range(einputs) + "] -> [" +
    //                    to_string_range(inputs) + "]");
    // return output;
    return inputs.front();
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

argument
runtime_code_object::compute(context& ctx, const shape&, 
                    const std::vector<argument>& args, const std::vector<module_ref>& submodule_list, std::function<std::vector<argument>(
                    module_ref&, const std::unordered_map<std::string, argument>&)>) const
{
    auto pf = gen::generate_pointwise(submodule_list.front(), "inner_pointwise", true);
    std::string lambda = "MIGRAPHX_LIFT(inner_pointwise)";
    auto kernel_name   = gen::generate_name_from_ops(submodule_list.front(), "kernel");

    std::vector<shape> inputs;
    std::transform(args.begin(), args.end(), 
                   std::back_inserter(input_shapes), [&](auto arg) { return arg->get_shape(); });

    migraphx::value v{{"lambda", lambda}, {"preamble", pf}, {"kernel", kernel_name}};

    hip_compile_options options;
    options.inputs         = flatten(inputs);
    options.output         = inputs.back();
    options.virtual_inputs = reduce_dims(normalize_permutation(options.inputs));
    options.emplace_param("-Wno-float-equal");
    auto axis              = find_fast_axis(options.virtual_inputs);
    auto vec               = vectorize::elements(ctx, axis, options.virtual_inputs);
    options.kernel_name    = kernel_name;
    auto noutputs = options.inputs.size() - inputs.size() + 1;
    auto t                 = tile::elements(options.virtual_inputs, noutputs);

    if(t.ntiles == 0)
        options.set_launch_params(
            v, compute_global_for(ctx, options.inputs.front().elements() / vec.size, 256));
    else
        options.set_launch_params(
            v, compute_global_for(ctx, t.ntiles * t.block_size, 256), t.block_size);
    auto src =
        interpolate_string(pointwise_kernel,
                            {{"kernel", kernel_name},
                            {"params", enum_params(options.inputs.size(), "void * private_p")},
                            {"args", enum_params(options.inputs.size(), "private_p")},
                            {"lambda", lambda},
                            {"transformers", make_transformer_args(t, vec)},
                            {"tiled", t.ntiles > 0 ? "true" : "false"},
                            {"noutputs", std::to_string(noutputs)},
                            {"preamble", prf}});
    
    auto cos = gpu::compile_hip_code_object_str(ctx, src, options);
    
    value::binary code_object = value::binary{cos.front()};
    kernel k{code_object, kernel_name};


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
    k.launch(ctx.get_stream().get(), options.global, options.local, kargs, start, stop);
    return args[get_output_arg(args.size())];
}

// void runtime_code_object::finalize(context&, const shape&, const std::vector<shape>&)
// {
//     assert(not code_object.empty());
//     k = kernel(code_object, symbol_name);
// }

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
