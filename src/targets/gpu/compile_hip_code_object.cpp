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
#include <migraphx/gpu/compile_hip_code_object.hpp>
#include <migraphx/gpu/compile_hip.hpp>
#include <migraphx/gpu/code_object_op.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/gpu/device_name.hpp>
#include <migraphx/context.hpp>
#include <migraphx_kernels.hpp>
#include <migraphx/stringutils.hpp>
#include <hip/hip_runtime_api.h>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

template <class T>
std::string generate_index_ints(const std::vector<T>& v)
{
    return "index_ints<" + to_string_range(v) + ">{}";
}

std::string generate_make_shape(const shape& s)
{
    return "make_shape(" + generate_index_ints(s.lens()) + ", " + generate_index_ints(s.strides()) +
           ")";
}

static const char* const make_tensor_template = R"__migraphx__(
template<>
struct make_tensor<${n}>
{
    static __device__ auto apply(void* __restrict__ p)
    {
        return make_tensor_view(reinterpret_cast<${type}* __restrict__>(p), make_shape(${lens}, ${strides}));
    }
};
)__migraphx__";

std::string generate_make_tensor(std::size_t n, const shape& s)
{
    return interpolate_string(make_tensor_template,
                              {{"n", std::to_string(n)},
                               {"type", shape::cpp_type(s.type())},
                               {"lens", generate_index_ints(s.lens())},
                               {"strides", generate_index_ints(s.strides())}});
}

std::string generate_args_hpp(const std::vector<shape>& inputs)
{
    std::string inner;
    for(std::size_t i = 0; i < inputs.size(); i++)
    {
        inner += generate_make_tensor(i, inputs[i]);
    }
    const std::string args_hpp = R"__migraphx__(
#ifndef MIGRAPHX_GUARD_AUTO_ARGS_HPP
#define MIGRAPHX_GUARD_AUTO_ARGS_HPP

#include <migraphx/kernels/args.hpp>
#include <migraphx/kernels/tensor_view.hpp>

namespace migraphx {

__content__

} // namespace migraphx
#endif
)__migraphx__";
    return replace_string(args_hpp, "__content__", inner);
}

const std::vector<std::string>& compiler_warnings()
{
    static std::vector<std::string> warnings = {"-Weverything",
                                                "-Wno-c++98-compat",
                                                "-Wno-c++98-compat-pedantic",
                                                "-Wno-conversion",
                                                "-Wno-double-promotion",
                                                "-Wno-exit-time-destructors",
                                                "-Wno-extra-semi",
                                                "-Wno-extra-semi-stmt",
                                                "-Wno-float-conversion",
                                                "-Wno-gnu-anonymous-struct",
                                                "-Wno-gnu-zero-variadic-macro-arguments",
                                                "-Wno-missing-prototypes",
                                                "-Wno-nested-anon-types",
                                                "-Wno-padded",
                                                "-Wno-shorten-64-to-32",
                                                "-Wno-sign-conversion",
                                                "-Wno-sign-compare",
                                                "-Wno-unused-command-line-argument",
                                                "-Wno-weak-vtables",
                                                "-Wno-c99-extensions"};
    return warnings;
}

void hip_compile_options::set_launch_params(
    const value& v,
    const std::function<std::size_t(std::size_t local)>& compute_global,
    std::size_t default_local)
{
    local = v.get("local", default_local);
    if(v.contains("global"))
        global = v.at("global").to<std::size_t>();
    else
        global = compute_global(local);
}

std::function<std::size_t(std::size_t local)>
compute_global_for(context& ctx, std::size_t n, std::size_t over)
{
    assert(over > 0);
    std::size_t max_global = ctx.get_current_device().get_cu_count() *
                             ctx.get_current_device().get_max_workitems_per_cu();
    return [n, over, max_global](std::size_t local) {
        std::size_t groups     = (n + local - 1) / local;
        std::size_t max_blocks = max_global / local;
        std::size_t nglobal    = std::min(max_blocks * over, groups) * local;
        return std::min(nglobal, n);
    };
}

std::size_t compute_block_size(std::size_t n, std::size_t max_block_size)
{
    const std::size_t min_block_size = 64;
    auto block_size                  = (((n - 1) / min_block_size + 1)) * min_block_size;
    return std::min(std::max(min_block_size, block_size), max_block_size);
}

operation compile_hip_code_object(const std::string& content, hip_compile_options options)
{
    assert(options.global > 0);
    assert(options.local > 0);
    assert(not options.inputs.empty());
    assert(options.inputs.size() == options.virtual_inputs.size() or
           options.virtual_inputs.empty());
    std::vector<src_file> srcs;
    std::transform(migraphx_kernels().begin(),
                   migraphx_kernels().end(),
                   std::back_inserter(srcs),
                   [](auto&& p) {
                       auto&& name = p.first;
                       auto&& c    = p.second;
                       auto path   = fs::path{"migraphx"} / "kernels" / name;
                       return src_file{path, c};
                   });
    srcs.push_back(src_file{fs::path{"main.cpp"},
                            std::make_pair(content.data(), content.data() + content.size())});
    auto args_hpp =
        generate_args_hpp(options.virtual_inputs.empty() ? options.inputs : options.virtual_inputs);
    srcs.push_back(src_file{fs::path{"args.hpp"},
                            std::make_pair(args_hpp.data(), args_hpp.data() + args_hpp.size())});
    options.params += " -DMIGRAPHX_NGLOBAL=" + std::to_string(options.global);
    options.params += " -DMIGRAPHX_NLOCAL=" + std::to_string(options.local);
    options.params += " " + join_strings(compiler_warnings(), " ");
    options.params += " -ftemplate-backtrace-limit=0";
    options.params += " -Werror";
    auto cos = compile_hip_src(srcs, std::move(options.params), get_device_name());
    if(cos.size() != 1)
        MIGRAPHX_THROW("No code object");
    return code_object_op{value::binary{cos.front()},
                          options.kernel_name,
                          options.global,
                          options.local,
                          options.inputs,
                          options.output};
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
