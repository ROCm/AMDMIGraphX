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
    static __device__ auto apply(void* p)
    {
        return make_tensor_view(reinterpret_cast<${type}*>(p), make_shape(${lens}, ${strides}));
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

operation compile_hip_code_object(const std::string& content, hip_compile_options options)
{
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
        generate_args_hpp(options.reduced_inputs.empty() ? options.inputs : options.reduced_inputs);
    srcs.push_back(src_file{fs::path{"args.hpp"},
                            std::make_pair(args_hpp.data(), args_hpp.data() + args_hpp.size())});
    options.params += " -DMIGRAPHX_NGLOBAL=" + std::to_string(options.global);
    options.params += " -DMIGRAPHX_NLOCAL=" + std::to_string(options.local);
    options.params += " " + join_strings(compiler_warnings(), " ");
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
