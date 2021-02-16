#include <migraphx/gpu/compile_hip_code_object.hpp>
#include <migraphx/gpu/compile_hip.hpp>
#include <migraphx/gpu/code_object_op.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/context.hpp>
#include <migraphx_kernels.hpp>
#include <migraphx/rank.hpp>
#include <migraphx/stringutils.hpp>
#include <hip/hip_runtime_api.h>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

std::string get_arch_name(rank<0>, const hipDeviceProp_t& props)
{
    return "gfx" + std::to_string(props.gcnArch);
}

auto get_arch_name(rank<1>, const hipDeviceProp_t& props)
    -> decltype(std::string(props.gcnArchName))
{
    return std::string(props.gcnArchName);
}

int get_device_id()
{
    int device;
    auto status = hipGetDevice(&device);
    if(status != hipSuccess)
        MIGRAPHX_THROW("No device");
    return device;
}

std::string get_device_name()
{
    hipDeviceProp_t props{};
    auto status = hipGetDeviceProperties(&props, get_device_id());
    if(status != hipSuccess)
        MIGRAPHX_THROW("Failed to get device properties");
    return get_arch_name(rank<1>{}, props);
}

template <class T>
std::string generate_index_ints(const std::vector<T>& v)
{
    return "index_ints<" + to_string_range(v) + ">{}";
}

std::string generate_cpp_type(shape::type_t t)
{
    switch(t)
    {
#define MIGRAPHX_GPU_GENERATE_TYPE_STRING(x, t) \
    case shape::x: return #t;
        MIGRAPHX_SHAPE_VISIT_TYPES(MIGRAPHX_GPU_GENERATE_TYPE_STRING)
    }
    MIGRAPHX_THROW("Invalid type");
}

std::string generate_make_shape(const shape& s)
{
    return "make_shape(" + generate_index_ints(s.lens()) + ", " + generate_index_ints(s.strides()) +
           ")";
}

std::string generate_make_tensor(std::size_t n, const shape& s)
{
    std::stringstream ss;
    ss << "__device__ auto make_tensor(arg<" << n << ">, void* p)\n";
    ss << "{\n";
    ss << "return make_tensor_view(reinterpret_cast<" << generate_cpp_type(s.type()) << "*>(p), ";
    ss << generate_make_shape(s) << ");\n";
    ss << "}\n";
    return ss.str();
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
    auto args_hpp = generate_args_hpp(options.inputs);
    srcs.push_back(src_file{fs::path{"args.hpp"},
                            std::make_pair(args_hpp.data(), args_hpp.data() + args_hpp.size())});
    options.params += " -I.";
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
