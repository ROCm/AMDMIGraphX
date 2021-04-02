#include <migraphx/gpu/compile_hip.hpp>
#include <migraphx/errors.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/compile_src.hpp>
#include <migraphx/process.hpp>
#include <cassert>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

bool is_hcc_compiler()
{
    static const auto result = ends_with(MIGRAPHX_STRINGIZE(MIGRAPHX_HIP_COMPILER), "hcc");
    return result;
}

bool is_hip_clang_compiler()
{
    static const auto result = ends_with(MIGRAPHX_STRINGIZE(MIGRAPHX_HIP_COMPILER), "clang++");
    return result;
}

std::vector<std::vector<char>>
compile_hip_src(const std::vector<src_file>& srcs, std::string params, const std::string& arch)
{
    assert(not srcs.empty());
    if(not is_hcc_compiler() and not is_hip_clang_compiler())
        MIGRAPHX_THROW("Unknown hip compiler: " +
                       std::string(MIGRAPHX_STRINGIZE(MIGRAPHX_HIP_COMPILER)));

    if(params.find("-std=") == std::string::npos)
        params += " --std=c++17";
    params += " -fno-gpu-rdc";
    params += " -c";
    if(is_hcc_compiler())
    {
        params += " -amdgpu-target=" + arch;
    }
    else if(is_hip_clang_compiler())
    {
        params += " --cuda-gpu-arch=" + arch;
        params += " --cuda-device-only";
        params += " -O3 ";
    }

    params += " -Wno-unused-command-line-argument -Wno-cuda-compat ";
    params += MIGRAPHX_STRINGIZE(MIGRAPHX_HIP_COMPILER_FLAGS);

    src_compiler compiler;
    compiler.flags    = params;
    compiler.compiler = MIGRAPHX_STRINGIZE(MIGRAPHX_HIP_COMPILER);

    if(is_hcc_compiler())
        compiler.process = [&](const fs::path& obj_path) -> fs::path {
            process{MIGRAPHX_STRINGIZE(MIGRAPHX_EXTRACT_KERNEL) + std::string{" -i "} +
                    obj_path.string()}
                .cwd(obj_path.parent_path());
            for(const auto& entry : fs::directory_iterator{obj_path.parent_path()})
            {
                const auto& hsaco_path = entry.path();
                if(not fs::is_regular_file(hsaco_path))
                    continue;
                if(hsaco_path.extension() != ".hsaco")
                    continue;
                return hsaco_path;
            }
            MIGRAPHX_THROW("Missing hsaco");
        };

    return {compiler.compile(srcs)};
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
