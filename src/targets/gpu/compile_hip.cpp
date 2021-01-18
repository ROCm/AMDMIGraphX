#include <migraphx/gpu/compile_hip.hpp>
#include <migraphx/file_buffer.hpp>
#include <migraphx/tmp_dir.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/errors.hpp>
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
    std::vector<std::vector<char>> hsacos;
    if(not is_hcc_compiler() and not is_hip_clang_compiler())
        MIGRAPHX_THROW("Unknown hip compiler: " +
                       std::string(MIGRAPHX_STRINGIZE(MIGRAPHX_HIP_COMPILER)));
    assert(not srcs.empty());
    tmp_dir td{};
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

    params += " -Wno-unused-command-line-argument -I. ";
    params += MIGRAPHX_STRINGIZE(MIGRAPHX_HIP_COMPILER_FLAGS);

    std::string output_flags{};

    for(const auto& src : srcs)
    {
        fs::path full_path   = td.path / src.path;
        fs::path parent_path = full_path.parent_path();
        fs::create_directories(parent_path);
        write_buffer(full_path.string(), src.content.first, src.len());
        if(src.path.extension().string() == ".cpp")
        {
            params += " " + src.path.filename().string();
            output_flags = " -o " + src.path.stem().string() + ".o";
        }
    }

    params += output_flags;

    td.execute(MIGRAPHX_STRINGIZE(MIGRAPHX_HIP_COMPILER), params);

    for(const auto& entry : fs::directory_iterator{td.path})
    {
        const auto& obj_path = entry.path();
        if(not fs::is_regular_file(obj_path))
            continue;
        if(obj_path.extension() != ".o")
            continue;
        if(is_hcc_compiler())
        {
            // call extract kernel
            td.execute(MIGRAPHX_STRINGIZE(MIGRAPHX_EXTRACT_KERNEL), " -i " + obj_path.string());
        }
    }

    const std::string ext = is_hcc_compiler() ? ".hsaco" : ".o";

    for(const auto& entry : fs::directory_iterator{td.path})
    {
        const auto& obj_path = entry.path();
        if(not fs::is_regular_file(obj_path))
            continue;
        if(obj_path.extension() != ext)
            continue;
        hsacos.push_back(read_buffer(obj_path.string()));
    }

    return hsacos;
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
