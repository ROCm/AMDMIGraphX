#include <migraphx/gpu/compile_hip.hpp>
#include <migraphx/file_buffer.hpp>
#include <migraphx/tmp_dir.hpp>
#include <migraphx/stringutils.hpp>

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

std::vector<std::vector<char>> compile_hip_src(const std::vector<src_file>& srcs, std::string params, const std::string& arch)
{
    std::vector<std::vector<char>> hsacos;
    tmp_dir td{};
    if(is_hcc_compiler())
    {
        params += " -amdgpu-target=" + arch;
    }
    else if(is_hip_clang_compiler())
    {
        if(params.find("-std=") == std::string::npos)
            params += " --std=c++17";
        params += " --cuda-gpu-arch=" + arch;
        params += " --cuda-device-only";
        params += " -c";
        params += " -O3 ";
    }

    for(const auto& src:srcs)
    {
        fs::path full_path = td.path / src.path;
        fs::path parent_path = full_path.parent_path();
        fs::create_directories(parent_path);
        write_buffer(full_path.string(), src.content.first, src.len());
        if (src.path.extension().string() == ".cpp")
            params += " " + src.path.filename().string();
    }

    params += " -Wno-unused-command-line-argument -I. ";
    params += MIGRAPHX_STRINGIZE(MIGRAPHX_HIP_COMPILER_FLAGS);

    for(auto entry:fs::directory_iterator{td.path})
    {
        if (not entry.is_regular_file())
            continue;
        auto obj_path = entry.path();
        if (obj_path.extension() != ".o")
            continue;
        if(is_hcc_compiler())
        {
            // call extract kernel
            td.execute(MIGRAPHX_STRINGIZE(MIGRAPHX_EXTRACT_KERNEL), " -i " + obj_path.string());
        }
        if(is_hip_clang_compiler())
        {
            // call clang-offload-bundler
            td.execute(MIGRAPHX_STRINGIZE(MIGRAPHX_OFFLOADBUNDLER_BIN),
                             "--type=o --targets=hip-amdgcn-amd-amdhsa-" + arch + " --inputs=" +
                                 obj_path.string() + " --outputs=" + obj_path.string() +
                                 ".hsaco --unbundle");
        }
    }

    for(auto entry:fs::directory_iterator{td.path})
    {
        if (not entry.is_regular_file())
            continue;
        auto obj_path = entry.path();
        if (obj_path.extension() != ".hsaco")
            continue;
        fs::path full_path = td.path / obj_path;
        hsacos.push_back(read_buffer(full_path.string()));
    }

    return hsacos;

}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
