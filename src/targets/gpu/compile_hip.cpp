#include <migraphx/gpu/compile_hip.hpp>
#include <migraphx/errors.hpp>
#include <migraphx/stringutils.hpp>
#include <cassert>
#include <iostream>

#if MIGRAPHX_USE_HIPRTC
#include <hip/hiprtc.h>
#include <migraphx/manage_ptr.hpp>
#include <migraphx/env.hpp>
#else
#include <migraphx/compile_src.hpp>
#include <migraphx/process.hpp>
#endif

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

#if MIGRAPHX_USE_HIPRTC

MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_TRACE_HIPRTC)

// Workaround hiprtc's broken API
void hiprtc_program_destroy(hiprtcProgram prog)
{
    hiprtcDestroyProgram(&prog);
}
using hiprtc_program_ptr = MIGRAPHX_MANAGE_PTR(hiprtcProgram, hiprtc_program_destroy);

template<class... Ts>
hiprtc_program_ptr hiprtc_program_create(Ts... xs)
{
    hiprtcProgram prog = nullptr;
    auto result = hiprtcCreateProgram(&prog, xs...);
    hiprtc_program_ptr p{prog};
    if (result != HIPRTC_SUCCESS)
        MIGRAPHX_THROW("Create program failed.");
    return p;
}

struct hiprtc_program
{
    struct string_array
    {
        std::vector<std::string> strings{};
        std::vector<const char*> c_strs{};

        string_array() {}
        string_array(const string_array&) = delete;

        std::size_t size() const
        {
            return strings.size();
        }

        const char** data()
        {
            return c_strs.data();
        }

        void push_back(std::string s)
        {
            strings.push_back(std::move(s));
            c_strs.push_back(strings.back().c_str());
        }
    };

    hiprtc_program_ptr prog = nullptr;
    string_array headers{};
    string_array include_names{};
    std::string cpp_src = "";
    std::string cpp_name = "";

    hiprtc_program(const std::vector<src_file>& srcs)
    {
        for(auto&& src:srcs)
        {
            std::string content{src.content.first, src.content.second};
            std::string path = src.path.string();
            if(src.path.extension().string() == ".cpp")
            {
                cpp_src = std::move(content);
                cpp_name = std::move(path);
            }
            else
            {
                headers.push_back(std::move(content));
                include_names.push_back(std::move(path));
            }
        }
        prog = hiprtc_program_create(cpp_src.c_str(), cpp_name.c_str(), headers.size(), headers.data(), include_names.data());
    }

    void compile(const std::vector<std::string>& options)
    {
        if(enabled(MIGRAPHX_TRACE_HIPRTC{}))
            std::cout << "hiprtc " << join_strings(options, " ") << " " << cpp_name << std::endl;
        std::vector<const char*> c_options;
        std::transform(options.begin(), options.end(), std::back_inserter(c_options), [](const std::string& s) {
            return s.c_str();
        });
        auto result = hiprtcCompileProgram(prog.get(), c_options.size(), c_options.data());
        std::cerr << log() << std::endl;
        if (result != HIPRTC_SUCCESS)
            MIGRAPHX_THROW("Failed to compile");
    }

    std::string log()
    {
        std::size_t n = 0;
        hiprtcGetProgramLogSize(prog.get(), &n);
        std::vector<char> buffer(n);
        hiprtcGetProgramLog(prog.get(), buffer.data());
        return {buffer.begin(), buffer.end()};
    }

    std::vector<char> get_code_obj()
    {
        std::size_t n = 0;
        hiprtcGetCodeSize(prog.get(), &n);
        std::vector<char> buffer(n);
        hiprtcGetCode(prog.get(), buffer.data());
        return buffer;
    }
};

std::vector<std::vector<char>>
compile_hip_src(const std::vector<src_file>& srcs, std::string params, const std::string& arch)
{
    hiprtc_program prog(srcs);
    auto options = split_string(params, ' ');
    if (std::none_of(options.begin(), options.end(), [](const std::string& s) {
        return starts_with(s, "--std=") or starts_with(s, "-std=");
    }))
        options.push_back("-std=c++17");
    options.push_back("-fno-gpu-rdc");
    options.push_back("-O3");
    options.push_back("-Wno-cuda-compat");
    options.push_back("--cuda-gpu-arch=" + arch);
    prog.compile(options);
    return {prog.get_code_obj()};
}

#else


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

#endif

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
