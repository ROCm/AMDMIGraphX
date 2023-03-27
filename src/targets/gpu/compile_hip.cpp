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
#include <migraphx/gpu/compile_hip.hpp>
#include <migraphx/errors.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/env.hpp>
#include <cassert>
#include <iostream>

#ifdef MIGRAPHX_USE_HIPRTC
#include <hip/hiprtc.h>
#include <migraphx/manage_ptr.hpp>
#include <migraphx/value.hpp>
#include <migraphx/tmp_dir.hpp>
#include <migraphx/dynamic_loader.hpp>
#include <migraphx/process.hpp>
#include <migraphx/msgpack.hpp>
#include <migraphx/serialize.hpp>
#include <migraphx/file_buffer.hpp>
#else
#include <migraphx/compile_src.hpp>
#include <migraphx/process.hpp>
#endif

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_GPU_DEBUG);
MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_GPU_DEBUG_SYM);
MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_GPU_OPTIMIZE);
MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_GPU_DUMP_ASM);
MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_GPU_DUMP_SRC);

#ifdef MIGRAPHX_USE_HIPRTC

MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_TRACE_HIPRTC);
MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_ENABLE_HIPRTC_WORKAROUNDS);

std::string hiprtc_error(hiprtcResult err, const std::string& msg)
{
    return "hiprtc: " + (hiprtcGetErrorString(err) + (": " + msg));
}

void hiprtc_check_error(hiprtcResult err, const std::string& msg, const std::string& ctx)
{
    if(err != HIPRTC_SUCCESS)
        throw make_exception(ctx, hiprtc_error(err, msg));
}

// NOLINTNEXTLINE
#define MIGRAPHX_HIPRTC(...) \
    hiprtc_check_error(__VA_ARGS__, #__VA_ARGS__, MIGRAPHX_MAKE_SOURCE_CTX())

#define MIGRAPHX_HIPRTC_THROW(error, msg) MIGRAPHX_THROW(hiprtc_error(error, msg))

// Workaround hiprtc's broken API
void hiprtc_program_destroy(hiprtcProgram prog) { hiprtcDestroyProgram(&prog); }
using hiprtc_program_ptr = MIGRAPHX_MANAGE_PTR(hiprtcProgram, hiprtc_program_destroy);

template <class... Ts>
hiprtc_program_ptr hiprtc_program_create(Ts... xs)
{
    hiprtcProgram prog = nullptr;
    auto result        = hiprtcCreateProgram(&prog, xs...);
    hiprtc_program_ptr p{prog};
    if(result != HIPRTC_SUCCESS)
        MIGRAPHX_HIPRTC_THROW(result, "Create program failed.");
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

        std::size_t size() const { return strings.size(); }

        const char** data() { return c_strs.data(); }

        void push_back(std::string s)
        {
            strings.push_back(std::move(s));
            c_strs.push_back(strings.back().c_str());
        }
    };

    hiprtc_program_ptr prog = nullptr;
    string_array headers{};
    string_array include_names{};
    std::string cpp_src  = "";
    std::string cpp_name = "";

    hiprtc_program(std::vector<hiprtc_src_file> srcs)
    {
        for(auto&& src : srcs)
        {
            if(ends_with(src.path, ".cpp"))
            {
                cpp_src  = std::move(src.content);
                cpp_name = std::move(src.path);
            }
            else
            {
                headers.push_back(std::move(src.content));
                include_names.push_back(std::move(src.path));
            }
        }
        prog = hiprtc_program_create(cpp_src.c_str(),
                                     cpp_name.c_str(),
                                     headers.size(),
                                     headers.data(),
                                     include_names.data());
    }

    void compile(const std::vector<std::string>& options) const
    {
        if(enabled(MIGRAPHX_TRACE_HIPRTC{}))
            std::cout << "hiprtc " << join_strings(options, " ") << " " << cpp_name << std::endl;
        std::vector<const char*> c_options;
        std::transform(options.begin(),
                       options.end(),
                       std::back_inserter(c_options),
                       [](const std::string& s) { return s.c_str(); });
        auto result   = hiprtcCompileProgram(prog.get(), c_options.size(), c_options.data());
        auto prog_log = log();
        if(not prog_log.empty())
        {
            std::cerr << prog_log << std::endl;
        }
        if(result != HIPRTC_SUCCESS)
            MIGRAPHX_HIPRTC_THROW(result, "Compilation failed.");
    }

    std::string log() const
    {
        std::size_t n = 0;
        MIGRAPHX_HIPRTC(hiprtcGetProgramLogSize(prog.get(), &n));
        if(n == 0)
            return {};
        std::string buffer(n, '\0');
        MIGRAPHX_HIPRTC(hiprtcGetProgramLog(prog.get(), buffer.data()));
        assert(buffer.back() != 0);
        return buffer;
    }

    std::vector<char> get_code_obj() const
    {
        std::size_t n = 0;
        MIGRAPHX_HIPRTC(hiprtcGetCodeSize(prog.get(), &n));
        std::vector<char> buffer(n);
        MIGRAPHX_HIPRTC(hiprtcGetCode(prog.get(), buffer.data()));
        return buffer;
    }
};

std::vector<std::vector<char>> compile_hip_src_with_hiprtc(std::vector<hiprtc_src_file> srcs,
                                                           std::string params,
                                                           const std::string& arch)
{
    hiprtc_program prog(std::move(srcs));
    auto options = split_string(params, ' ');
    options.push_back("-DMIGRAPHX_USE_HIPRTC=1");
    // remove following three compilation flags for HIPRTC once fixes from hipRTC are available in
    if(enabled(MIGRAPHX_ENABLE_HIPRTC_WORKAROUNDS{}))
    {
        options.push_back("-DMIGRAPHX_HAS_DPP=0");
        options.push_back("-DMIGRAPHX_ENABLE_HIPRTC_WORKAROUNDS=1");
        options.push_back("-Wno-reserved-identifier");
        options.push_back("-Wno-gnu-line-marker");
        options.push_back("-Wno-old-style-cast");
    }

    if(enabled(MIGRAPHX_GPU_DEBUG{}))
        options.push_back("-DMIGRAPHX_DEBUG");
    if(std::none_of(options.begin(), options.end(), [](const std::string& s) {
           return starts_with(s, "--std=") or starts_with(s, "-std=");
       }))
        options.push_back("-std=c++17");
    options.push_back("-fno-gpu-rdc");
    options.push_back("-O" + string_value_of(MIGRAPHX_GPU_OPTIMIZE{}, "3"));
    options.push_back("-Wno-cuda-compat");
    options.push_back("--offload-arch=" + arch);
    prog.compile(options);
    return {prog.get_code_obj()};
}

std::vector<std::vector<char>>
compile_hip_src(const std::vector<src_file>& srcs, std::string params, const std::string& arch)
{
    std::vector<hiprtc_src_file> hsrcs{srcs.begin(), srcs.end()};
    auto p      = dynamic_loader::path(&compile_hip_src_with_hiprtc);
    auto driver = p.parent_path().parent_path() / "bin" / "migraphx-hiprtc-driver";

    if(fs::exists(driver))
    {
        value v;
        v["srcs"]   = to_value(hsrcs);
        v["params"] = to_value(params);
        v["arch"]   = to_value(arch);

        tmp_dir td{};
        auto out = td.path / "output";

        process(driver.string() + " " + out.string()).write([&](auto writer) {
            to_msgpack(v, writer);
        });
        if(not fs::exists(out))
            return {};
        return {read_buffer(out.string())};
    }
    return compile_hip_src_with_hiprtc(std::move(hsrcs), std::move(params), arch);
}

#else // MIGRAPHX_USE_HIPRTC

std::vector<std::vector<char>> compile_hip_src_with_hiprtc(std::vector<hiprtc_src_file>,
                                                           std::string,
                                                           const std::string&)
{
    MIGRAPHX_THROW("Not using hiprtc");
}

bool is_hip_clang_compiler()
{
    static const auto result = ends_with(MIGRAPHX_STRINGIZE(MIGRAPHX_HIP_COMPILER), "clang++");
    return result;
}

bool has_compiler_launcher()
{
    static const auto result = fs::exists(MIGRAPHX_STRINGIZE(MIGRAPHX_HIP_COMPILER_LAUNCHER));
    return result;
}

src_compiler assemble(src_compiler compiler)
{
    compiler.out_ext = ".S";
    compiler.flags   = replace_string(compiler.flags, " -c", " -S");
    return compiler;
}

std::vector<std::vector<char>>
compile_hip_src(const std::vector<src_file>& srcs, std::string params, const std::string& arch)
{
    assert(not srcs.empty());
    if(not is_hip_clang_compiler())
        MIGRAPHX_THROW("Unknown hip compiler: " +
                       std::string(MIGRAPHX_STRINGIZE(MIGRAPHX_HIP_COMPILER)));

    if(params.find("-std=") == std::string::npos)
        params += " --std=c++17";
    params += " -fno-gpu-rdc";
    if(enabled(MIGRAPHX_GPU_DEBUG_SYM{}))
        params += " -g";
    params += " -c";
    params += " --offload-arch=" + arch;
    params += " --cuda-device-only";
    params += " -O" + string_value_of(MIGRAPHX_GPU_OPTIMIZE{}, "3") + " ";

    if(enabled(MIGRAPHX_GPU_DEBUG{}))
        params += " -DMIGRAPHX_DEBUG";

    params += " -Wno-unused-command-line-argument -Wno-cuda-compat ";
    params += MIGRAPHX_STRINGIZE(MIGRAPHX_HIP_COMPILER_FLAGS);

    src_compiler compiler;
    compiler.flags    = params;
    compiler.compiler = MIGRAPHX_STRINGIZE(MIGRAPHX_HIP_COMPILER);
#ifdef MIGRAPHX_HIP_COMPILER_LAUNCHER
    if(has_compiler_launcher())
        compiler.launcher = MIGRAPHX_STRINGIZE(MIGRAPHX_HIP_COMPILER_LAUNCHER);
#endif
    if(enabled(MIGRAPHX_GPU_DUMP_SRC{}))
    {
        for(const auto& src : srcs)
        {
            if(src.path.extension() != ".cpp")
                continue;
            std::cout << std::string(src.content.first, src.len()) << std::endl;
        }
    }

    if(enabled(MIGRAPHX_GPU_DUMP_ASM{}))
    {

        std::cout << assemble(compiler).compile(srcs).data() << std::endl;
    }

    return {compiler.compile(srcs)};
}

#endif // MIGRAPHX_USE_HIPRTC

std::string enum_params(std::size_t count, std::string param)
{
    std::vector<std::string> items(count);
    transform(range(count), items.begin(), [&](auto i) { return param + std::to_string(i); });
    return join_strings(items, ",");
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
