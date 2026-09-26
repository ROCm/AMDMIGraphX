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
#include <migraphx/gpu/compile_hip.hpp>
#include <migraphx/errors.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/env.hpp>
#include <migraphx/logger.hpp>
#include <algorithm>
#include <cassert>
#include <iostream>
#include <deque>

#ifdef MIGRAPHX_USE_HIPRTC
#include <hip/hiprtc.h>
#include <migraphx/manage_ptr.hpp>
#endif

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_GPU_DEBUG);
MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_GPU_OPTIMIZE);
MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_GPU_HIP_FLAGS);

#ifdef MIGRAPHX_USE_HIPRTC

namespace {
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

} // namespace

struct hiprtc_program
{
    struct string_array
    {
        std::deque<std::string> strings{};
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

    hiprtc_program(const std::string& src, const std::string& name = "main.cpp")
        : cpp_src(src), cpp_name(name)
    {
        create_program();
    }

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
        create_program();
    }

    void create_program()
    {
        assert(not cpp_src.empty());
        assert(not cpp_name.empty());
        assert(headers.size() == include_names.size());
        prog = hiprtc_program_create(cpp_src.c_str(),
                                     cpp_name.c_str(),
                                     headers.size(),
                                     headers.data(),
                                     include_names.data());
    }

    void compile(const std::vector<std::string>& options, bool quiet = false) const
    {
        if(enabled(MIGRAPHX_TRACE_HIPRTC{}))
            // stderr, not stdout: in migraphx-hiprtc-driver stdout carries the msgpack reply.
            std::cerr << "hiprtc " << join_strings(options, " ") << " " << cpp_name << std::endl;
        std::vector<const char*> c_options;
        std::transform(options.begin(),
                       options.end(),
                       std::back_inserter(c_options),
                       [](const std::string& s) { return s.c_str(); });
        auto result   = hiprtcCompileProgram(prog.get(), c_options.size(), c_options.data());
        auto prog_log = log();
        if(not prog_log.empty() and not quiet)
        {
            log::warn() << prog_log;
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
                                                           const std::vector<std::string>& params,
                                                           const std::string& arch,
                                                           bool quiet)
{
    hiprtc_program prog(std::move(srcs));
    auto options = params;
    options.push_back("-DMIGRAPHX_USE_HIPRTC=1");
    if(enabled(MIGRAPHX_GPU_DEBUG{}))
    {
        options.push_back("-DMIGRAPHX_DEBUG");
        options.push_back("-DROCM_DEBUG");
    }
    if(std::none_of(options.begin(), options.end(), [](const std::string& s) {
           return starts_with(s, "--std=") or starts_with(s, "-std=");
       }))
        options.push_back("-std=c++17");
    options.push_back("-fno-gpu-rdc");
    options.push_back("-O" + string_value_of(MIGRAPHX_GPU_OPTIMIZE{}, "3"));
    options.push_back("-Wno-cuda-compat");
    options.push_back("--offload-arch=" + arch);
    std::vector<std::string> extra_flags =
        split_string(string_value_of(MIGRAPHX_GPU_HIP_FLAGS{}, ""), ' ');
    options.insert(options.end(), extra_flags.begin(), extra_flags.end());

    prog.compile(options, quiet);

    auto code_obj = prog.get_code_obj();
    // Checked here so the in-process path and migraphx-hiprtc-driver both get it: hiprtc can
    // report success and still hand back nothing, and an empty buffer reaches hipModuleLoadData as
    // a null image.
    if(code_obj.empty())
        MIGRAPHX_THROW("hiprtc produced an empty code object for " + arch);
    return {std::move(code_obj)};
}

#else // MIGRAPHX_USE_HIPRTC

std::vector<std::vector<char>>
compile_hip_src_with_hiprtc(std::vector<hiprtc_src_file>,    // NOLINT
                            const std::vector<std::string>&, // NOLINT
                            const std::string&,
                            bool)
{
    MIGRAPHX_THROW("Not using hiprtc");
}

#endif // MIGRAPHX_USE_HIPRTC

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
