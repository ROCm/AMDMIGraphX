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
#include <migraphx/ranges.hpp>
#include <migraphx/env.hpp>
#include <migraphx/fileutils.hpp>
#include <migraphx/logger.hpp>
#include <cassert>
#include <iostream>
#include <deque>
#include <string_view>

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
MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_GPU_HIP_FLAGS);

std::vector<std::string> compile_hip_options(const std::vector<std::string>& params,
                                             const std::string& arch)
{
    auto options = params;
    if(enabled(MIGRAPHX_GPU_DEBUG{}))
        options.push_back("-DMIGRAPHX_DEBUG");
    if(std::none_of(options.begin(), options.end(), [](const std::string& s) {
           return starts_with(s, "--std=") or starts_with(s, "-std=");
       }))
        options.push_back("-std=c++17");
    if(enabled(MIGRAPHX_GPU_DEBUG_SYM{}))
        options.emplace_back("-g");
    options.push_back("-fno-gpu-rdc");
    options.push_back("-O" + string_value_of(MIGRAPHX_GPU_OPTIMIZE{}, "3"));
    options.push_back("-Wno-cuda-compat");
    options.push_back("--offload-arch=" + arch);
    std::vector<std::string> extra_flags =
        split_string(string_value_of(MIGRAPHX_GPU_HIP_FLAGS{}, ""), ' ');
    options.insert(options.end(), extra_flags.begin(), extra_flags.end());

    return options;
}

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
            std::cout << "hiprtc " << join_strings(options, " ") << " " << cpp_name << std::endl;
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
    auto options = compile_hip_options(params, arch);
    options.push_back("-DMIGRAPHX_USE_HIPRTC=1");

    prog.compile(options, quiet);

    return {prog.get_code_obj()};
}

std::vector<std::vector<char>> compile_hip_src(const std::vector<src_file>& srcs,
                                               const std::vector<std::string>& params,
                                               const std::string& arch,
                                               bool quiet)
{
    std::vector<hiprtc_src_file> hsrcs{srcs.begin(), srcs.end()};
    if(enabled(MIGRAPHX_GPU_DUMP_SRC{}))
    {
        for(const auto& src : srcs)
        {
            if(src.path.extension() != ".cpp")
                continue;
            std::cout << std::string(src.content) << std::endl;
        }
    }

    auto fname  = make_executable_filename("migraphx-hiprtc-driver");
    auto p      = dynamic_loader::path(&compile_hip_src_with_hiprtc);
    auto driver = p.parent_path() / fname;

    bool found = fs::exists(driver);
    if(not found)
    {
        driver = p.parent_path().parent_path() / "bin" / fname;
        found  = fs::exists(driver);
    }

    if(found)
    {
        value v;
        v["srcs"]   = to_value(hsrcs);
        v["params"] = to_value(params);
        v["arch"]   = to_value(arch);
        v["quiet"]  = quiet;

        tmp_dir td{};
        auto out = td.path / "output";

        process(driver, {quote_string(out.string())}).write([&](auto writer) {
            to_msgpack(v, std::move(writer));
        });
        if(fs::exists(out))
            return {read_buffer(out)};
        MIGRAPHX_THROW("hiprtc compilation failed!");
    }
    return compile_hip_src_with_hiprtc(std::move(hsrcs), params, arch, quiet);
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

static bool is_hip_clang_compiler()
{
    static const auto result = fs::path{MIGRAPHX_HIP_COMPILER}.stem() == "clang++";
    return result;
}

#ifdef MIGRAPHX_HIP_COMPILER_LAUNCHER

static bool has_compiler_launcher()
{
    static const auto result = fs::exists(MIGRAPHX_HIP_COMPILER_LAUNCHER);
    return result;
}

#endif

static src_compiler assemble(src_compiler compiler)
{
    compiler.out_ext = ".S";
    std::replace(compiler.flags.begin(), compiler.flags.end(), "-c", "-S");
    return compiler;
}

std::vector<std::vector<char>> compile_hip_src(const std::vector<src_file>& srcs,
                                               const std::vector<std::string>& params,
                                               const std::string& arch,
                                               bool)
{
    assert(not srcs.empty());

    if(not is_hip_clang_compiler())
        MIGRAPHX_THROW("Unknown hip compiler: " MIGRAPHX_HIP_COMPILER);

    src_compiler compiler;
    compiler.flags    = compile_hip_options(params, arch);
    compiler.compiler = MIGRAPHX_HIP_COMPILER;
#ifdef MIGRAPHX_HIP_COMPILER_LAUNCHER
    if(has_compiler_launcher())
        compiler.launcher = MIGRAPHX_HIP_COMPILER_LAUNCHER;
#endif

    compiler.flags.emplace_back("-c");
    compiler.flags.emplace_back("--cuda-device-only");
    compiler.flags.emplace_back("-Wno-unused-command-line-argument");
    compiler.flags.emplace_back(MIGRAPHX_HIP_COMPILER_FLAGS);

    if(enabled(MIGRAPHX_GPU_DUMP_SRC{}))
    {
        for(const auto& src : srcs)
        {
            if(src.path.extension() != ".cpp")
                continue;
            std::cout << std::string(src.content) << std::endl;
        }
    }

    if(enabled(MIGRAPHX_GPU_DUMP_ASM{}))
    {

        std::cout << assemble(compiler).compile(srcs).data() << std::endl;
    }

    return {compiler.compile(srcs)};
}

#endif // MIGRAPHX_USE_HIPRTC

// Markers around the version string, chosen so it can be found in the compiled output without
// having to understand the object format.
static constexpr std::string_view version_prefix = "<<<migraphx-compiler-version:";
static constexpr std::string_view version_suffix = ">>>";

// The probe is compiled but never run, so the architecture only has to be one the compiler
// accepts. Keeping it fixed means the answer does not depend on the device present.
static const char* const version_probe_arch = "gfx950";

// Probe that makes the device compiler embed its own version string in the object. It need not
// be the compiler this library was built with, so its version is only known at runtime.
static const char* const version_probe = R"__migraphx__(
#define MIGRAPHX_STRINGIZE_(x) #x
#define MIGRAPHX_STRINGIZE(x) MIGRAPHX_STRINGIZE_(x)

#ifdef __clang_major__
#define MIGRAPHX_COMPILER_MAJOR MIGRAPHX_STRINGIZE(__clang_major__)
#define MIGRAPHX_COMPILER_MINOR MIGRAPHX_STRINGIZE(__clang_minor__)
#define MIGRAPHX_COMPILER_VERSION __clang_version__
#else
#define MIGRAPHX_COMPILER_MAJOR "0"
#define MIGRAPHX_COMPILER_MINOR "0"
#define MIGRAPHX_COMPILER_VERSION __VERSION__
#endif

extern "C" __attribute__((used)) __device__ const char migraphx_compiler_version[] =
    "${prefix}" MIGRAPHX_COMPILER_MAJOR "|" MIGRAPHX_COMPILER_MINOR "|"
    MIGRAPHX_COMPILER_VERSION "${suffix}";

extern "C" __global__ void migraphx_version_probe(char* p)
{
    p[0] = migraphx_compiler_version[0];
}
)__migraphx__";

static hip_compiler_info parse_version(const std::vector<char>& obj)
{
    auto begin = std::search(obj.begin(), obj.end(), version_prefix.begin(), version_prefix.end());
    if(begin == obj.end())
        return {};
    begin += version_prefix.size();
    auto end = std::search(begin, obj.end(), version_suffix.begin(), version_suffix.end());
    if(end == obj.end())
        return {};

    auto fields = split_string(std::string{begin, end}, '|');
    if(fields.size() != 3)
        return {};
    return {.major = fields[0], .minor = fields[1], .version = fields[2]};
}

const hip_compiler_info& hip_compiler_version()
{
    static const hip_compiler_info info = [] {
        try
        {
            // Interpolated so the markers parse_version searches for cannot drift from the ones
            // the probe embeds.
            auto probe = interpolate_string(
                version_probe,
                {{"prefix", std::string{version_prefix}}, {"suffix", std::string{version_suffix}}});
            auto cos = compile_hip_src({src_file{"main.cpp", probe}},
                                       {"-std=c++17"},
                                       version_probe_arch,
                                       /* quiet */ true);
            if(not cos.empty())
                return parse_version(cos.front());
        }
        catch(const std::exception& e)
        {
            // Callers treat an empty version as "cannot tell compilers apart" and stop sharing
            // results between runs, which is slower but never wrong.
            log::warn() << "Cannot identify the hip compiler: " << e.what();
        }
        return hip_compiler_info{};
    }();
    return info;
}

bool hip_can_compile(const std::string& src, const std::vector<std::string>& flags)
{
    src_file input{"main.cpp", src};
    std::vector<src_file> srcs = {input};

    try
    {
        std::string arch = "gfx900";
        compile_hip_src(srcs, flags, arch, true);
        return true;
    }
    catch(...)
    {
        return false;
    }
}

bool hip_has_flags(const std::vector<std::string>& flags) { return hip_can_compile(" ", flags); }

std::string enum_params(std::size_t count, std::string param)
{
    std::vector<std::string> items(count);
    transform(range(count), items.begin(), [&](auto i) { return param + std::to_string(i); });
    return join_strings(items, ",");
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
