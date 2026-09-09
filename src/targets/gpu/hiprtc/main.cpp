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
#include <migraphx/serialize.hpp>
#include <migraphx/value.hpp>
#include <migraphx/msgpack.hpp>
#include <migraphx/errors.hpp>
#include <migraphx/ranges.hpp>
#include <array>
#include <iostream>
#include <cstdio>

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif

// CRLF translation would corrupt both the msgpack request and the code object. Nothing restores the
// old mode: each stream is used once and the process exits straight after.
// https://learn.microsoft.com/en-us/cpp/c-runtime-library/reference/setmode?view=msvc-170
static void set_binary_mode([[maybe_unused]] FILE* stream)
{
#ifdef _WIN32
    if(_setmode(_fileno(stream), _O_BINARY) == -1)
        MIGRAPHX_THROW("Failed to set stream to binary mode");
#endif
}

static std::vector<char> read_stdin()
{
    set_binary_mode(stdin);
    std::vector<char> result;
    std::array<char, 1024> buffer{};
    std::size_t len = 0;
    while((len = std::fread(buffer.data(), 1, buffer.size(), stdin)) > 0)
    {
        if(std::ferror(stdin) != 0 and std::feof(stdin) == 0)
            MIGRAPHX_THROW("Failed reading request from stdin");

        result.insert(result.end(), buffer.data(), buffer.data() + len);
    }
    return result;
}

// stdout is a binary channel: nothing else in this process may write to it, or the parent reads a
// corrupted code object.
static void write_stdout(const std::vector<char>& buffer)
{
    set_binary_mode(stdout);
    if(std::fwrite(buffer.data(), 1, buffer.size(), stdout) != buffer.size())
        MIGRAPHX_THROW("Failed writing code object to stdout");
    if(std::fflush(stdout) != 0)
        MIGRAPHX_THROW("Failed flushing stdout");
}

int main(int argc, char const* argv[])
{
    // The compile request arrives on stdin and the code object leaves on stdout, so no arguments
    // are expected in normal operation.
    if(argc > 1)
    {
        // stderr, not stdout: stdout is reserved for the code object.
        std::cerr << "USAGE:" << std::endl;
        std::cerr << "    ";
        std::cerr << "Used internally by migraphx to compile hip programs out-of-process."
                  << std::endl;
        std::exit(migraphx::contains({"-h", "--help", "-v", "--version"}, std::string(argv[1])) ? 0
                                                                                                : 1);
    }
    bool quiet = false;
    try
    {
        auto v = migraphx::from_msgpack(read_stdin());
        quiet  = v.at("quiet").to<bool>();
        std::vector<migraphx::gpu::hiprtc_src_file> srcs;
        migraphx::from_value(v.at("srcs"), srcs);
        auto out =
            migraphx::gpu::compile_hip_src_with_hiprtc(std::move(srcs),
                                                       v.at("params").to_vector<std::string>(),
                                                       v.at("arch").to<std::string>(),
                                                       quiet);
        if(out.empty())
            MIGRAPHX_THROW("hiprtc produced no code object");
        write_stdout(out.front());
    }
    catch(const std::exception& err)
    {
        if(not quiet)
            std::cerr << err.what() << std::endl;
        // Exit status is the parent's only failure signal.
        return 1;
    }
    return 0;
}
