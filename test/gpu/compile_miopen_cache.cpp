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
#include <migraphx/context.hpp>
#include <migraphx/file_buffer.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/gpu/convolution.hpp>
#include <migraphx/op/convolution.hpp>
#include <migraphx/tmp_dir.hpp>
#include <test.hpp>

#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string_view>

#ifndef _WIN32
#include <fcntl.h>
#include <unistd.h>
#endif

namespace {

std::size_t count_occurrences(std::string_view text, std::string_view needle)
{
    if(needle.empty())
        return 0;

    std::size_t count = 0;
    std::size_t pos   = 0;
    while((pos = text.find(needle, pos)) != std::string_view::npos)
    {
        ++count;
        pos += needle.size();
    }
    return count;
}

struct scoped_env_var
{
    scoped_env_var(const char* env_name, const char* env_value) : name(env_name)
    {
        if(const char* current = std::getenv(env_name))
        {
            had_previous   = true;
            previous_value = current;
        }
        setenv(env_name, env_value, 1);
    }

    ~scoped_env_var()
    {
        if(had_previous)
            setenv(name.c_str(), previous_value.c_str(), 1);
        else
            unsetenv(name.c_str());
    }

    std::string name           = {};
    std::string previous_value = {};
    bool had_previous          = false;
};

#ifndef _WIN32
struct scoped_output_capture
{
    explicit scoped_output_capture(const migraphx::fs::path& path)
    {
        fd = ::open(path.c_str(), O_CREAT | O_TRUNC | O_RDWR, 0600);
        if(fd < 0)
            throw std::runtime_error("Failed to open capture file");

        saved_stdout = ::dup(STDOUT_FILENO);
        saved_stderr = ::dup(STDERR_FILENO);
        if(saved_stdout < 0 or saved_stderr < 0)
            throw std::runtime_error("Failed to duplicate stdout/stderr");

        std::fflush(stdout);
        std::fflush(stderr);
        if(::dup2(fd, STDOUT_FILENO) < 0 or ::dup2(fd, STDERR_FILENO) < 0)
            throw std::runtime_error("Failed to redirect stdout/stderr");
    }

    scoped_output_capture(const scoped_output_capture&) = delete;
    scoped_output_capture& operator=(const scoped_output_capture&) = delete;

    ~scoped_output_capture() { restore(); }

    void restore()
    {
        if(restored)
            return;

        std::fflush(stdout);
        std::fflush(stderr);
        if(saved_stdout >= 0)
            ::dup2(saved_stdout, STDOUT_FILENO);
        if(saved_stderr >= 0)
            ::dup2(saved_stderr, STDERR_FILENO);
        if(saved_stdout >= 0)
            ::close(saved_stdout);
        if(saved_stderr >= 0)
            ::close(saved_stderr);
        if(fd >= 0)
            ::close(fd);
        restored = true;
    }

    int fd           = -1;
    int saved_stdout = -1;
    int saved_stderr = -1;
    bool restored    = false;
};
#endif

} // namespace

TEST_CASE(miopen_convolution_compile_reuses_cached_solution)
{
#if !MIGRAPHX_USE_MIOPEN || defined(_WIN32)
    EXPECT(true);
#else
    scoped_env_var logging_env{"MIOPEN_ENABLE_LOGGING", "1"};
    scoped_env_var logging_cmd_env{"MIOPEN_ENABLE_LOGGING_CMD", "1"};
    migraphx::tmp_dir td{"miopen-compile-cache"};
    const auto capture_path = td.path / "trace.log";

    migraphx::gpu::context gpu_ctx;
    migraphx::context ctx{gpu_ctx};

    migraphx::op::convolution conv{};
    conv.padding = {1, 1};

    const migraphx::shape x{migraphx::shape::float_type, {1, 64, 16, 16}};
    const migraphx::shape w{migraphx::shape::float_type, {64, 64, 3, 3}};
    const migraphx::shape y = migraphx::compute_shape(conv, {x, w});
    const std::vector<migraphx::shape> inputs = {x, w, y};

    migraphx::gpu::miopen_convolution<migraphx::op::convolution> first_op{conv};
    migraphx::gpu::miopen_convolution<migraphx::op::convolution> second_op{conv};

    scoped_output_capture capture(capture_path);
    const auto first_result  = first_op.compile(ctx, y, inputs);
    const auto second_result = second_op.compile(ctx, y, inputs);
    capture.restore();

    EXPECT(first_result == second_result);

    const auto trace = migraphx::read_string(capture_path);
    EXPECT(count_occurrences(trace, "miopenFindSolutions") == 1);
#endif
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
