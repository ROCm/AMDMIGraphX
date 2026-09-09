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

#include <algorithm>
#include <array>
#include <cstdio>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <migraphx/filesystem.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/subprocess.hpp>

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif

#include "test.hpp"

static migraphx::fs::path executable; // NOLINT

// A payload with every byte value in it, so a text-mode stdout that mangles \n or truncates at \0
// cannot pass.
static std::vector<char> make_payload(std::size_t n)
{
    std::vector<char> result(n);
    for(std::size_t i = 0; i < n; i++)
        result[i] = static_cast<char>((i * 31 + 7) % 256);
    return result;
}

static void set_binary_mode()
{
#ifdef _WIN32
    _setmode(_fileno(stdin), _O_BINARY);
    _setmode(_fileno(stdout), _O_BINARY);
#endif
}

static std::vector<char> child_read_stdin()
{
    std::vector<char> result;
    std::array<char, 1024> buffer{};
    std::size_t len = 0;
    while((len = std::fread(buffer.data(), 1, buffer.size(), stdin)) > 0)
        result.insert(result.end(), buffer.begin(), buffer.begin() + len);
    return result;
}

TEST_CASE(echo_text)
{
    std::string data = "the quick brown fox";
    auto result      = migraphx::execute_subprocess(
        executable, {"--echo", "1"}, std::vector<char>{data.begin(), data.end()});
    EXPECT(result.exit_code == 0);
    EXPECT(std::string(result.stdout_data.begin(), result.stdout_data.end()) == data);
}

TEST_CASE(echo_binary)
{
    // Includes embedded nulls and stray CR and LF bytes.
    auto data   = make_payload(4096);
    auto result = migraphx::execute_subprocess(executable, {"--echo", "1"}, data);
    EXPECT(result.exit_code == 0);
    EXPECT(result.stdout_data == data);
}

TEST_CASE(empty_stdin)
{
    auto result = migraphx::execute_subprocess(executable, {"--echo", "1"}, {});
    EXPECT(result.exit_code == 0);
    EXPECT(result.stdout_data.empty());
}

// Both directions exceed the pipe buffer at once: an implementation that writes all of stdin before
// reading any stdout deadlocks here.
TEST_CASE(large_bidirectional)
{
    auto data   = make_payload(1024 * 1024);
    auto result = migraphx::execute_subprocess(executable, {"--echo", "4"}, data);
    EXPECT(result.exit_code == 0);
    EXPECT(result.stdout_data.size() == data.size() * 4);
    for(std::size_t i = 0; i < 4; i++)
    {
        EXPECT(std::equal(data.begin(),
                          data.end(),
                          result.stdout_data.begin() + static_cast<std::ptrdiff_t>(i * data.size())));
    }
}

TEST_CASE(nonzero_exit_code)
{
    auto result = migraphx::execute_subprocess(executable, {"--fail"}, {});
    EXPECT(result.exit_code != 0);
}

TEST_CASE(stderr_does_not_corrupt_stdout)
{
    auto data   = make_payload(8192);
    auto result = migraphx::execute_subprocess(executable, {"--echo", "1", "--noise"}, data);
    EXPECT(result.exit_code == 0);
    EXPECT(result.stdout_data == data);
}

TEST_CASE(missing_executable_fails)
{
    // posix_spawn is allowed to defer the exec failure to the child, so accept either a throw or a
    // non-zero exit status.
    auto missing = executable.parent_path() / "migraphx-no-such-program";
    try
    {
        EXPECT(migraphx::execute_subprocess(missing, {}, {}).exit_code != 0);
    }
    catch(const std::exception&)
    {
    }
}

// Concurrent spawns must not inherit each other's pipe ends, or somebody never sees EOF and hangs.
TEST_CASE(concurrent_spawns)
{
    constexpr std::size_t n = 8;
    auto data               = make_payload(64 * 1024);
    std::vector<migraphx::subprocess_result> results(n);
    std::vector<std::thread> threads;
    for(std::size_t i = 0; i < n; i++)
    {
        threads.emplace_back([&, i] {
            results[i] = migraphx::execute_subprocess(executable, {"--echo", "2"}, data);
        });
    }
    for(auto& t : threads)
        t.join();
    for(const auto& result : results)
    {
        EXPECT(result.exit_code == 0);
        EXPECT(result.stdout_data.size() == data.size() * 2);
    }
}

int main(int argc, const char* argv[])
{
    if(argc > 1)
    {
        std::vector<std::string> args(argv + 1, argv + argc);
        set_binary_mode();
        if(args[0] == "--fail")
            return 3;
        if(args[0] == "--echo")
        {
            auto repeat = std::stoul(args.at(1));
            auto data   = child_read_stdin();
            if(migraphx::contains(args, std::string{"--noise"}))
                std::cerr << std::string(8192, 'x') << std::endl;
            for(std::size_t i = 0; i < repeat; i++)
                std::fwrite(data.data(), 1, data.size(), stdout);
            std::fflush(stdout);
            return 0;
        }
        return 2;
    }
    executable = argv[0];
    test::run(argc, argv);
}
