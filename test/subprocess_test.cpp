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
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <mutex>
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

// Every child mode is introduced by this token, so anything else on the command line belongs to
// test::run and its --list / --start-from / case-filter handling.
static const char* const child_flag = "--migraphx-subprocess-child";

static migraphx::fs::path& executable()
{
    static migraphx::fs::path path;
    return path;
}

// A payload with every byte value in it, so a text-mode stdout that mangles \n or truncates at \0
// cannot pass.
static std::vector<char> make_payload(std::size_t n)
{
    std::vector<char> result(n);
    std::size_t i = 0;
    std::generate(
        result.begin(), result.end(), [&] { return static_cast<char>((i++ * 31 + 7) % 256); });
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

static void child_write(const std::vector<char>& data)
{
    // fwrite with a null pointer is undefined even for a zero count, and data() may be null for an
    // empty vector.
    if(not data.empty())
        std::fwrite(data.data(), 1, data.size(), stdout);
}

TEST_CASE(echo_text)
{
    std::string data = "the quick brown fox";
    auto result      = migraphx::execute_subprocess(
        executable(), {child_flag, "echo", "1"}, std::vector<char>{data.begin(), data.end()});
    EXPECT(result.success());
    EXPECT(std::string(result.stdout_data.begin(), result.stdout_data.end()) == data);
}

TEST_CASE(echo_binary)
{
    // Includes embedded nulls and stray CR and LF bytes.
    auto data   = make_payload(4096);
    auto result = migraphx::execute_subprocess(executable(), {child_flag, "echo", "1"}, data);
    EXPECT(result.success());
    EXPECT(result.stdout_data == data);
}

TEST_CASE(empty_stdin)
{
    auto result = migraphx::execute_subprocess(executable(), {child_flag, "echo", "1"}, {});
    EXPECT(result.success());
    EXPECT(result.stdout_data.empty());
}

// The anti-deadlock regression test. The "stream" child echoes each chunk as it reads it rather
// than draining stdin first, so both pipes fill at the same time: an implementation that writes all
// of stdin before reading any stdout blocks here and never finishes.
TEST_CASE(large_bidirectional)
{
    auto data   = make_payload(1024 * 1024);
    auto result = migraphx::execute_subprocess(executable(), {child_flag, "stream", "4"}, data);
    EXPECT(result.success());
    EXPECT(result.stdout_data.size() == data.size() * 4);
}

TEST_CASE(nonzero_exit_code)
{
    auto result = migraphx::execute_subprocess(executable(), {child_flag, "fail"}, {});
    EXPECT(result.exit_code == 3);
    EXPECT(not result.success());
}

// A child that dies from a signal (POSIX) or an SEH exception (Windows) has no exit status, and is
// reported as -1. Without this mapping WEXITSTATUS of a signalled child is 0, so a crashed
// compiler would look like success to compile_hip_src.
TEST_CASE(abnormal_termination)
{
    auto result = migraphx::execute_subprocess(executable(), {child_flag, "crash"}, {});
    EXPECT(result.exit_code == -1);
    EXPECT(not result.success());
}

// Output written before a failure must still be drained: the exit status and the bytes are
// independent results.
TEST_CASE(output_then_failure)
{
    auto data   = make_payload(256 * 1024);
    auto result = migraphx::execute_subprocess(executable(), {child_flag, "echo-then-fail"}, data);
    EXPECT(result.exit_code == 3);
    EXPECT(result.stdout_data == data);
}

// A child that exits without reading its stdin must not turn into an exception on either platform;
// the exit status is the verdict.
TEST_CASE(child_ignores_stdin)
{
    auto data   = make_payload(1024 * 1024);
    auto result = migraphx::execute_subprocess(executable(), {child_flag, "ignore-stdin"}, data);
    EXPECT(result.exit_code == 3);
}

TEST_CASE(stderr_does_not_corrupt_stdout)
{
    auto data = make_payload(8192);
    auto result =
        migraphx::execute_subprocess(executable(), {child_flag, "echo", "1", "noise"}, data);
    EXPECT(result.success());
    EXPECT(result.stdout_data == data);
}

TEST_CASE(missing_executable_throws)
{
    auto missing = executable().parent_path() / "migraphx-no-such-program";
    EXPECT(test::throws([&] { migraphx::execute_subprocess(missing, {}, {}); }));
}

// Arguments round-trip through the platform's command-line encoding. On Windows quote_arg has to
// double only the backslash runs that precede a quote; escaping every backslash turns C:\a\b into
// C:\\a\\b on the far side, and neither form is visible without echoing argv back.
TEST_CASE(argv_round_trip)
{
    std::vector<std::string> args = {
        "C:\\a\\b", "C:\\a\\b\\", "a\"b", "a\\\"b", "a\\\\", "x y", "", "--looks-like-a-flag"};
    std::vector<std::string> argv = {child_flag, "args"};
    argv.insert(argv.end(), args.begin(), args.end());

    auto result = migraphx::execute_subprocess(executable(), argv, {});
    EXPECT(result.success());

    // The child writes each argument followed by a null byte.
    std::vector<std::string> got;
    std::string current;
    for(auto c : result.stdout_data)
    {
        if(c == '\0')
        {
            got.push_back(current);
            current.clear();
        }
        else
        {
            current.push_back(c);
        }
    }
    EXPECT(got == args);
}

// Concurrent spawns must not inherit each other's pipe ends, or somebody never sees EOF. The
// barrier makes the threads enter the spawn together, which is the window the handle list closes.
TEST_CASE(concurrent_spawns)
{
    constexpr std::size_t n = 8;
    auto data               = make_payload(64 * 1024);

    std::mutex mutex;
    std::condition_variable ready;
    std::size_t waiting = 0;

    std::vector<migraphx::subprocess_result> results(n);
    std::vector<std::string> errors(n);
    std::vector<std::thread> threads;
    for(std::size_t i = 0; i < n; i++)
    {
        threads.emplace_back([&, i] {
            {
                std::unique_lock<std::mutex> lock(mutex);
                waiting++;
                if(waiting == n)
                    ready.notify_all();
                else
                    ready.wait(lock, [&] { return waiting == n; });
            }
            // An exception escaping a std::thread calls std::terminate, which would abort the whole
            // binary with no indication of which case died.
            try
            {
                results[i] =
                    migraphx::execute_subprocess(executable(), {child_flag, "echo", "2"}, data);
            }
            catch(const std::exception& e)
            {
                errors[i] = e.what();
            }
        });
    }
    for(auto& t : threads)
        t.join();

    for(std::size_t i = 0; i < n; i++)
    {
        EXPECT(errors[i].empty());
        EXPECT(results[i].success());
        // Content, not just size: crossed pipe ends corrupt the bytes long before they hang.
        EXPECT(results[i].stdout_data.size() == data.size() * 2);
        EXPECT(std::equal(data.begin(), data.end(), results[i].stdout_data.begin()));
        EXPECT(std::equal(data.begin(), data.end(), results[i].stdout_data.begin() + data.size()));
    }
}

static int run_child(const std::vector<std::string>& args)
{
    set_binary_mode();
    const auto& mode = args.at(0);
    if(mode == "fail")
        return 3;
    if(mode == "crash")
    {
#ifdef _WIN32
        // Exit with an SEH-shaped status so the >INT_MAX mapping is exercised without raising a
        // real fault, which would pop a Windows Error Reporting dialog in CI.
        std::exit(static_cast<int>(0xC0000005u));
#else
        std::abort();
#endif
    }
    if(mode == "ignore-stdin")
        return 3;
    if(mode == "args")
    {
        for(auto it = args.begin() + 1; it != args.end(); ++it)
        {
            std::fwrite(it->data(), 1, it->size(), stdout);
            std::fputc('\0', stdout);
        }
        std::fflush(stdout);
        return 0;
    }
    if(mode == "echo-then-fail")
    {
        child_write(child_read_stdin());
        std::fflush(stdout);
        return 3;
    }
    if(mode == "echo")
    {
        auto repeat = std::stoul(args.at(1));
        auto data   = child_read_stdin();
        if(migraphx::contains(args, std::string{"noise"}))
            std::cerr << std::string(8192, 'x') << std::endl;
        for(std::size_t i = 0; i < repeat; i++)
            child_write(data);
        std::fflush(stdout);
        return 0;
    }
    if(mode == "stream")
    {
        // Echo as we read, so stdin and stdout are both in flight at once.
        auto repeat = std::stoul(args.at(1));
        std::array<char, 4096> buffer{};
        std::size_t len = 0;
        while((len = std::fread(buffer.data(), 1, buffer.size(), stdin)) > 0)
        {
            for(std::size_t i = 0; i < repeat; i++)
                std::fwrite(buffer.data(), 1, len, stdout);
        }
        std::fflush(stdout);
        return 0;
    }
    std::cerr << "unknown child mode: " << mode << std::endl;
    return 2;
}

int main(int argc, const char* argv[])
{
    if(argc > 1 and std::string(argv[1]) == child_flag)
        return run_child(std::vector<std::string>(argv + 2, argv + argc));
    // posix_spawn and CreateProcessA do not search PATH, so a bare-name argv[0] would not resolve.
    executable() = migraphx::fs::absolute(argv[0]);
    test::run(argc, argv);
}
