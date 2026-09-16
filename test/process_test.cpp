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
#include <climits>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#include <thread>

#include <migraphx/tmp_dir.hpp>
#include <migraphx/env.hpp>
#include <migraphx/file_buffer.hpp>
#include <migraphx/process.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/filesystem.hpp>

#ifndef _WIN32
#include <cstring>
#else
#include <io.h>
#include <fcntl.h>
#endif

#include "test.hpp"

static migraphx::fs::path executable; // NOLINT

constexpr std::string_view string_data =
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit, "
    "sed do eiusmod tempor incididunt ut labore et dolore magna "
    "aliqua. Ut enim ad minim veniam, quis nostrud exercitation "
    "ullamco laboris nisi ut aliquip ex ea commodo consequat. "
    "Duis aute irure dolor in reprehenderit in voluptate velit "
    "esse cillum dolore eu fugiat nulla pariatur. Excepteur sint "
    "occaecat cupidatat non proident, sunt in culpa qui officia "
    "deserunt mollit anim id est laborum.";

static std::vector<char> read_stdin()
{
    std::vector<char> result;
    std::array<char, 1024> buffer{};
    std::size_t len = 0;
#ifdef _WIN32
    // Set stream translation mode to BINARY to suppress translations.
    // https://learn.microsoft.com/en-us/cpp/c-runtime-library/reference/setmode?view=msvc-170
    auto old_mode = _setmode(_fileno(stdin), _O_BINARY);
    if(old_mode == -1)
        throw std::runtime_error{"failure setting IO mode to binary"};
#endif
    while((len = std::fread(buffer.data(), 1, buffer.size(), stdin)) > 0)
    {
        if(std::ferror(stdin) != 0 and std::feof(stdin) == 0)
            throw std::runtime_error{std::strerror(errno)};

        result.insert(result.end(), buffer.begin(), buffer.begin() + len);
    }
#ifdef _WIN32
    // Reset to the previously set translation mode.
    _setmode(_fileno(stdin), old_mode);
#endif
    return result;
}

TEST_CASE(string_stdin)
{
    auto tmp = migraphx::tmp_dir{};
    auto out = (tmp.path / "output.txt").string();

    migraphx::process{executable, {"--stdin", out}}.write(
        [&](const auto& writer) { writer(string_data.data(), string_data.size()); });

    EXPECT(migraphx::fs::is_regular_file(out));

    std::string result{migraphx::read_string(out)};
    EXPECT(result == string_data);

    EXPECT(migraphx::fs::remove(out));
}

TEST_CASE(binary_stdin)
{
    std::random_device rd;
    std::independent_bits_engine<std::mt19937, CHAR_BIT, unsigned short> rbe(rd());

    std::vector<char> binary_data(4096);
    std::generate(binary_data.begin(), binary_data.end(), std::ref(rbe));

    auto tmp = migraphx::tmp_dir{};
    auto out = (tmp.path / "output.bin").string();

    migraphx::process{executable, {"--stdin", out}}.write(
        [&](const auto& writer) { writer(binary_data.data(), binary_data.size()); });

    EXPECT(migraphx::fs::is_regular_file(out));

    std::vector<char> result{migraphx::read_buffer(out)};
    EXPECT(result == binary_data);

    EXPECT(migraphx::fs::remove(out));
}

TEST_CASE(read_stdout)
{
    std::string buffer;
    migraphx::process{executable, {"--stdout"}}.read(
        [&buffer](const char* buf, std::size_t size) { buffer = std::string{buf, size}; });
    EXPECT(buffer == string_data);
}

TEST_CASE(current_working_dir)
{
    constexpr auto filename = "output.txt";
    auto tmp                = migraphx::tmp_dir{};

    auto out = tmp.path / filename;

    migraphx::process{executable, {"--stdin", filename}}.cwd(tmp.path).write(
        [&](const auto& writer) { writer(string_data.data(), string_data.size()); });

    EXPECT(migraphx::fs::is_regular_file(out));

    std::string result{migraphx::read_string(out)};
    EXPECT(result == string_data);

    EXPECT(migraphx::fs::remove(out));
}

TEST_CASE(environment_variable)
{
    std::string buffer;
    migraphx::process{executable, {"--stdout"}}
        .env({"MIGRAPHX_PROCESS_TEST_ENVIRONMENT_VARIABLE=1"})
        .read([&buffer](const char* buf, std::size_t size) { buffer = std::string{buf, size}; });
    std::string reversed(string_data);
    std::reverse(reversed.begin(), reversed.end());
    EXPECT(buffer == reversed);
}

// ---------------------------------------------------------------------------------------------
// read_write: stdin and stdout pumped at the same time.
//
// Every child mode below is introduced by this token, so anything else on the command line belongs
// to the modes above or to test::run and its --list / --start-from / case-filter handling.
// ---------------------------------------------------------------------------------------------

static const char* const child_flag = "--migraphx-read-write-child";

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

static std::vector<char> read_write_child(const std::vector<std::string>& mode_args,
                                          const std::vector<char>& data)
{
    std::vector<std::string> args{child_flag};
    args.insert(args.end(), mode_args.begin(), mode_args.end());
    std::vector<char> result;
    migraphx::process{executable, args}.read_write(
        [&](const auto& writer) {
            if(not data.empty())
                writer(data.data(), data.size());
        },
        [&](const char* buf, std::size_t n) {
            if(n > 0)
                result.assign(buf, buf + n);
        });
    return result;
}

TEST_CASE(read_write_text)
{
    std::vector<char> data{string_data.begin(), string_data.end()};
    auto result = read_write_child({"echo", "1"}, data);
    EXPECT(result == data);
}

TEST_CASE(read_write_binary)
{
    // Includes embedded nulls and stray CR and LF bytes.
    auto data = make_payload(4096);
    EXPECT(read_write_child({"echo", "1"}, data) == data);
}

TEST_CASE(read_write_empty_stdin) { EXPECT(read_write_child({"echo", "1"}, {}).empty()); }

// The anti-deadlock regression test. The "stream" child echoes each chunk as it reads it rather
// than draining stdin first, so both pipes fill at the same time: an implementation that writes all
// of stdin before reading any stdout blocks here and never finishes.
TEST_CASE(read_write_large_bidirectional)
{
    auto data = make_payload(1024 * 1024);
    EXPECT(read_write_child({"stream", "4"}, data).size() == data.size() * 4);
}

TEST_CASE(read_write_nonzero_exit_code)
{
    EXPECT(test::throws([&] { read_write_child({"fail"}, {}); }));
}

// A child that dies from a signal (POSIX) or an SEH exception (Windows) has no exit status. Without
// a mapping for that, WEXITSTATUS of a signalled child is 0 and a crashed compiler would look like
// success.
TEST_CASE(read_write_abnormal_termination)
{
    EXPECT(test::throws([&] { read_write_child({"crash"}, {}); }));
}

// Output produced before a failure must still be drained. The bytes are discarded along with the
// exception, but a child left blocked on a full stdout pipe would hang the wait instead of
// throwing.
TEST_CASE(read_write_output_then_failure)
{
    auto data = make_payload(256 * 1024);
    EXPECT(test::throws([&] { read_write_child({"echo-then-fail"}, data); }));
}

// A child that exits without reading its stdin must surface as an exception rather than a SIGPIPE
// that takes the whole test binary down.
TEST_CASE(read_write_child_ignores_stdin)
{
    auto data = make_payload(1024 * 1024);
    EXPECT(test::throws([&] { read_write_child({"ignore-stdin"}, data); }));
}

TEST_CASE(read_write_stderr_does_not_corrupt_stdout)
{
    auto data = make_payload(8192);
    EXPECT(read_write_child({"echo", "1", "noise"}, data) == data);
}

TEST_CASE(read_write_missing_executable_throws)
{
    auto missing = executable.parent_path() / "migraphx-no-such-program";
    EXPECT(test::throws([&] {
        migraphx::process{missing}.read_write([](const auto&) {}, [](const char*, std::size_t) {});
    }));
}

// cwd and env are not plumbed through the direct spawn, so asking for them must fail loudly rather
// than be silently ignored.
TEST_CASE(read_write_rejects_cwd_and_env)
{
    auto tmp = migraphx::tmp_dir{};
    EXPECT(test::throws([&] {
        migraphx::process{executable, {child_flag, "fail"}}.cwd(tmp.path).read_write(
            [](const auto&) {}, [](const char*, std::size_t) {});
    }));
    EXPECT(test::throws([&] {
        migraphx::process{executable, {child_flag, "fail"}}.env({"A=1"}).read_write(
            [](const auto&) {}, [](const char*, std::size_t) {});
    }));
}

// Arguments round-trip through the platform's command-line encoding. On Windows the quoting has to
// double only the backslash runs that precede a quote; escaping every backslash turns C:\a\b into
// C:\\a\\b on the far side, and neither form is visible without echoing argv back.
TEST_CASE(read_write_argv_round_trip)
{
    std::vector<std::string> args = {
        "C:\\a\\b", "C:\\a\\b\\", "a\"b", "a\\\"b", "a\\\\", "x y", "", "--looks-like-a-flag"};
    std::vector<std::string> mode_args = {"args"};
    mode_args.insert(mode_args.end(), args.begin(), args.end());

    // The child writes each argument followed by a null byte.
    std::vector<std::string> got;
    std::string current;
    for(auto c : read_write_child(mode_args, {}))
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
TEST_CASE(read_write_concurrent_spawns)
{
    constexpr std::size_t n = 8;
    auto data               = make_payload(64 * 1024);

    std::mutex mutex;
    std::condition_variable ready;
    std::size_t waiting = 0;

    std::vector<std::vector<char>> results(n);
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
                results[i] = read_write_child({"echo", "2"}, data);
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
        // Content, not just size: crossed pipe ends corrupt the bytes long before they hang.
        EXPECT(results[i].size() == data.size() * 2);
        EXPECT(std::equal(data.begin(), data.end(), results[i].begin()));
        EXPECT(std::equal(data.begin(), data.end(), results[i].begin() + data.size()));
    }
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

MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_PROCESS_TEST_ENVIRONMENT_VARIABLE)

int main(int argc, const char* argv[])
{
    if(argc > 1)
    {
        std::string arg = argv[1];
        if(arg == child_flag)
            return run_child(std::vector<std::string>(argv + 2, argv + argc));
        if(arg == "--stdin")
        {
            migraphx::write_buffer(argv[2], read_stdin());
            return 0;
        }
        if(arg == "--stdout")
        {
            std::vector<char> result{string_data.begin(), string_data.end()};
            if(migraphx::enabled(MIGRAPHX_PROCESS_TEST_ENVIRONMENT_VARIABLE{}))
                std::reverse(result.begin(), result.end());
            std::fwrite(result.data(), 1, result.size(), stdout);
            return 0;
        }
    }
    // Anything else is for test::run. posix_spawn and CreateProcessA do not search PATH, so a
    // bare-name argv[0] would not resolve.
    executable = migraphx::fs::absolute(argv[0]);
    test::run(argc, argv);
    return 0;
}
