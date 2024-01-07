/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2024 Advanced Micro Devices, Inc. All rights reserved.
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
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#include <thread>

#include <migraphx/env.hpp>
#include <migraphx/file_buffer.hpp>
#include <migraphx/process.hpp>
#include <migraphx/filesystem.hpp>

#ifndef _WIN32
#include <cstring>
#endif
#include "test.hpp"

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#include <io.h>
#include <fcntl.h>
#undef getpid
#define getpid _getpid
#endif

static migraphx::fs::path executable;
static std::string string_data = "Lorem ipsum dolor sit amet, consectetur adipiscing elit, "
                                 "sed do eiusmod tempor incididunt ut labore et dolore magna "
                                 "aliqua. Ut enim ad minim veniam, quis nostrud exercitation "
                                 "ullamco laboris nisi ut aliquip ex ea commodo consequat. "
                                 "Duis aute irure dolor in reprehenderit in voluptate velit "
                                 "esse cillum dolore eu fugiat nulla pariatur. Excepteur sint "
                                 "occaecat cupidatat non proident, sunt in culpa qui officia "
                                 "deserunt mollit anim id est laborum.";

std::vector<char> read_stdin()
{
    std::vector<char> result;
    std::array<char, 1024> buffer{};
    std::size_t len = 0;
#ifdef _WIN32
    if(_setmode(_fileno(stdin), _O_BINARY) == -1)
        throw std::runtime_error{"failure setting IO mode to binary"};
#endif
    while((len = std::fread(buffer.data(), 1, buffer.size(), stdin)) > 0)
    {
        if(std::ferror(stdin) != 0 and std::feof(stdin) == 0)
            throw std::runtime_error{std::strerror(errno)};

        result.insert(result.end(), buffer.begin(), buffer.begin() + len);
    }
    return result;
}

std::string get_temp_file(std::string_view test_id)
{
    auto pid = getpid();
    auto tid = std::this_thread::get_id();
    auto clk = std::chrono::steady_clock::now().time_since_epoch().count();
    std::stringstream ss;
    ss << std::hex << "migx-test-" << pid << '-' << tid << '-' << clk << '-' << test_id << ".txt";
    return ss.str();
}

TEST_CASE(string_stdin)
{
    auto out = migraphx::fs::temp_directory_path() / get_temp_file("string");

    migraphx::process{executable}.write(
        [&](auto writer) { writer(string_data.c_str(), string_data.size()); },
        "--stdin " + out.string());

    EXPECT(migraphx::fs::is_regular_file(out));

    std::string result{migraphx::read_string(out.string())};
    EXPECT(result == string_data);

    EXPECT(migraphx::fs::remove(out));
}

TEST_CASE(binary_stdin)
{
    std::random_device rd;
    std::independent_bits_engine<std::mt19937, CHAR_BIT, unsigned short> rbe(rd());

    std::vector<char> binary_data(4096);
    std::generate(binary_data.begin(), binary_data.end(), std::ref(rbe));

    auto out = migraphx::fs::temp_directory_path() / get_temp_file("binary");

    migraphx::process{executable}.write(
        [&](auto writer) { writer(binary_data.data(), binary_data.size()); },
        "--stdin " + out.string());

    EXPECT(migraphx::fs::is_regular_file(out));

    std::vector<char> result{migraphx::read_buffer(out)};
    EXPECT(result == binary_data);

    EXPECT(migraphx::fs::remove(out));
}

TEST_CASE(read_stdout)
{
    migraphx::process child{executable};
#ifdef WIN32
    auto out      = migraphx::fs::temp_directory_path() / get_temp_file("stdout");
    HANDLE handle = CreateFile(out.string().c_str(),
                               GENERIC_READ | GENERIC_WRITE,
                               0,
                               nullptr,
                               CREATE_ALWAYS,
                               FILE_ATTRIBUTE_NORMAL,
                               nullptr);
    EXPECT(handle != INVALID_HANDLE_VALUE);
    child.redirect_std_out(handle);
#else
    std::stringstream ss;
    std::streambuf* old{std::cout.rdbuf(ss.rdbuf())};
#endif
    child.exec("--stdout");
#ifdef WIN32
    CloseHandle(handle);
    handle = CreateFile(out.string().c_str(),
                        GENERIC_READ | GENERIC_WRITE,
                        0,
                        nullptr,
                        OPEN_EXISTING,
                        FILE_ATTRIBUTE_NORMAL,
                        nullptr);
    EXPECT(handle != INVALID_HANDLE_VALUE);
    auto size = GetFileSize(handle, nullptr);
    std::string buffer(size, '\0');
    DWORD read;
    EXPECT(ReadFile(handle, buffer.data(), size, &read, nullptr) == TRUE);
    CloseHandle(handle);
    EXPECT(migraphx::fs::remove(out));
#else
    std::string buffer = ss.str();
    std::cout.rdbuf(old);
#endif
    EXPECT(buffer == string_data);
}

TEST_CASE(current_working_dir)
{
    std::string filename{get_temp_file("cwd")};
    migraphx::fs::path tmp{migraphx::fs::temp_directory_path()};

    auto out = tmp / filename;

    migraphx::process{executable}.cwd(tmp).write(
        [&](auto writer) { writer(string_data.c_str(), string_data.size()); },
        "--stdin " + filename);

    EXPECT(migraphx::fs::is_regular_file(out));

    std::string result{migraphx::read_string(out)};
    EXPECT(result == string_data);

    EXPECT(migraphx::fs::remove(out));
}

TEST_CASE(environment_variable)
{
    migraphx::process child{executable};
#ifdef WIN32
    auto out      = migraphx::fs::temp_directory_path() / get_temp_file("stdout");
    HANDLE handle = CreateFile(out.string().c_str(),
                               GENERIC_WRITE,
                               0,
                               nullptr,
                               CREATE_ALWAYS,
                               FILE_ATTRIBUTE_NORMAL,
                               nullptr);
    EXPECT(handle != INVALID_HANDLE_VALUE);
    child.redirect_std_out(handle);
#else
    std::stringstream ss;
    std::streambuf* old{std::cout.rdbuf(ss.rdbuf())};
#endif
    child.exec("--stdout", "MIGRAPHX_PROCESS_TEST_ENVIRONMENT_VARIABLE=1");
#ifdef WIN32
    CloseHandle(handle);
    handle = CreateFile(out.string().c_str(),
                        GENERIC_READ,
                        0,
                        nullptr,
                        OPEN_EXISTING,
                        FILE_ATTRIBUTE_NORMAL,
                        nullptr);
    EXPECT(handle != INVALID_HANDLE_VALUE);
    auto size = GetFileSize(handle, nullptr);
    std::string buffer(size, '\0');
    DWORD read;
    EXPECT(ReadFile(handle, buffer.data(), size, &read, nullptr) == TRUE);
    CloseHandle(handle);
    EXPECT(migraphx::fs::remove(out));
#else
    std::string buffer = ss.str();
    std::cout.rdbuf(old);
#endif
    std::string reversed(string_data);
    std::reverse(reversed.begin(), reversed.end());
    EXPECT(buffer == reversed);
}

MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_PROCESS_TEST_ENVIRONMENT_VARIABLE)

int main(int argc, const char* argv[])
{
    if(argc > 1)
    {
        std::string arg = argv[1];
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
    else
    {
        executable = argv[0];
        test::run(argc, argv);
    }
}
