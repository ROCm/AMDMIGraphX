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
#include <migraphx/process.hpp>
#include <migraphx/errors.hpp>
#include <migraphx/env.hpp>
#include <functional>
#include <iostream>
#include <optional>

#ifdef _WIN32
// cppcheck-suppress definePrefix
#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#else
#include <unistd.h>
#endif

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_TRACE_CMD_EXECUTE)

#ifndef _WIN32

std::function<void(const char*)> redirect_to(std::ostream& os)
{
    return [&](const char* x) { os << x; };
}

template <class F>
int exec(const std::string& cmd, const char* type, F f)
{
    int ec = 0;
    if(enabled(MIGRAPHX_TRACE_CMD_EXECUTE{}))
        std::cout << cmd << std::endl;
    auto closer = [&](FILE* stream) {
        auto status = pclose(stream);
        ec          = WIFEXITED(status) ? WEXITSTATUS(status) : 0; // NOLINT
    };
    {
        // TODO: Use execve instead of popen
        std::unique_ptr<FILE, decltype(closer)> pipe(popen(cmd.c_str(), type), closer); // NOLINT
        if(not pipe)
            MIGRAPHX_THROW("popen() failed: " + cmd);
        f(pipe.get());
    }
    return ec;
}

int exec(const std::string& cmd, const std::function<void(const char*)>& std_out)
{
    return exec(cmd, "r", [&](FILE* f) {
        std::array<char, 128> buffer;
        while(fgets(buffer.data(), buffer.size(), f) != nullptr)
            std_out(buffer.data());
    });
}

int exec(const std::string& cmd, std::function<void(process::writer)> std_in)
{
    return exec(cmd, "w", [&](FILE* f) {
        std_in([&](const char* buffer, std::size_t n) { std::fwrite(buffer, 1, n, f); });
    });
}

#else

constexpr std::size_t MIGRAPHX_PROCESS_BUFSIZE = 4096;

class pipe
{
    public:
    explicit pipe(bool inherit_handle = true)
    {
        SECURITY_ATTRIBUTES attrs;
        attrs.nLength              = sizeof(SECURITY_ATTRIBUTES);
        attrs.bInheritHandle       = inherit_handle ? TRUE : FALSE;
        attrs.lpSecurityDescriptor = nullptr;

        if(CreatePipe(&m_read, &m_write, &attrs, 0) == FALSE)
            throw GetLastError();

        if(SetHandleInformation(&m_read, HANDLE_FLAG_INHERIT, 0) == FALSE)
            throw GetLastError();
    }

    pipe(const pipe&) = delete;
    pipe& operator=(const pipe&) = delete;

    pipe(pipe&&) = default;

    ~pipe()
    {
        CloseHandle(m_read);
        m_read = nullptr;
        CloseHandle(m_write);
        m_write = nullptr;
    }

    std::optional<std::pair<bool, DWORD>> read(LPVOID buffer, DWORD length) const
    {
        DWORD bytes_read;
        if(ReadFile(m_read, buffer, length, &bytes_read, nullptr) == FALSE)
        {
            DWORD error{GetLastError()};
            if(error != ERROR_MORE_DATA)
            {
                return std::nullopt;
            }
            return {{true, bytes_read}};
        }
        return {{false, bytes_read}};
    }

    HANDLE get_read_handle() const { return m_read; }

    bool write(LPCVOID buffer, DWORD length) const
    {
        DWORD bytes_written;
        return WriteFile(m_write, buffer, length, &bytes_written, nullptr) == TRUE;
    }

    HANDLE get_write_handle() const { return m_write; }

    private:
    HANDLE m_write = nullptr, m_read = nullptr;
};

template <typename F>
int exec(const std::string& cmd, F f)
{
    try
    {
        if(enabled(MIGRAPHX_TRACE_CMD_EXECUTE{}))
            std::cout << cmd << std::endl;

        STARTUPINFO info;
        PROCESS_INFORMATION process_info;

        pipe in{}, out{};

        ZeroMemory(&info, sizeof(STARTUPINFO));
        info.cb         = sizeof(STARTUPINFO);
        info.hStdError  = out.get_write_handle();
        info.hStdOutput = out.get_write_handle();
        info.hStdInput  = in.get_read_handle();
        info.dwFlags |= STARTF_USESTDHANDLES;

        ZeroMemory(&process_info, sizeof(process_info));

        if(CreateProcess(nullptr,
                         const_cast<LPSTR>(cmd.c_str()),
                         nullptr,
                         nullptr,
                         TRUE,
                         0,
                         nullptr,
                         nullptr,
                         &info,
                         &process_info) == FALSE)
        {
            return GetLastError();
        }

        f(in, out);

        WaitForSingleObject(process_info.hProcess, INFINITE);

        DWORD status{};
        GetExitCodeProcess(process_info.hProcess, &status);

        CloseHandle(process_info.hProcess);
        CloseHandle(process_info.hThread);

        return static_cast<int>(status);
    }
    // cppcheck-suppress catchExceptionByValue
    catch(DWORD last_error)
    {
        return last_error;
    }
}

int exec(const std::string& cmd)
{
    TCHAR buffer[MIGRAPHX_PROCESS_BUFSIZE];
    HANDLE std_out{GetStdHandle(STD_OUTPUT_HANDLE)};
    return (std_out == nullptr || std_out == INVALID_HANDLE_VALUE)
               ? GetLastError()
               : exec(cmd, [&](const pipe&, const pipe& out) {
                     for(;;)
                     {
                         if(auto result = out.read(buffer, MIGRAPHX_PROCESS_BUFSIZE))
                         {
                             auto [more_data, bytes_read] = *result;
                             if(!more_data || bytes_read == 0)
                                 break;
                             DWORD written;
                             if(WriteFile(std_out, buffer, bytes_read, &written, nullptr) == FALSE)
                                 break;
                         }
                     }
                 });
}

int exec(const std::string& cmd, std::function<void(process::writer)> std_in)
{
    return exec(cmd, [&](const pipe& in, const pipe&) {
        std_in([&](const char* buffer, std::size_t n) { in.write(buffer, n); });
    });
}

#endif

struct process_impl
{
    std::string command{};
    fs::path cwd{};

    std::string get_command() const
    {
        std::string result;
        if(not cwd.empty())
            result += "cd " + cwd.string() + "; ";
        result += command;
        return result;
    }

    template <class... Ts>
    void check_exec(Ts&&... xs) const
    {
        int ec = migraphx::exec(std::forward<Ts>(xs)...);
        if(ec != 0)
            MIGRAPHX_THROW("Command " + get_command() + " exited with status " +
                           std::to_string(ec));
    }
};

process::process(const std::string& cmd) : impl(std::make_unique<process_impl>())
{
    impl->command = cmd;
}

process::process(process&&) noexcept = default;

process& process::operator=(process rhs)
{
    std::swap(impl, rhs.impl);
    return *this;
}

process::~process() noexcept = default;

process& process::cwd(const fs::path& p)
{
    impl->cwd = p;
    return *this;
}

void process::exec()
{
#ifndef _WIN32
    impl->check_exec(impl->get_command(), redirect_to(std::cout));
#else
    impl->check_exec(impl->get_command());
#endif
}

void process::write(std::function<void(process::writer)> pipe_in)
{
    impl->check_exec(impl->get_command(), std::move(pipe_in));
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
