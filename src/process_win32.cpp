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

// cppcheck-suppress definePrefix
#define WIN32_LEAN_AND_MEAN
#include <Windows.h>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_TRACE_CMD_EXECUTE)

#define MIGRAPHX_PROCESS_BUFSIZE 4096

class pipe
{
    public:
    explicit pipe(bool inherit_handle = true)
    {
        SECURITY_ATTRIBUTES attrs;
        attrs.nLength              = sizeof(SECURITY_ATTRIBUTES);
        attrs.bInheritHandle       = inherit_handle ? TRUE : FALSE;
        attrs.lpSecurityDescriptor = nullptr;

        if(CreatePipe(&hRead_, &hWrite_, &attrs, 0) == FALSE)
            throw GetLastError();

        if(SetHandleInformation(&hRead_, HANDLE_FLAG_INHERIT, 0) == FALSE)
            throw GetLastError();
    }

    ~pipe()
    {
        close_write_handle();
        close_read_handle();
    }

    HANDLE get_read_handle() const { return hRead_; }

    void close_read_handle()
    {
        if(hRead_ != nullptr)
        {
            CloseHandle(hRead_);
            hRead_ = nullptr;
        }
    }

    HANDLE get_write_handle() const { return hWrite_; }

    void close_write_handle()
    {
        if(hWrite_ != nullptr)
        {
            CloseHandle(hWrite_);
            hWrite_ = nullptr;
        }
    }

    private:
    HANDLE hWrite_{nullptr}, hRead_{nullptr};
};

int exec(const std::string& cmd)
{
    try
    {
        if(enabled(MIGRAPHX_TRACE_CMD_EXECUTE{}))
            std::cout << cmd << std::endl;

        pipe stdin_{}, stdout_{};

        STARTUPINFO info;
        PROCESS_INFORMATION processInfo;

        ZeroMemory(&info, sizeof(STARTUPINFO));
        info.cb         = sizeof(STARTUPINFO);
        info.hStdError  = stdout_.get_write_handle();
        info.hStdOutput = stdout_.get_write_handle();
        info.hStdInput  = stdin_.get_read_handle();
        info.dwFlags |= STARTF_USESTDHANDLES;

        ZeroMemory(&processInfo, sizeof(processInfo));

        LPSTR lpCmdLn{const_cast<LPSTR>(cmd.c_str())};

        BOOL bSuccess = CreateProcess(
            nullptr, lpCmdLn, nullptr, nullptr, TRUE, 0, nullptr, nullptr, &info, &processInfo);

        if(bSuccess == FALSE)
            return GetLastError();

        DWORD dwRead, dwWritten;
        TCHAR chBuf[MIGRAPHX_PROCESS_BUFSIZE];
        HANDLE hStdOut{GetStdHandle(STD_OUTPUT_HANDLE)};

        for(;;)
        {
            BOOL bRead = ReadFile(
                stdout_.get_read_handle(), chBuf, MIGRAPHX_PROCESS_BUFSIZE, &dwRead, nullptr);

            if(bRead == FALSE)
            {
                if(GetLastError() != ERROR_MORE_DATA)
                    break;
            }

            if(dwRead == 0)
                break;

            BOOL bWrite = WriteFile(hStdOut, chBuf, dwRead, &dwWritten, nullptr);

            if(bWrite == FALSE)
                break;
        }

        WaitForSingleObject(processInfo.hProcess, INFINITE);

        DWORD status{};
        GetExitCodeProcess(processInfo.hProcess, &status);

        CloseHandle(processInfo.hProcess);
        CloseHandle(processInfo.hThread);

        return static_cast<int>(status);
    }
    // cppcheck-suppress catchExceptionByValue
    catch(DWORD lastError)
    {
        return lastError;
    }
}

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

void process::exec() { impl->check_exec(impl->get_command()); }

void process::write(std::function<void(process::writer)> pipe_in)
{
    impl->check_exec(impl->get_command());
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
