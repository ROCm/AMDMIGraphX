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
#include <migraphx/env.hpp>
#include <migraphx/errors.hpp>
#include <migraphx/process.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/tmp_dir.hpp>
#include <migraphx/fileutils.hpp>
#include <migraphx/scope_guard.hpp>
#include <algorithm>
#include <numeric>
#include <functional>
#include <iostream>
#include <vector>

#ifdef _WIN32
// cppcheck-suppress definePrefix
#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#include <cstring>
#include <sstream>
#include <optional>
#endif

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_TRACE_CMD_EXECUTE)

#ifndef _WIN32

#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>
#include <fcntl.h>
#include <cerrno>
#include <array>

extern char** environ;

static std::function<void(const char*)> redirect_to(std::ostream& os)
{
    return [&](const char* x) { os << x; };
}

static std::vector<std::string> copy_environ()
{
    std::vector<std::string> env;
    if(environ == nullptr)
        return env;
    for(char** e = environ; *e != nullptr; ++e)
        env.emplace_back(*e);
    return env;
}

static std::vector<char*> to_cstr_array(std::vector<std::string>& storage)
{
    std::vector<char*> result;
    result.reserve(storage.size() + 1);
    for(auto& s : storage)
        result.push_back(s.data());
    result.push_back(nullptr);
    return result;
}

static int wait_for_pid(pid_t pid)
{
    int status = 0;
    while(waitpid(pid, &status, 0) < 0)
    {
        if(errno == EINTR)
            continue;
        MIGRAPHX_THROW("waitpid() failed");
    }
    if(WIFEXITED(status))
        return WEXITSTATUS(status);
    MIGRAPHX_THROW("Child process terminated abnormally");
}

struct spawn_options
{
    const fs::path* cwd           = nullptr;
    const std::vector<std::string>* env = nullptr;
    int stdin_fd                  = -1;
    int stdout_fd                 = -1;
    bool close_stdin_write        = false;
};

static pid_t spawn_process_async(const std::string& command,
                                 const std::vector<std::string>& args,
                                 const spawn_options& options)
{
    std::vector<std::string> argv_storage;
    argv_storage.push_back(command);
    argv_storage.insert(argv_storage.end(), args.begin(), args.end());
    auto argv = to_cstr_array(argv_storage);

    std::vector<std::string> env_storage;
    std::vector<char*> envp;
    if(options.env != nullptr and not options.env->empty())
    {
        env_storage = copy_environ();
        env_storage.insert(env_storage.end(), options.env->begin(), options.env->end());
        envp = to_cstr_array(env_storage);
    }

    posix_spawn_file_actions_t actions;
    if(posix_spawn_file_actions_init(&actions) != 0)
        MIGRAPHX_THROW("posix_spawn_file_actions_init() failed");

    auto cleanup = scope_guard{[&] { posix_spawn_file_actions_destroy(&actions); }};

    if(options.cwd != nullptr and not options.cwd->empty())
    {
        if(posix_spawn_file_actions_addchdir_np(&actions, options.cwd->c_str()) != 0)
            MIGRAPHX_THROW("posix_spawn_file_actions_addchdir_np() failed");
    }

    if(options.stdin_fd >= 0)
    {
        if(posix_spawn_file_actions_adddup2(&actions, options.stdin_fd, STDIN_FILENO) != 0)
            MIGRAPHX_THROW("posix_spawn_file_actions_adddup2() failed for stdin");
    }

    if(options.stdout_fd >= 0)
    {
        if(posix_spawn_file_actions_adddup2(&actions, options.stdout_fd, STDOUT_FILENO) != 0)
            MIGRAPHX_THROW("posix_spawn_file_actions_adddup2() failed for stdout");
    }

    pid_t pid = 0;
    const int spawn_result =
        envp.empty()
            ? posix_spawn(&pid, command.c_str(), &actions, nullptr, argv.data(), environ)
            : posix_spawn(
                  &pid, command.c_str(), &actions, nullptr, argv.data(), envp.data());
    if(spawn_result != 0)
        MIGRAPHX_THROW("posix_spawn() failed for command: " + command);

    if(options.close_stdin_write)
        close(options.stdin_fd);

    return pid;
}

static int spawn_process(const std::string& command,
                         const std::vector<std::string>& args,
                         const spawn_options& options)
{
    return wait_for_pid(spawn_process_async(command, args, options));
}

template <class F>
static int exec_process(const std::string& command,
                        const std::vector<std::string>& args,
                        const spawn_options& options,
                        F&& read_output)
{
    int pipefd[2] = {-1, -1};
    spawn_options spawn_opts = options;
    const bool capture_output = options.stdout_fd < 0 and options.stdin_fd < 0;

    if(capture_output)
    {
        if(pipe(pipefd) != 0)
            MIGRAPHX_THROW("pipe() failed");
        spawn_opts.stdout_fd = pipefd[1];
    }

    int ec = spawn_process(command, args, spawn_opts);

    if(capture_output)
    {
        close(pipefd[1]);
        std::array<char, 128> buffer{};
        ssize_t n = 0;
        while((n = read(pipefd[0], buffer.data(), buffer.size())) > 0)
            read_output(buffer.data(), static_cast<std::size_t>(n));
        close(pipefd[0]);
    }

    return ec;
}

static int exec_process(const std::string& command,
                        const std::vector<std::string>& args,
                        const spawn_options& options,
                        const std::function<void(const char*)>& std_out)
{
    return exec_process(command, args, options, [&](const char* data, std::size_t) {
        std_out(data);
    });
}

static int exec_process(const std::string& command,
                        const std::vector<std::string>& args,
                        const spawn_options& options,
                        std::function<void(process::writer)> std_in)
{
    int pipefd[2] = {-1, -1};
    if(pipe(pipefd) != 0)
        MIGRAPHX_THROW("pipe() failed");

    spawn_options spawn_opts = options;
    spawn_opts.stdin_fd      = pipefd[0];

    const pid_t pid = spawn_process_async(command, args, spawn_opts);
    close(pipefd[0]);

    std_in([&](const char* buffer, std::size_t n) {
        ssize_t written = 0;
        while(written < static_cast<ssize_t>(n))
        {
            const auto rc =
                write(pipefd[1], buffer + written, n - static_cast<std::size_t>(written));
            if(rc < 0)
                MIGRAPHX_THROW("write() failed while sending data to child process");
            written += rc;
        }
    });
    close(pipefd[1]);

    return wait_for_pid(pid);
}

#else

constexpr std::size_t MIGRAPHX_PROCESS_BUFSIZE = 4096;

enum class direction
{
    input,
    output
};

template <direction dir>
class pipe
{
    public:
    explicit pipe()
    {
        SECURITY_ATTRIBUTES attrs;
        attrs.nLength              = sizeof(SECURITY_ATTRIBUTES);
        attrs.bInheritHandle       = TRUE;
        attrs.lpSecurityDescriptor = nullptr;

        if(CreatePipe(&m_read, &m_write, &attrs, 0) == FALSE)
            throw GetLastError();

        if constexpr(dir == direction::output)
        {
            // Do not inherit the read handle for the output pipe
            if(SetHandleInformation(m_read, HANDLE_FLAG_INHERIT, 0) == 0)
                throw GetLastError();
        }
        else
        {
            // Do not inherit the write handle for the input pipe
            if(SetHandleInformation(m_write, HANDLE_FLAG_INHERIT, 0) == 0)
                throw GetLastError();
        }
    }

    pipe(const pipe&)            = delete;
    pipe& operator=(const pipe&) = delete;

    pipe(pipe&&) = default;

    ~pipe()
    {
        if(m_write != nullptr)
        {
            CloseHandle(m_write);
        }
        if(m_read != nullptr)
        {
            CloseHandle(m_read);
        }
    }

    bool close_write_handle()
    {
        auto result = true;
        if(m_write != nullptr)
        {
            result  = CloseHandle(m_write) == TRUE;
            m_write = nullptr;
        }
        return result;
    }

    bool close_read_handle()
    {
        auto result = true;
        if(m_read != nullptr)
        {
            result = CloseHandle(m_read) == TRUE;
            m_read = nullptr;
        }
        return result;
    }

    std::pair<bool, DWORD> read(LPVOID buffer, DWORD length) const
    {
        DWORD bytes_read;
        if(ReadFile(m_read, buffer, length, &bytes_read, nullptr) == FALSE and
           GetLastError() == ERROR_MORE_DATA)
        {
            return {true, bytes_read};
        }
        return {false, bytes_read};
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

// clang-format off
template <typename F>
int exec(const std::string& cmd, const std::string& cwd, const std::string& args,
         const std::string& envs, F f)
// clang-format on
{
    if(enabled(MIGRAPHX_TRACE_CMD_EXECUTE{}))
    {
        std::cout << "[cwd=" << cwd << "];  cmd='" << cmd << "\'; args='" << args << "'; envs='"
                  << envs << "'\n";
    }

    // See CreateProcess() WIN32 documentation for details.
    constexpr std::size_t CMDLINE_LENGTH = 32767;

    // Build lpCommandLine parameter.
    std::string cmdline = quote_string(cmd);
    if(not args.empty())
        cmdline += " " + args;

    // clang-format off
    if(cmdline.size() > CMDLINE_LENGTH)
        MIGRAPHX_THROW("Command line too long, required maximum " +
                       std::to_string(CMDLINE_LENGTH) + " characters.");
    // clang-format on

    if(cmdline.size() < CMDLINE_LENGTH)
        cmdline.resize(CMDLINE_LENGTH, '\0');

    // Build lpEnvironment parameter.
    std::vector<TCHAR> environment{};
    if(not envs.empty())
    {
        std::istringstream iss{envs};
        std::string str;
        while(iss >> str)
        {
            environment.insert(environment.end(), str.begin(), str.end());
            environment.push_back('\0');
        }
        environment.push_back('\0');
    }

    try
    {
        STARTUPINFO info;
        PROCESS_INFORMATION process_info;

        pipe<direction::input> input{};
        pipe<direction::output> output{};

        ZeroMemory(&info, sizeof(STARTUPINFO));
        info.cb         = sizeof(STARTUPINFO);
        info.hStdError  = output.get_write_handle();
        info.hStdOutput = output.get_write_handle();
        info.hStdInput  = input.get_read_handle();
        info.dwFlags |= STARTF_USESTDHANDLES;
        info.wShowWindow = SW_HIDE;

        ZeroMemory(&process_info, sizeof(process_info));

        if(CreateProcess(cmd.c_str(),
                         cmdline.data(),
                         nullptr,
                         nullptr,
                         TRUE,
                         CREATE_NO_WINDOW,
                         environment.empty() ? nullptr : environment.data(),
                         cwd.empty() ? nullptr : static_cast<LPCSTR>(cwd.c_str()),
                         &info,
                         &process_info) == FALSE)
        {
            MIGRAPHX_THROW("Error creating process (" + std::to_string(GetLastError()) + ")");
        }

        CloseHandle(process_info.hThread);

        if(not output.close_write_handle())
            MIGRAPHX_THROW("Error closing STDOUT handle for writing (" +
                           std::to_string(GetLastError()) + ")");

        if(not input.close_read_handle())
            MIGRAPHX_THROW("Error closing STDIN handle for reading (" +
                           std::to_string(GetLastError()) + ")");

        f(input, output);

        if(not input.close_write_handle())
            MIGRAPHX_THROW("Error closing STDIN handle for writing (" +
                           std::to_string(GetLastError()) + ")");

        {
            TCHAR buf[MIGRAPHX_PROCESS_BUFSIZE];
            while(true)
            {
                DWORD available{};
                BOOL result = PeekNamedPipe(
                    output.get_read_handle(), nullptr, 0, nullptr, &available, nullptr);
                if(result == FALSE)
                    break;
                if(available == 0)
                {
                    if(WaitForSingleObject(process_info.hProcess, 0) == WAIT_OBJECT_0)
                        break;
                    Sleep(0);
                    continue;
                }
                while(available > 0)
                {
                    DWORD bytes_read{};
                    DWORD to_read = std::min<DWORD>(available, sizeof(buf));
                    result = ReadFile(output.get_read_handle(), buf, to_read, &bytes_read, nullptr);
                    WriteFile(GetStdHandle(STD_OUTPUT_HANDLE), buf, bytes_read, nullptr, nullptr);
                    available -= bytes_read;
                    if(result == FALSE or bytes_read == 0)
                        break;
                }
            }
        }

        WaitForSingleObject(process_info.hProcess, INFINITE);

        DWORD status{};
        GetExitCodeProcess(process_info.hProcess, &status);

        CloseHandle(process_info.hProcess);

        return static_cast<int>(status);
    }
    // cppcheck-suppress catchExceptionByValue
    catch(DWORD error)
    {
        MIGRAPHX_THROW("Error spawning process (" + std::to_string(error) + ")");
    }
}

// clang-format off
int exec(const std::string& cmd, const std::string& cwd, const std::string& args,
         const std::string& envs, HANDLE std_out)
{
    TCHAR buffer[MIGRAPHX_PROCESS_BUFSIZE];
    return (std_out == nullptr or std_out == INVALID_HANDLE_VALUE)
               ? GetLastError() : exec(cmd, cwd, args, envs,
                    [&](const pipe<direction::input>&, const pipe<direction::output>& out) {
                         for(;;)
                         {
                             auto [more_data, bytes_read] = out.read(buffer, MIGRAPHX_PROCESS_BUFSIZE);
                             if(bytes_read == 0)
                                 break;
                             if(WriteFile(std_out, buffer, bytes_read, nullptr, nullptr) == FALSE)
                                 break;
                             if(not more_data)
                                 break;
                         }
                    });
}

int exec(const std::string& cmd, const std::string& cwd, const std::string& args,
         const std::string& envs, std::function<void(process::writer)> std_in)
{
    return exec(cmd, cwd, args, envs,
        [&](const pipe<direction::input>& input, const pipe<direction::output>&) {
            std_in([&](const char* buffer, std::size_t n) { input.write(buffer, n); });
        });
}
// clang-format on

#endif

struct process_impl
{
    std::vector<std::string> args{};
    std::vector<std::string> envs{};
    std::string command{};
    fs::path cwd{};

    spawn_options spawn_opts() const
    {
        spawn_options opts{};
        if(not cwd.empty())
            opts.cwd = &cwd;
        if(not envs.empty())
            opts.env = &envs;
        return opts;
    }

    std::string describe() const
    {
        std::string result = command;
        for(const auto& arg : args)
            result += " " + arg;
        return result;
    }

    template <class... Ts>
    void check_exec(Ts&&... xs) const
    {
        if(enabled(MIGRAPHX_TRACE_CMD_EXECUTE{}))
        {
            std::cout << "command=" << describe();
            if(not cwd.empty())
                std::cout << " cwd=" << cwd.string();
            std::cout << std::endl;
        }
        int ec = migraphx::exec_process(std::forward<Ts>(xs)...);
        if(ec != 0)
            MIGRAPHX_THROW("Command " + describe() + " exited with status " + std::to_string(ec));
    }
};

process::process(const std::string& cmd, const std::vector<std::string>& args)
    : impl(std::make_unique<process_impl>())
{
    impl->command = cmd;
    impl->args    = args;
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

process& process::env(const std::vector<std::string>& envs)
{
    impl->envs = envs;
    return *this;
}

void process::read(const writer& output) const
{
#ifdef _WIN32
    // clang-format off
    constexpr std::string_view filename = "stdout";
    auto tmp = tmp_dir{};
    HANDLE handle = CreateFile((tmp.path / filename).string().c_str(),
                               GENERIC_READ | GENERIC_WRITE,
                               0,
                               nullptr,
                               CREATE_ALWAYS,
                               FILE_ATTRIBUTE_NORMAL,
                               nullptr);
    impl->check_exec(impl->command, impl->cwd.string(), impl->args, impl->envs,
                     handle == nullptr or handle == INVALID_HANDLE_VALUE ?
                                     GetStdHandle(STD_OUTPUT_HANDLE) : handle);
    CloseHandle(handle);
    handle = CreateFile((tmp.path / filename).string().c_str(),
                        GENERIC_READ | GENERIC_WRITE,
                        0,
                        nullptr,
                        OPEN_EXISTING,
                        FILE_ATTRIBUTE_NORMAL,
                        nullptr);
    if(handle == nullptr or handle == INVALID_HANDLE_VALUE)
        MIGRAPHX_THROW("Unable to open file: " + (tmp.path / filename));
    auto size = GetFileSize(handle, nullptr);
    std::string result(size, '\0');
    if(ReadFile(handle, result.data(), size, nullptr, nullptr) == FALSE)
        MIGRAPHX_THROW("Failed reading file: " + (tmp.path / filename));
    CloseHandle(handle);
    // clang-format on
#else
    std::stringstream ss;
    impl->check_exec(impl->command, impl->args, impl->spawn_opts(), redirect_to(ss));
    auto result = ss.str();
#endif
    output(result.data(), result.size());
}

void process::exec()
{
#ifndef _WIN32
    impl->check_exec(impl->command, impl->args, impl->spawn_opts(), redirect_to(std::cout));
#else
    // clang-format off
    impl->check_exec(impl->command, impl->cwd.string(), impl->args, impl->envs,
                     GetStdHandle(STD_OUTPUT_HANDLE));
    // clang-format on
#endif
}

void process::write(std::function<void(writer)> pipe_in)
{
#ifndef _WIN32
    impl->check_exec(impl->command, impl->args, impl->spawn_opts(), std::move(pipe_in));
#else
    // clang-format off
    impl->check_exec(impl->command, impl->cwd.string(),
                     impl->args, impl->envs, std::move(pipe_in));
    // clang-format on
#endif
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
