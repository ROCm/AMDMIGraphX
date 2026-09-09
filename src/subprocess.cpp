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

// Derived from rocFFT's shared/subprocess.h,
// Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved. (MIT)

#include <migraphx/subprocess.hpp>
#include <migraphx/env.hpp>
#include <migraphx/errors.hpp>
#include <migraphx/stringutils.hpp>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdio>
#include <iostream>

#ifdef _WIN32
// cppcheck-suppress definePrefix
#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#else
#include <fcntl.h>
#include <poll.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>
extern "C" char** environ; // NOLINT
#endif

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

namespace {

MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_TRACE_CMD_EXECUTE)

// Read back at most this much at a time from the child. Small on purpose: the point is to keep
// draining while we are still writing, not to minimize syscalls.
constexpr std::size_t read_chunk_size = 1024;

// Write at most a page at a time so a write to a full pipe cannot block.
constexpr std::size_t write_chunk_size = 4096;

void trace_command(const fs::path& exe, const std::vector<std::string>& argv)
{
    if(not enabled(MIGRAPHX_TRACE_CMD_EXECUTE{}))
        return;
    std::cout << exe.string();
    if(not argv.empty())
        std::cout << " " << join_strings(argv, " ");
    std::cout << std::endl;
}

#ifdef _WIN32

// RAII wrapper around a Win32 handle. Both NULL and INVALID_HANDLE_VALUE mean "nothing to close"
// here, since the two APIs we use disagree about which one they return on failure.
struct handle_wrapper
{
    handle_wrapper() = default;
    explicit handle_wrapper(HANDLE h) : handle(h) {}
    handle_wrapper(const handle_wrapper&)            = delete;
    handle_wrapper& operator=(const handle_wrapper&) = delete;
    ~handle_wrapper() { close(); }

    void close()
    {
        if(valid())
            CloseHandle(handle);
        handle = nullptr;
    }

    bool valid() const { return handle != nullptr and handle != INVALID_HANDLE_VALUE; }

    operator HANDLE() const { return handle; } // NOLINT

    HANDLE handle = nullptr;
};

[[noreturn]] void throw_last_error(const std::string& msg)
{
    MIGRAPHX_THROW(msg + " (" + std::to_string(GetLastError()) + ")");
}

// Anonymous pipes created by CreatePipe cannot be used for overlapped I/O, and we need overlapped
// I/O to read and write at the same time from one thread. So make a uniquely named pipe instead.
void make_overlapped_pipe(handle_wrapper& read, handle_wrapper& write)
{
    // The name only has to be unique among live pipes; both ends are closed when this function's
    // caller returns.
    std::array<char, 200> buffer{};
    std::snprintf(buffer.data(),
                  buffer.size(),
                  "\\\\.\\pipe\\migraphx_subprocess_%lx_%lx_%p",
                  GetCurrentProcessId(),
                  GetCurrentThreadId(),
                  static_cast<void*>(&read));
    const std::string pipe_name = buffer.data();

    SECURITY_ATTRIBUTES sa;
    sa.nLength              = sizeof(SECURITY_ATTRIBUTES);
    sa.bInheritHandle       = TRUE;
    sa.lpSecurityDescriptor = nullptr;

    constexpr DWORD pipe_size = 4096;
    read.handle               = CreateNamedPipeA(pipe_name.c_str(),
                                   PIPE_ACCESS_INBOUND | FILE_FLAG_OVERLAPPED,
                                   PIPE_TYPE_BYTE | PIPE_WAIT,
                                   1,
                                   pipe_size,
                                   pipe_size,
                                   0,
                                   &sa);
    if(not read.valid())
        throw_last_error("Failed to create read end of pipe");

    write.handle = CreateFileA(pipe_name.c_str(),
                               GENERIC_WRITE,
                               0,
                               &sa,
                               OPEN_EXISTING,
                               FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OVERLAPPED,
                               nullptr);
    if(not write.valid())
        throw_last_error("Failed to create write end of pipe");
}

std::string quote_arg(const std::string& arg)
{
    std::string result = "\"";
    for(auto c : arg)
    {
        if(c == '\\' or c == '"')
            result.push_back('\\');
        result.push_back(c);
    }
    result.push_back('"');
    return result;
}

// Restrict inheritance to exactly the handles the child needs. Without this, a CreateProcess on
// another thread can inherit our pipe ends, and then nobody ever sees EOF.
struct proc_thread_attribute_list
{
    explicit proc_thread_attribute_list(const std::vector<HANDLE>& inherited)
    {
        SIZE_T size = 0;
        // Deliberately ignored: this call always fails, it only reports the required size.
        InitializeProcThreadAttributeList(nullptr, 1, 0, &size);
        storage.resize(size);
        list = reinterpret_cast<LPPROC_THREAD_ATTRIBUTE_LIST>(storage.data());
        if(InitializeProcThreadAttributeList(list, 1, 0, &size) == FALSE)
        {
            list = nullptr;
            throw_last_error("Failed to initialize process attribute list");
        }
        if(UpdateProcThreadAttribute(list,
                                     0,
                                     PROC_THREAD_ATTRIBUTE_HANDLE_LIST,
                                     const_cast<HANDLE*>(inherited.data()), // NOLINT
                                     inherited.size() * sizeof(HANDLE),
                                     nullptr,
                                     nullptr) == FALSE)
        {
            throw_last_error("Failed to set inherited handle list");
        }
    }
    proc_thread_attribute_list(const proc_thread_attribute_list&)            = delete;
    proc_thread_attribute_list& operator=(const proc_thread_attribute_list&) = delete;
    ~proc_thread_attribute_list()
    {
        if(list != nullptr)
            DeleteProcThreadAttributeList(list);
    }

    std::vector<char> storage{};
    LPPROC_THREAD_ATTRIBUTE_LIST list = nullptr;
};

subprocess_result run(const fs::path& exe,
                      const std::vector<std::string>& argv,
                      const std::vector<char>& stdin_data)
{
    handle_wrapper child_stdin_read;
    handle_wrapper child_stdin_write;
    handle_wrapper child_stdout_read;
    handle_wrapper child_stdout_write;
    make_overlapped_pipe(child_stdin_read, child_stdin_write);
    make_overlapped_pipe(child_stdout_read, child_stdout_write);

    // The child has no use for our ends, and letting it hold them open would keep us from ever
    // seeing EOF.
    if(SetHandleInformation(child_stdin_write, HANDLE_FLAG_INHERIT, 0) == FALSE)
        throw_last_error("Failed to uninherit stdin write handle");
    if(SetHandleInformation(child_stdout_read, HANDLE_FLAG_INHERIT, 0) == FALSE)
        throw_last_error("Failed to uninherit stdout read handle");

    // Hand our own stderr to the child so its diagnostics land wherever ours do.
    HANDLE stderr_handle = GetStdHandle(STD_ERROR_HANDLE);
    if(stderr_handle == INVALID_HANDLE_VALUE)
        stderr_handle = nullptr;
    if(stderr_handle != nullptr)
        SetHandleInformation(stderr_handle, HANDLE_FLAG_INHERIT, HANDLE_FLAG_INHERIT);

    std::vector<HANDLE> inherited = {child_stdin_read.handle, child_stdout_write.handle};
    if(stderr_handle != nullptr)
        inherited.push_back(stderr_handle);
    proc_thread_attribute_list attributes{inherited};

    STARTUPINFOEXA info         = {};
    info.StartupInfo.cb         = sizeof(STARTUPINFOEXA);
    info.StartupInfo.dwFlags    = STARTF_USESTDHANDLES;
    info.StartupInfo.hStdInput  = child_stdin_read;
    info.StartupInfo.hStdOutput = child_stdout_write;
    info.StartupInfo.hStdError  = stderr_handle;
    info.lpAttributeList        = attributes.list;

    const std::string exe_string = exe.string();
    std::string cmdline          = quote_arg(exe_string);
    for(const auto& arg : argv)
        cmdline += " " + quote_arg(arg);

    PROCESS_INFORMATION process_info = {};
    if(CreateProcessA(exe_string.c_str(),
                      cmdline.data(),
                      nullptr,
                      nullptr,
                      TRUE,
                      EXTENDED_STARTUPINFO_PRESENT | CREATE_NO_WINDOW,
                      nullptr,
                      nullptr,
                      &info.StartupInfo,
                      &process_info) == FALSE)
    {
        throw_last_error("Failed to create process " + exe_string);
    }
    handle_wrapper process_handle{process_info.hProcess};
    handle_wrapper thread_handle{process_info.hThread};

    // Now that the child holds its own copies, drop ours. Otherwise the child never sees EOF on
    // stdin and we never see EOF on stdout.
    child_stdin_read.close();
    child_stdout_write.close();

    handle_wrapper stdin_write_event{CreateEventA(nullptr, TRUE, FALSE, nullptr)};
    handle_wrapper stdout_read_event{CreateEventA(nullptr, TRUE, FALSE, nullptr)};
    if(not stdin_write_event.valid() or not stdout_read_event.valid())
        throw_last_error("Failed to create overlapped I/O event");

    OVERLAPPED stdin_write_overlapped = {};
    stdin_write_overlapped.hEvent     = stdin_write_event;
    OVERLAPPED stdout_read_overlapped = {};
    stdout_read_overlapped.hEvent     = stdout_read_event;

    std::size_t total_bytes_written = 0;
    std::size_t total_bytes_read    = 0;
    std::vector<char> stdout_data(read_chunk_size);

    // stdout is deliberately first: WaitForMultipleObjects reports the lowest signalled index, so a
    // child that writes its output and immediately exits still gets drained before we notice it
    // died.
    std::array<HANDLE, 3> handles{stdout_read_event, stdin_write_event, process_handle};
    DWORD handle_count = 3;
    bool writing       = not stdin_data.empty();

    // Signal EOF to the child and drop the write event, which can no longer fire. The process
    // handle takes its slot so we still notice the child dying.
    auto finish_writing = [&] {
        child_stdin_write.close();
        handles[1]   = process_handle;
        handle_count = 2;
        writing      = false;
    };

    if(writing)
    {
        if(WriteFile(child_stdin_write,
                     stdin_data.data(),
                     static_cast<DWORD>(std::min(stdin_data.size(), write_chunk_size)),
                     nullptr,
                     &stdin_write_overlapped) == FALSE and
           GetLastError() != ERROR_IO_PENDING)
        {
            throw_last_error("Failed to write to child stdin");
        }
    }
    else
    {
        finish_writing();
    }

    if(ReadFile(child_stdout_read,
                stdout_data.data(),
                read_chunk_size,
                nullptr,
                &stdout_read_overlapped) == FALSE and
       GetLastError() != ERROR_IO_PENDING)
    {
        throw_last_error("Failed to read from child stdout");
    }

    for(;;)
    {
        auto wait_result = WaitForMultipleObjects(handle_count, handles.data(), FALSE, INFINITE);
        if(wait_result == WAIT_OBJECT_0)
        {
            DWORD bytes_read = 0;
            if(GetOverlappedResult(child_stdout_read, &stdout_read_overlapped, &bytes_read, FALSE) ==
               TRUE)
            {
                total_bytes_read += bytes_read;
            }
            else if(GetLastError() == ERROR_HANDLE_EOF or GetLastError() == ERROR_BROKEN_PIPE)
            {
                stdout_data.resize(total_bytes_read);
                break;
            }

            // Grow before issuing the read: the buffer must not move while the read is pending.
            stdout_data.resize(total_bytes_read + read_chunk_size);
            if(ReadFile(child_stdout_read,
                        stdout_data.data() + total_bytes_read,
                        read_chunk_size,
                        nullptr,
                        &stdout_read_overlapped) == FALSE)
            {
                auto error = GetLastError();
                if(error == ERROR_BROKEN_PIPE or error == ERROR_HANDLE_EOF)
                {
                    stdout_data.resize(total_bytes_read);
                    break;
                }
                if(error != ERROR_IO_PENDING)
                    throw_last_error("Failed to read from child stdout");
            }
        }
        else if(writing and wait_result == WAIT_OBJECT_0 + 1)
        {
            DWORD bytes_written = 0;
            if(GetOverlappedResult(
                   child_stdin_write, &stdin_write_overlapped, &bytes_written, FALSE) == TRUE)
            {
                total_bytes_written += bytes_written;
            }

            if(total_bytes_written >= stdin_data.size())
            {
                finish_writing();
            }
            else
            {
                auto remaining = stdin_data.size() - total_bytes_written;
                if(WriteFile(child_stdin_write,
                             stdin_data.data() + total_bytes_written,
                             static_cast<DWORD>(std::min(remaining, write_chunk_size)),
                             nullptr,
                             &stdin_write_overlapped) == FALSE and
                   GetLastError() != ERROR_IO_PENDING)
                {
                    throw_last_error("Failed to write to child stdin");
                }
            }
        }
        else
        {
            // The child exited with nothing left to read, or the wait failed.
            stdout_data.resize(total_bytes_read);
            break;
        }
    }
    child_stdout_read.close();

    if(WaitForSingleObject(process_handle, INFINITE) != WAIT_OBJECT_0)
        throw_last_error("Failed to wait for child process");

    DWORD exit_code = 0;
    if(GetExitCodeProcess(process_handle, &exit_code) == FALSE)
        throw_last_error("Failed to get child exit code");

    return {static_cast<int>(exit_code), std::move(stdout_data)};
}

#else

// RAII wrapper around a file descriptor.
struct fd_wrapper
{
    fd_wrapper() = default;
    explicit fd_wrapper(int f) : fd(f) {}
    fd_wrapper(const fd_wrapper&)            = delete;
    fd_wrapper& operator=(const fd_wrapper&) = delete;
    ~fd_wrapper() { close(); }

    void close()
    {
        if(fd != -1)
            ::close(fd);
        fd = -1;
    }

    operator int() const { return fd; } // NOLINT

    int fd = -1;
};

subprocess_result run(const fs::path& exe,
                      const std::vector<std::string>& argv,
                      const std::vector<char>& stdin_data)
{
    std::array<int, 2> stdin_fds{-1, -1};
    if(pipe2(stdin_fds.data(), O_CLOEXEC) != 0)
        MIGRAPHX_THROW("Failed to create stdin pipe");
    fd_wrapper child_stdin_read{stdin_fds[0]};
    fd_wrapper child_stdin_write{stdin_fds[1]};

    std::array<int, 2> stdout_fds{-1, -1};
    if(pipe2(stdout_fds.data(), O_CLOEXEC) != 0)
        MIGRAPHX_THROW("Failed to create stdout pipe");
    fd_wrapper child_stdout_read{stdout_fds[0]};
    fd_wrapper child_stdout_write{stdout_fds[1]};

    const std::string exe_string = exe.string();
    std::vector<const char*> child_argv;
    child_argv.reserve(argv.size() + 2);
    child_argv.push_back(exe_string.c_str());
    std::transform(argv.begin(),
                   argv.end(),
                   std::back_inserter(child_argv),
                   [](const std::string& arg) { return arg.c_str(); });
    child_argv.push_back(nullptr);

    // dup2 clears O_CLOEXEC on the target, so the child keeps exactly these two descriptors and
    // inherits stderr untouched. Use posix_spawn rather than fork: the parent has the HIP runtime
    // loaded and many threads running, and fork in that state is a hazard.
    posix_spawn_file_actions_t file_actions;
    posix_spawn_file_actions_init(&file_actions);
    posix_spawn_file_actions_adddup2(&file_actions, child_stdin_read, STDIN_FILENO);
    posix_spawn_file_actions_adddup2(&file_actions, child_stdout_write, STDOUT_FILENO);

    pid_t pid        = 0;
    int spawn_result = posix_spawn(&pid,
                                   exe_string.c_str(),
                                   &file_actions,
                                   nullptr,
                                   const_cast<char* const*>(child_argv.data()), // NOLINT
                                   environ);
    posix_spawn_file_actions_destroy(&file_actions);
    if(spawn_result != 0)
        MIGRAPHX_THROW("Failed to spawn process " + exe_string);

    // Drop our copies of the child's ends so EOF propagates in both directions.
    child_stdin_read.close();
    child_stdout_write.close();

    std::array<pollfd, 2> fds{};
    fds[0].fd     = child_stdin_write;
    fds[0].events = POLLOUT;
    fds[1].fd     = child_stdout_read;
    fds[1].events = POLLIN;

    if(stdin_data.empty())
    {
        // Nothing to send, so signal EOF right away. A negative fd is ignored by poll.
        child_stdin_write.close();
        fds[0].fd = -1;
    }

    std::size_t total_bytes_written = 0;
    std::vector<char> stdout_data;
    for(;;)
    {
        if(poll(fds.data(), static_cast<nfds_t>(fds.size()), -1) < 0)
        {
            if(errno == EINTR)
                continue;
            MIGRAPHX_THROW("Failed to poll child pipes for " + exe_string);
        }

        if((fds[0].revents & POLLERR) != 0 or (fds[1].revents & POLLERR) != 0)
            break;

        if((fds[0].revents & POLLOUT) != 0)
        {
            auto remaining = stdin_data.size() - total_bytes_written;
            auto written   = write(child_stdin_write,
                                 stdin_data.data() + total_bytes_written,
                                 std::min(remaining, write_chunk_size));
            if(written < 0 and errno == EINTR)
                continue;
            if(written <= 0)
                break;
            total_bytes_written += static_cast<std::size_t>(written);

            if(total_bytes_written >= stdin_data.size())
            {
                // Close the child's stdin so it knows we are done writing.
                child_stdin_write.close();
                fds[0].fd = -1;
            }
        }

        if((fds[1].revents & POLLIN) != 0)
        {
            auto offset = stdout_data.size();
            stdout_data.resize(offset + read_chunk_size);
            auto bytes_read = read(child_stdout_read, stdout_data.data() + offset, read_chunk_size);
            stdout_data.resize(offset + (bytes_read > 0 ? static_cast<std::size_t>(bytes_read) : 0));
            if(bytes_read < 0)
            {
                if(errno == EINTR)
                    continue;
                MIGRAPHX_THROW("Failed to read from child stdout");
            }
            // A zero-length read on a pipe means every write end is closed.
            if(bytes_read == 0)
                break;
        }
        else if((fds[1].revents & POLLHUP) != 0)
        {
            break;
        }
    }
    child_stdout_read.close();

    int wait_status = 0;
    if(waitpid(pid, &wait_status, 0) != pid)
        MIGRAPHX_THROW("Failed to wait for child process " + exe_string);

    // Report a signalled child as a generic failure; the caller only needs "did it work".
    int exit_code = WIFSIGNALED(wait_status) ? -1 : WEXITSTATUS(wait_status); // NOLINT
    return {exit_code, std::move(stdout_data)};
}

#endif

} // namespace

subprocess_result execute_subprocess(const fs::path& exe,
                                     const std::vector<std::string>& argv,
                                     const std::vector<char>& stdin_data)
{
    trace_command(exe, argv);
    return run(exe, argv, stdin_data);
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
