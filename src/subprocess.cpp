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

// Derived from rocFFT's shared/subprocess.h (Copyright (C) 2024 Advanced Micro Devices, Inc., MIT).

#include <migraphx/subprocess.hpp>
#include <migraphx/env.hpp>
#include <migraphx/errors.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/stringutils.hpp>

#include <algorithm>
#include <array>
#include <cassert>
#include <iostream>
#include <limits>

#ifdef _WIN32
// cppcheck-suppress definePrefix
#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#include <cstdint>
#else
#include <cerrno>
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

// Big enough that a multi-megabyte payload does not cost thousands of syscalls. Neither a POSIX
// read() after POLLIN nor an overlapped ReadFile blocks waiting to fill this, so size does not
// affect how promptly we drain.
constexpr std::size_t read_chunk_size = 64 * 1024;

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

// RAII wrapper around a Win32 handle. Both NULL and INVALID_HANDLE_VALUE count as "nothing to
// close": CreateNamedPipeA and CreateFileA return the latter on failure, CreateEventA the former.
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

// Anonymous pipes created by CreatePipe cannot do overlapped I/O, and we need overlapped I/O to
// read and write at the same time from one thread. So make a uniquely named pipe instead.
void make_overlapped_pipe(handle_wrapper& read, handle_wrapper& write)
{
    // Unique among live pipes only: pid + tid + the address of a caller-owned handle that outlives
    // the pipe.
    const std::string pipe_name =
        "\\\\.\\pipe\\migraphx_subprocess_" + std::to_string(GetCurrentProcessId()) + "_" +
        std::to_string(GetCurrentThreadId()) + "_" +
        std::to_string(reinterpret_cast<std::uintptr_t>(&read)); // NOLINT

    SECURITY_ATTRIBUTES sa;
    sa.nLength              = sizeof(SECURITY_ATTRIBUTES);
    sa.bInheritHandle       = TRUE;
    sa.lpSecurityDescriptor = nullptr;

    // Sized to match read_chunk_size: a smaller kernel buffer would throttle a multi-megabyte
    // transfer into many producer/consumer handoffs no matter how much we ask for per read.
    constexpr DWORD pipe_size = read_chunk_size;
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

// Quote one argument for CommandLineToArgvW, which the child's CRT uses to rebuild argv. A
// backslash is literal unless it is part of a run immediately preceding a quote, so only those runs
// get doubled -- escaping every backslash would turn C:\foo into C:\\foo on the far side.
std::string quote_arg(const std::string& arg)
{
    std::string result = "\"";
    std::size_t slashes = 0;
    for(auto c : arg)
    {
        if(c == '\\')
        {
            slashes++;
        }
        else
        {
            if(c == '"')
                result.append(slashes + 1, '\\');
            slashes = 0;
        }
        result.push_back(c);
    }
    // The closing quote also terminates a trailing backslash run.
    result.append(slashes, '\\');
    result.push_back('"');
    return result;
}

// Restrict inheritance to exactly the handles the child needs. Without this, a CreateProcess on
// another thread can inherit our pipe ends, and then nobody ever sees EOF.
struct proc_thread_attribute_list
{
    explicit proc_thread_attribute_list(std::vector<HANDLE>& inherited)
    {
        SIZE_T size = 0;
        // Always fails; it only reports the required size.
        InitializeProcThreadAttributeList(nullptr, 1, 0, &size);
        // std::allocator<char> goes through ::operator new, which is aligned for any fundamental
        // type, so the attribute list is suitably aligned.
        storage.resize(size);
        auto* attributes = reinterpret_cast<LPPROC_THREAD_ATTRIBUTE_LIST>(storage.data()); // NOLINT
        if(InitializeProcThreadAttributeList(attributes, 1, 0, &size) == FALSE)
            throw_last_error("Failed to initialize process attribute list");
        list = attributes;
        if(UpdateProcThreadAttribute(list,
                                     0,
                                     PROC_THREAD_ATTRIBUTE_HANDLE_LIST,
                                     inherited.data(),
                                     inherited.size() * sizeof(HANDLE),
                                     nullptr,
                                     nullptr) == FALSE)
        {
            // The destructor does not run for a constructor that throws.
            DeleteProcThreadAttributeList(list);
            list = nullptr;
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

    // Keep our own ends out of the child: a stray duplicate of the stdin write end would stop the
    // child from ever seeing EOF on stdin, and one of the stdout read end would let it steal our
    // output.
    if(SetHandleInformation(child_stdin_write, HANDLE_FLAG_INHERIT, 0) == FALSE)
        throw_last_error("Failed to uninherit stdin write handle");
    if(SetHandleInformation(child_stdout_read, HANDLE_FLAG_INHERIT, 0) == FALSE)
        throw_last_error("Failed to uninherit stdout read handle");

    std::vector<HANDLE> inherited = {child_stdin_read.handle, child_stdout_write.handle};

    // Hand our own stderr to the child so its diagnostics land wherever ours do.
    HANDLE stderr_handle = GetStdHandle(STD_ERROR_HANDLE);
    if(stderr_handle == INVALID_HANDLE_VALUE)
        stderr_handle = nullptr;
    if(stderr_handle != nullptr)
    {
        SetHandleInformation(stderr_handle, HANDLE_FLAG_INHERIT, HANDLE_FLAG_INHERIT);
        inherited.push_back(stderr_handle);
    }
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

    // The child has its own copies now. Ours must go, or we never see EOF on stdout.
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
    std::vector<char> stdout_data;

    // stdout is deliberately first: WaitForMultipleObjects reports the lowest signalled index, so a
    // child that writes its output and immediately exits still gets drained before we notice it
    // died. Slot 1 holds the write event only while we still have data to send.
    std::array<HANDLE, 3> handles{stdout_read_event, stdin_write_event, process_handle};
    bool writing = not stdin_data.empty();

    // Signal EOF to the child and give the write event's slot to the process handle, so we still
    // notice the child dying once there is nothing left to send.
    auto finish_writing = [&] {
        child_stdin_write.close();
        handles[1] = process_handle;
        writing    = false;
    };

    // Overlapped writes never block on a full pipe, so there is no reason to chunk: hand the whole
    // remainder over and let it complete as the child drains.
    auto issue_write = [&] {
        assert(total_bytes_written < stdin_data.size());
        auto remaining = stdin_data.size() - total_bytes_written;
        if(WriteFile(child_stdin_write,
                     stdin_data.data() + total_bytes_written,
                     static_cast<DWORD>(remaining),
                     nullptr,
                     &stdin_write_overlapped) == FALSE and
           GetLastError() != ERROR_IO_PENDING)
        {
            throw_last_error("Failed to write to child stdin");
        }
    };

    // Returns false once the child has closed stdout.
    auto issue_read = [&] {
        // Grow before issuing: the buffer must not move while a read is pending.
        stdout_data.resize(total_bytes_read + read_chunk_size);
        if(ReadFile(child_stdout_read,
                    stdout_data.data() + total_bytes_read,
                    read_chunk_size,
                    nullptr,
                    &stdout_read_overlapped) != FALSE)
        {
            return true;
        }
        auto error = GetLastError();
        if(error == ERROR_BROKEN_PIPE or error == ERROR_HANDLE_EOF)
            return false;
        if(error != ERROR_IO_PENDING)
            throw_last_error("Failed to read from child stdout");
        return true;
    };

    if(writing)
        issue_write();
    else
        finish_writing();

    bool reading = issue_read();
    while(reading)
    {
        auto wait_result =
            WaitForMultipleObjects(writing ? 3 : 2, handles.data(), FALSE, INFINITE);
        if(wait_result == WAIT_OBJECT_0)
        {
            DWORD bytes_read = 0;
            if(GetOverlappedResult(child_stdout_read, &stdout_read_overlapped, &bytes_read, FALSE) ==
               FALSE)
            {
                auto error = GetLastError();
                // Anything else means the read is still in flight, so we must not reissue it: the
                // resize inside issue_read could move the buffer out from under the kernel.
                if(error != ERROR_HANDLE_EOF and error != ERROR_BROKEN_PIPE)
                    throw_last_error("Failed to complete read from child stdout");
                break;
            }
            total_bytes_read += bytes_read;
            reading = issue_read();
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
                finish_writing();
            else
                issue_write();
        }
        else
        {
            // The child exited with nothing left to read, or the wait failed.
            break;
        }
    }
    stdout_data.resize(total_bytes_read);
    child_stdout_read.close();

    if(WaitForSingleObject(process_handle, INFINITE) != WAIT_OBJECT_0)
        throw_last_error("Failed to wait for child process");

    DWORD exit_code = 0;
    if(GetExitCodeProcess(process_handle, &exit_code) == FALSE)
        throw_last_error("Failed to get child exit code");

    // An SEH-terminated child exits with something like 0xC0000005, which does not fit an int.
    // Report those the same way POSIX reports a signalled child.
    if(exit_code > static_cast<DWORD>(std::numeric_limits<int>::max()))
        return {-1, std::move(stdout_data)};
    return {static_cast<int>(exit_code), std::move(stdout_data)};
}

#else

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

// Reaps the child on every path out of run(), so a throw between posix_spawn and the explicit wait
// does not leave a zombie behind.
struct child_reaper
{
    explicit child_reaper(pid_t p) : pid(p) {}
    child_reaper(const child_reaper&)            = delete;
    child_reaper& operator=(const child_reaper&) = delete;
    ~child_reaper()
    {
        if(pid > 0)
        {
            int status = 0;
            waitpid(pid, &status, 0);
        }
    }

    int wait()
    {
        int status  = 0;
        auto result = waitpid(pid, &status, 0);
        auto waited = pid;
        pid         = -1;
        if(result != waited)
            MIGRAPHX_THROW("Failed to wait for child process");
        return status;
    }

    pid_t pid;
};

// POSIX guarantees a page can be written without blocking once poll() reports POLLOUT, and these
// fds are blocking, so cap each write there.
constexpr std::size_t write_chunk_size = 4096;

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
    migraphx::transform(
        argv, std::back_inserter(child_argv), [](const std::string& arg) { return arg.c_str(); });
    child_argv.push_back(nullptr);

    // posix_spawn's dup2 action clears O_CLOEXEC on the duplicate, so the child gets these as its
    // stdin/stdout while our other fds stay O_CLOEXEC; stderr is inherited untouched. Use
    // posix_spawn rather than fork: the parent has the HIP runtime loaded and many threads running,
    // and fork in that state is a hazard.
    posix_spawn_file_actions_t file_actions;
    posix_spawn_file_actions_init(&file_actions);
    int action_result =
        posix_spawn_file_actions_adddup2(&file_actions, child_stdin_read, STDIN_FILENO) |
        posix_spawn_file_actions_adddup2(&file_actions, child_stdout_write, STDOUT_FILENO);

    pid_t pid        = 0;
    int spawn_result = action_result;
    if(spawn_result == 0)
    {
        // posix_spawn does not modify argv, but its signature is not const-correct.
        spawn_result = posix_spawn(&pid,
                                   exe_string.c_str(),
                                   &file_actions,
                                   nullptr,
                                   const_cast<char* const*>(child_argv.data()), // NOLINT
                                   environ);
    }
    posix_spawn_file_actions_destroy(&file_actions);
    if(spawn_result != 0)
        MIGRAPHX_THROW("Failed to spawn process " + exe_string);
    child_reaper child{pid};

    // Drop our copies of the child's ends, or stdout never reports EOF and a dead child never shows
    // up as an error on the write side.
    child_stdin_read.close();
    child_stdout_write.close();

    std::array<pollfd, 2> fds{};
    fds[0].fd     = child_stdin_write;
    fds[0].events = POLLOUT;
    fds[1].fd     = child_stdout_read;
    fds[1].events = POLLIN;

    if(stdin_data.empty())
    {
        // Nothing to send, so let the child see EOF right away. A negative fd is ignored by poll.
        child_stdin_write.close();
        fds[0].fd = -1;
    }

    std::size_t total_bytes_written = 0;
    std::vector<char> stdout_data;
    std::vector<char> buffer(read_chunk_size);
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
            assert(total_bytes_written < stdin_data.size());
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
                // Done writing: close our end so the child sees EOF.
                child_stdin_write.close();
                fds[0].fd = -1;
            }
        }

        if((fds[1].revents & POLLIN) != 0)
        {
            auto bytes_read = read(child_stdout_read, buffer.data(), buffer.size());
            if(bytes_read < 0)
            {
                if(errno == EINTR)
                    continue;
                MIGRAPHX_THROW("Failed to read from child stdout");
            }
            // A zero-length read on a pipe means every write end is closed.
            if(bytes_read == 0)
                break;
            stdout_data.insert(stdout_data.end(), buffer.data(), buffer.data() + bytes_read);
        }
        else if((fds[1].revents & POLLHUP) != 0)
        {
            break;
        }
    }
    child_stdout_read.close();

    int wait_status = child.wait();
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
