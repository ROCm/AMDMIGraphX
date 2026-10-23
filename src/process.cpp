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
// These must precede every include: the Windows SDK headers latch _WIN32_WINNT the first time one
// of them is reached, and NOMINMAX only suppresses the min/max macros if it gets there before
// Windows.h does.
#ifdef _WIN32
// STARTUPINFOEX and the process-thread attribute list are Vista+. Guarded rather than forced, so a
// build that asks for a newer target keeps it.
#ifndef _WIN32_WINNT
// cppcheck-suppress definePrefix
#define _WIN32_WINNT 0x0600
#endif
// Windows.h defines min/max as macros, which breaks std::numeric_limits<int>::max().
#ifndef NOMINMAX
// cppcheck-suppress definePrefix
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
// cppcheck-suppress definePrefix
#define WIN32_LEAN_AND_MEAN
#endif
#endif

#include <migraphx/env.hpp>
#include <migraphx/errors.hpp>
#include <migraphx/process.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/tmp_dir.hpp>
#include <migraphx/fileutils.hpp>
#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <limits>
#include <numeric>
#include <functional>
#include <iostream>

#ifdef _WIN32
#include <Windows.h>
#include <cstdint>
#include <cstring>
#include <sstream>
#include <optional>
#else
#include <cerrno>
#include <csignal>
#include <fcntl.h>
#include <poll.h>
#include <pthread.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>
extern "C" char** environ; // NOLINT
#endif

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_TRACE_CMD_EXECUTE)

#ifdef _WIN32

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

// ---------------------------------------------------------------------------------------------
// Direct spawning, and the bidirectional exec backing process::read_write.
//
// On POSIX every process method spawns the command itself rather than through a shell, so nothing
// in the command, its arguments, its cwd, or its environment is ever interpreted. exec, read, and
// write each need only a single pipe, as do the Windows exec helpers above.
//
// read_write must pump both directions at once: a request larger than the 64 KB pipe buffer
// deadlocks if write and read are sequenced, and hipRTC compile requests are an order of magnitude
// larger than that. Both pipes are driven from one thread, with overlapped I/O on Windows and poll()
// on POSIX.
//
// Derived from rocFFT's shared/subprocess.h (Copyright (C) 2024 Advanced Micro Devices, Inc., MIT).
// ---------------------------------------------------------------------------------------------

namespace {

struct exec_result
{
    /// The child's exit status, or -1 if it did not exit normally at all -- killed by a signal on
    /// POSIX, terminated by an SEH exception on Windows.
    int exit_code = 0;
    std::vector<char> stdout_data{};
};

// Big enough that a multi-megabyte payload does not cost thousands of syscalls. Neither a POSIX
// read() after POLLIN nor an overlapped ReadFile blocks waiting to fill this, so size does not
// affect how promptly we drain.
constexpr std::size_t read_chunk_size = 64 * 1024;

void trace_command(const std::string& command)
{
    if(not enabled(MIGRAPHX_TRACE_CMD_EXECUTE{}))
        return;
    // stderr, not stdout: a caller may be redirecting stdout to collect real output.
    std::cerr << command << std::endl;
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

    HANDLE get() const { return handle; }

    HANDLE handle = nullptr;
};

// Takes the error code as an argument so callers capture it as the first statement after the API
// that failed; building the message allocates, and an allocation can clobber the thread's last
// error.
[[noreturn]] void throw_error(DWORD error, const std::string& msg)
{
    MIGRAPHX_THROW(msg + " (" + std::to_string(error) + ")");
}

// Anonymous pipes created by CreatePipe cannot do overlapped I/O, and we need overlapped I/O to
// read and write at the same time from one thread. So make a uniquely named pipe instead. Only the
// end we keep is asynchronous: the child does synchronous CRT I/O on its end, which is unsupported
// on a FILE_FLAG_OVERLAPPED handle.
void make_pipe(handle_wrapper& read, handle_wrapper& write, bool async_read)
{
    // Unique among live pipes only: pid + tid + the address of a caller-owned handle that outlives
    // the pipe.
    const std::string pipe_name = "\\\\.\\pipe\\migraphx_process_" +
                                  std::to_string(GetCurrentProcessId()) + "_" +
                                  std::to_string(GetCurrentThreadId()) + "_" +
                                  std::to_string(reinterpret_cast<std::uintptr_t>(&read)); // NOLINT

    SECURITY_ATTRIBUTES sa{};
    sa.nLength              = sizeof(SECURITY_ATTRIBUTES);
    sa.bInheritHandle       = TRUE;
    sa.lpSecurityDescriptor = nullptr;

    // Sized to match read_chunk_size: a smaller kernel buffer would throttle a multi-megabyte
    // transfer into many producer/consumer handoffs no matter how much we ask for per read.
    constexpr DWORD pipe_size = read_chunk_size;
    // FIRST_PIPE_INSTANCE so a local process that guessed the name cannot pre-create it and
    // become the peer; REJECT_REMOTE_CLIENTS so the name is not reachable over SMB.
    DWORD open_mode = PIPE_ACCESS_INBOUND | FILE_FLAG_FIRST_PIPE_INSTANCE;
    if(async_read)
        open_mode |= FILE_FLAG_OVERLAPPED;
    read.handle = CreateNamedPipeA(pipe_name.c_str(),
                                   open_mode,
                                   PIPE_TYPE_BYTE | PIPE_WAIT | PIPE_REJECT_REMOTE_CLIENTS,
                                   1,
                                   pipe_size,
                                   pipe_size,
                                   0,
                                   &sa);
    if(not read.valid())
        throw_error(GetLastError(), "Failed to create read end of pipe");

    DWORD flags = FILE_ATTRIBUTE_NORMAL;
    if(not async_read)
        flags |= FILE_FLAG_OVERLAPPED;
    write.handle =
        CreateFileA(pipe_name.c_str(), GENERIC_WRITE, 0, &sa, OPEN_EXISTING, flags, nullptr);
    if(not write.valid())
        throw_error(GetLastError(), "Failed to create write end of pipe");
}

// Quote one argument for CommandLineToArgvW, which the child's CRT uses to rebuild argv. A
// backslash is literal unless it is part of a run immediately preceding a quote, so only those runs
// get doubled -- escaping every backslash would turn C:\foo into C:\\foo on the far side.
std::string quote_arg(const std::string& arg)
{
    std::string result  = "\"";
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
        assert(size > 0);
        // Held as max_align_t rather than char so the alignment the cast below needs is guaranteed
        // by the element type instead of by whatever std::allocator<char> happens to do.
        storage.resize((size + sizeof(std::max_align_t) - 1) / sizeof(std::max_align_t));
        auto* attributes = reinterpret_cast<LPPROC_THREAD_ATTRIBUTE_LIST>(storage.data()); // NOLINT
        if(InitializeProcThreadAttributeList(attributes, 1, 0, &size) == FALSE)
            throw_error(GetLastError(), "Failed to initialize process attribute list");
        list = attributes;
        if(UpdateProcThreadAttribute(list,
                                     0,
                                     PROC_THREAD_ATTRIBUTE_HANDLE_LIST,
                                     inherited.data(),
                                     inherited.size() * sizeof(HANDLE),
                                     nullptr,
                                     nullptr) == FALSE)
        {
            auto error = GetLastError();
            // The destructor does not run for a constructor that throws.
            DeleteProcThreadAttributeList(list);
            list = nullptr;
            throw_error(error, "Failed to set inherited handle list");
        }
    }
    proc_thread_attribute_list(const proc_thread_attribute_list&)            = delete;
    proc_thread_attribute_list& operator=(const proc_thread_attribute_list&) = delete;
    ~proc_thread_attribute_list()
    {
        if(list != nullptr)
            DeleteProcThreadAttributeList(list);
    }

    std::vector<std::max_align_t> storage{};
    LPPROC_THREAD_ATTRIBUTE_LIST list = nullptr;
};

// Cancels a pending overlapped operation and waits for the kernel to release its buffer. Without
// this, an exception unwinding out of exec_read_write frees memory the kernel is still writing
// into.
struct pending_io
{
    pending_io(HANDLE h, OVERLAPPED& o) : handle(h), overlapped(&o) {}
    pending_io(const pending_io&)            = delete;
    pending_io& operator=(const pending_io&) = delete;
    ~pending_io() { cancel(); }

    void arm() { active = true; }

    // The operation completed, so there is nothing left to cancel.
    void disarm() { active = false; }

    void cancel()
    {
        if(not active)
            return;
        active = false;
        CancelIoEx(handle, overlapped);
        DWORD transferred = 0;
        // Blocking wait: the buffer cannot be released until the kernel is done with it.
        GetOverlappedResult(handle, overlapped, &transferred, TRUE);
    }

    HANDLE handle;
    OVERLAPPED* overlapped;
    bool active = false;
};

// Kills the child if exec_read_write exits without a clean wait, so a throw does not leave a
// compiler process running against pipes nobody holds.
struct child_guard
{
    explicit child_guard(HANDLE h) : handle(h) {}
    child_guard(const child_guard&)            = delete;
    child_guard& operator=(const child_guard&) = delete;
    ~child_guard()
    {
        if(not armed)
            return;
        TerminateProcess(handle, 1);
        WaitForSingleObject(handle, INFINITE);
    }

    void release() { armed = false; }

    HANDLE handle;
    bool armed = true;
};

exec_result exec_read_write(const std::string& cmd,
                            const std::vector<std::string>& argv,
                            const std::vector<char>& stdin_data)
{
    // Declaration order below is load-bearing. Locals destruct in reverse, and the required order
    // is: cancel pending I/O, kill the child, close handles, then release the buffers the kernel
    // was pointed at. So buffers come first and the guards come last.
    std::vector<char> stdout_data;
    OVERLAPPED stdin_write_overlapped = {};
    OVERLAPPED stdout_read_overlapped = {};

    handle_wrapper stdin_write_event{CreateEventA(nullptr, TRUE, FALSE, nullptr)};
    handle_wrapper stdout_read_event{CreateEventA(nullptr, TRUE, FALSE, nullptr)};
    if(not stdin_write_event.valid() or not stdout_read_event.valid())
        throw_error(GetLastError(), "Failed to create overlapped I/O event");
    stdin_write_overlapped.hEvent = stdin_write_event.get();
    stdout_read_overlapped.hEvent = stdout_read_event.get();

    handle_wrapper child_stdin_read;
    handle_wrapper child_stdin_write;
    handle_wrapper child_stdout_read;
    handle_wrapper child_stdout_write;
    make_pipe(child_stdin_read, child_stdin_write, false);
    make_pipe(child_stdout_read, child_stdout_write, true);

    // Keep our own ends out of the child: a stray duplicate of the stdin write end would stop the
    // child from ever seeing EOF on stdin, and one of the stdout read end would let it steal our
    // output.
    if(SetHandleInformation(child_stdin_write.get(), HANDLE_FLAG_INHERIT, 0) == FALSE)
        throw_error(GetLastError(), "Failed to uninherit stdin write handle");
    if(SetHandleInformation(child_stdout_read.get(), HANDLE_FLAG_INHERIT, 0) == FALSE)
        throw_error(GetLastError(), "Failed to uninherit stdout read handle");

    // Give the child an inheritable duplicate of our stderr rather than marking the real one
    // inheritable, which would be a permanent mutation of a process we are only a library in.
    handle_wrapper child_stderr;
    HANDLE our_stderr = GetStdHandle(STD_ERROR_HANDLE);
    if(our_stderr != nullptr and our_stderr != INVALID_HANDLE_VALUE)
    {
        // A failure here is not fatal: the child simply runs without stderr.
        DuplicateHandle(GetCurrentProcess(),
                        our_stderr,
                        GetCurrentProcess(),
                        &child_stderr.handle,
                        0,
                        TRUE,
                        DUPLICATE_SAME_ACCESS);
    }

    std::vector<HANDLE> inherited = {child_stdin_read.get(), child_stdout_write.get()};
    if(child_stderr.valid())
        inherited.push_back(child_stderr.get());
    proc_thread_attribute_list attributes{inherited};

    STARTUPINFOEXA info         = {};
    info.StartupInfo.cb         = sizeof(STARTUPINFOEXA);
    info.StartupInfo.dwFlags    = STARTF_USESTDHANDLES;
    info.StartupInfo.hStdInput  = child_stdin_read.get();
    info.StartupInfo.hStdOutput = child_stdout_write.get();
    info.StartupInfo.hStdError  = child_stderr.get();
    info.lpAttributeList        = attributes.list;

    std::string cmdline = quote_arg(cmd);
    for(const auto& arg : argv)
        cmdline += " " + quote_arg(arg);

    PROCESS_INFORMATION process_info = {};
    if(CreateProcessA(cmd.c_str(),
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
        throw_error(GetLastError(), "Failed to create process " + cmd);
    }
    handle_wrapper process_handle{process_info.hProcess};
    handle_wrapper thread_handle{process_info.hThread};
    child_guard guard{process_handle.get()};

    // The child has its own copies now. Ours must go, or we never see EOF on stdout.
    child_stdin_read.close();
    child_stdout_write.close();
    child_stderr.close();

    pending_io write_io{child_stdin_write.get(), stdin_write_overlapped};
    pending_io read_io{child_stdout_read.get(), stdout_read_overlapped};

    std::size_t total_bytes_written = 0;
    std::size_t total_bytes_read    = 0;

    // stdout is deliberately first: WaitForMultipleObjects reports the lowest signaled index, so a
    // child that writes its output and immediately exits still gets drained before we notice it
    // died. Slot 1 holds the write event only while we still have data to send.
    std::array<HANDLE, 3> handles{
        stdout_read_event.get(), stdin_write_event.get(), process_handle.get()};
    bool writing = not stdin_data.empty();

    // Signal EOF to the child and give the write event's slot to the process handle, so we still
    // notice the child dying once there is nothing left to send.
    auto finish_writing = [&] {
        write_io.cancel();
        child_stdin_write.close();
        handles[1] = process_handle.get();
        writing    = false;
    };

    // Overlapped writes never block on a full pipe, so there is no reason to chunk: hand the whole
    // remainder over and let it complete as the child drains. Returns false if the child has
    // stopped reading, which is not an error -- its exit status is the verdict.
    auto issue_write = [&] {
        assert(total_bytes_written < stdin_data.size());
        auto remaining = std::min<std::size_t>(stdin_data.size() - total_bytes_written, MAXDWORD);
        if(WriteFile(child_stdin_write.get(),
                     stdin_data.data() + total_bytes_written,
                     static_cast<DWORD>(remaining),
                     nullptr,
                     &stdin_write_overlapped) != FALSE)
        {
            write_io.arm();
            return true;
        }
        auto error = GetLastError();
        if(error == ERROR_IO_PENDING)
        {
            write_io.arm();
            return true;
        }
        if(error == ERROR_BROKEN_PIPE or error == ERROR_NO_DATA)
            return false;
        throw_error(error, "Failed to write to child stdin");
    };

    // Returns false once the child has closed stdout.
    auto issue_read = [&] {
        // Grow before issuing: the buffer must not move while a read is pending.
        assert(not read_io.active);
        stdout_data.resize(total_bytes_read + read_chunk_size);
        if(ReadFile(child_stdout_read.get(),
                    stdout_data.data() + total_bytes_read,
                    read_chunk_size,
                    nullptr,
                    &stdout_read_overlapped) != FALSE)
        {
            read_io.arm();
            return true;
        }
        auto error = GetLastError();
        if(error == ERROR_IO_PENDING)
        {
            read_io.arm();
            return true;
        }
        if(error == ERROR_BROKEN_PIPE or error == ERROR_HANDLE_EOF)
            return false;
        throw_error(error, "Failed to read from child stdout");
    };

    if(not writing or not issue_write())
        finish_writing();

    bool reading = issue_read();
    while(reading)
    {
        // The shortened wait is only correct because finish_writing() hands slot 1 to the process
        // handle as it clears `writing`; otherwise we would stop watching the child for death.
        assert(writing or handles[1] == process_handle.get());
        auto wait_result = WaitForMultipleObjects(writing ? 3 : 2, handles.data(), FALSE, INFINITE);
        if(wait_result == WAIT_OBJECT_0)
        {
            DWORD bytes_read = 0;
            read_io.disarm();
            if(GetOverlappedResult(
                   child_stdout_read.get(), &stdout_read_overlapped, &bytes_read, FALSE) == FALSE)
            {
                auto error = GetLastError();
                // Anything else means the read is still in flight; re-arm so the guard cancels it
                // rather than letting the buffer be freed underneath the kernel.
                if(error != ERROR_HANDLE_EOF and error != ERROR_BROKEN_PIPE)
                {
                    read_io.arm();
                    throw_error(error, "Failed to complete read from child stdout");
                }
                break;
            }
            total_bytes_read += bytes_read;
            reading = issue_read();
        }
        else if(writing and wait_result == WAIT_OBJECT_0 + 1)
        {
            DWORD bytes_written = 0;
            write_io.disarm();
            if(GetOverlappedResult(
                   child_stdin_write.get(), &stdin_write_overlapped, &bytes_written, FALSE) ==
               FALSE)
            {
                // The child stopped reading. Stop writing and keep draining stdout; the exit
                // status will say what happened.
                finish_writing();
            }
            else
            {
                total_bytes_written += bytes_written;
                if(total_bytes_written >= stdin_data.size() or not issue_write())
                    finish_writing();
            }
        }
        else
        {
            // The child exited with nothing left to read, or the wait failed.
            break;
        }
    }
    // Cancel before touching the buffer: leaving the loop early can leave a read in flight, and the
    // kernel must be done with the memory before we resize it.
    read_io.cancel();
    stdout_data.resize(total_bytes_read);
    // Both pipes must be closed before the wait. A child that closed stdout early but is still
    // reading would otherwise never see EOF, and we would wait on it forever.
    finish_writing();
    child_stdout_read.close();

    if(WaitForSingleObject(process_handle.get(), INFINITE) != WAIT_OBJECT_0)
        throw_error(GetLastError(), "Failed to wait for child process");
    guard.release();

    DWORD exit_code = 0;
    if(GetExitCodeProcess(process_handle.get(), &exit_code) == FALSE)
        throw_error(GetLastError(), "Failed to get child exit code");

    // An SEH-terminated child exits with something like 0xC0000005, which does not fit an int.
    // Report those the same way POSIX reports a signaled child.
    if(exit_code > static_cast<DWORD>((std::numeric_limits<int>::max)()))
        return {-1, std::move(stdout_data)};
    return {static_cast<int>(exit_code), std::move(stdout_data)};
}

#else

struct fd_wrapper
{
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

    int get() const { return fd; }

    int fd = -1;
};

// Reaps the child on every path out of the exec helpers, so a throw between posix_spawn and the
// explicit wait does not leave a zombie behind. Declared before the pipe wrappers so that it is
// destroyed after them -- waiting while we still hold the pipes open would deadlock against a child
// blocked on a full stdout pipe or on a stdin that never reaches EOF.
struct child_reaper
{
    child_reaper()                               = default;
    child_reaper(const child_reaper&)            = delete;
    child_reaper& operator=(const child_reaper&) = delete;
    ~child_reaper()
    {
        while(pid > 0)
        {
            int status  = 0;
            auto result = waitpid(pid, &status, 0);
            if(result >= 0 or errno != EINTR)
                pid = -1;
        }
    }

    void reset(pid_t p) { pid = p; }

    int wait()
    {
        for(;;)
        {
            int status  = 0;
            auto result = waitpid(pid, &status, 0);
            if(result == pid)
            {
                pid = -1;
                return status;
            }
            if(errno == EINTR)
                continue;
            // Nothing left to reap (the host may have SIGCHLD set to SIG_IGN), so do not let the
            // destructor block on it either.
            pid = -1;
            MIGRAPHX_THROW("Failed to wait for child process");
        }
    }

    pid_t pid = -1;
};

// Blocks SIGPIPE for this thread while we write to the child. Without it a child that dies
// mid-request kills the whole host application, since a write to a pipe has no MSG_NOSIGNAL and
// MIGraphX is a library that must not change the process-wide disposition.
struct sigpipe_blocker
{
    sigpipe_blocker()
    {
        sigset_t block_set;
        sigemptyset(&block_set);
        sigaddset(&block_set, SIGPIPE);
        blocked = pthread_sigmask(SIG_BLOCK, &block_set, &old_set) == 0;
    }
    sigpipe_blocker(const sigpipe_blocker&)            = delete;
    sigpipe_blocker& operator=(const sigpipe_blocker&) = delete;
    ~sigpipe_blocker()
    {
        if(not blocked)
            return;
        // Consume a SIGPIPE raised while it was blocked, so unblocking does not deliver it. Only
        // safe to do when we are the ones who blocked it, which is what the saved mask tells us.
        // sigtimedwait with a zero timeout rather than sigwait: sigpending also reports
        // process-directed signals, and two threads racing for the same one would leave the loser
        // blocked forever.
        if(sigismember(&old_set, SIGPIPE) != 1)
        {
            sigset_t wait_set;
            sigemptyset(&wait_set);
            sigaddset(&wait_set, SIGPIPE);
            const struct timespec no_wait = {0, 0};
            // Empty body is intentional: sigtimedwait does the work, we only retry on EINTR.
            // cppcheck-suppress migraphx-EmptyWhileStatement
            while(sigtimedwait(&wait_set, nullptr, &no_wait) < 0 and errno == EINTR)
            {
            }
        }
        pthread_sigmask(SIG_SETMASK, &old_set, nullptr);
    }

    sigset_t old_set{};
    bool blocked = false;
};

// poll() flags are signed ints against a signed short; mask in unsigned to keep the bit test out of
// signed-bitwise territory.
bool has_event(short revents, int flag)
{
    return (static_cast<unsigned int>(revents) & static_cast<unsigned int>(flag)) != 0;
}

// POSIX guarantees a page can be written without blocking once poll() reports POLLOUT, and these
// fds are blocking, so cap each write there.
constexpr std::size_t write_chunk_size = 4096;

// The two halves of the poll loop, split out so neither the loop nor either half is large enough to
// obscure what it does.
struct pipe_pump
{
    pipe_pump(const std::vector<char>& data, fd_wrapper& write_end, fd_wrapper& read_end)
        : stdin_data(data), stdin_write(write_end), stdout_read(read_end)
    {
        fds[0].fd     = stdin_write.get();
        fds[0].events = POLLOUT;
        fds[1].fd     = stdout_read.get();
        fds[1].events = POLLIN;
        if(stdin_data.empty())
            finish_writing();
    }

    const std::vector<char>& stdin_data;
    fd_wrapper& stdin_write;
    fd_wrapper& stdout_read;
    std::array<pollfd, 2> fds{};
    std::size_t total_bytes_written = 0;
    std::size_t total_bytes_read    = 0;
    std::vector<char> stdout_data{};

    // Stop feeding the child and let it see EOF. A negative fd is ignored by poll.
    void finish_writing()
    {
        stdin_write.close();
        fds[0].fd = -1;
    }

    void handle_write_event()
    {
        // An error on the write side only means the child stopped reading; the caller keeps
        // draining stdout so whatever it already produced is not thrown away.
        if(has_event(fds[0].revents, POLLERR) or has_event(fds[0].revents, POLLHUP))
        {
            finish_writing();
            return;
        }
        if(not has_event(fds[0].revents, POLLOUT))
            return;
        assert(total_bytes_written < stdin_data.size());
        auto remaining = stdin_data.size() - total_bytes_written;
        auto written   = ::write(stdin_write.get(),
                                 stdin_data.data() + total_bytes_written,
                                 std::min(remaining, write_chunk_size));
        if(written <= 0)
        {
            // EINTR wrote nothing at all, so just come back on the next poll.
            if(written < 0 and errno == EINTR)
                return;
            finish_writing();
            return;
        }
        total_bytes_written += static_cast<std::size_t>(written);
        if(total_bytes_written >= stdin_data.size())
            finish_writing();
    }

    // Returns false once stdout has reached EOF and there is nothing more to collect.
    bool handle_read_event()
    {
        if(not has_event(fds[1].revents, POLLIN))
        {
            return not has_event(fds[1].revents, POLLHUP) and
                   not has_event(fds[1].revents, POLLERR);
        }
        // Read straight into the result rather than via a scratch buffer, so a multi-megabyte code
        // object is not copied a second time on its way out.
        stdout_data.resize(total_bytes_read + read_chunk_size);
        auto bytes_read =
            ::read(stdout_read.get(), stdout_data.data() + total_bytes_read, read_chunk_size);
        if(bytes_read < 0)
        {
            if(errno == EINTR)
                return true;
            MIGRAPHX_THROW("Failed to read from child stdout");
        }
        // A zero-length read on a pipe means every write end is closed.
        if(bytes_read == 0)
            return false;
        total_bytes_read += static_cast<std::size_t>(bytes_read);
        return true;
    }

    // Drop the slack left by the last resize. Must not be called with a read outstanding.
    std::vector<char> take_stdout()
    {
        stdout_data.resize(total_bytes_read);
        return std::move(stdout_data);
    }
};

// Both ends are O_CLOEXEC, so a spawn racing on another thread cannot inherit them. The spawn that
// is meant to get one end dup2s it into place, which clears the flag on the copy.
std::array<int, 2> make_pipe(const std::string& name)
{
    std::array<int, 2> fds{-1, -1};
    if(pipe2(fds.data(), O_CLOEXEC) != 0)
        MIGRAPHX_THROW("Failed to create " + name + " pipe");
    return fds;
}

// The child's exit status, or -1 if it did not exit normally at all because a signal killed it.
// The caller only needs "did it work", so which signal is not reported.
int exit_code_of(int wait_status)
{
    return WIFEXITED(wait_status) ? WEXITSTATUS(wait_status) : -1; // NOLINT
}

struct spawn_options
{
    fs::path cwd{};
    // NAME=value entries layered over our own environment.
    std::vector<std::string> envs{};
    // Installed as the child's stdin and stdout; -1 leaves it sharing ours.
    int stdin_fd  = -1;
    int stdout_fd = -1;
};

// Our environment, with every variable named in envs replaced by the value given there -- what a
// `NAME=value cmd` prefix does in a shell.
std::vector<std::string> make_child_environment(const std::vector<std::string>& envs)
{
    // Kept as "NAME=", so that a variable whose name merely starts with NAME is left alone.
    std::vector<std::string> prefixes;
    migraphx::transform(envs, std::back_inserter(prefixes), [](const std::string& entry) {
        auto eq = entry.find('=');
        if(eq == std::string::npos or eq == 0)
            MIGRAPHX_THROW("Environment variable must be given as NAME=value: " + entry);
        return entry.substr(0, eq + 1);
    });

    std::vector<std::string> result;
    if(environ != nullptr)
    {
        char** last = environ;
        while(*last != nullptr)
            ++last;
        result.assign(environ, last);
    }
    auto overridden = [&](const std::string& var) {
        return std::any_of(prefixes.begin(), prefixes.end(), [&](const std::string& p) {
            return starts_with(var, p);
        });
    };
    result.erase(std::remove_if(result.begin(), result.end(), overridden), result.end());
    result.insert(result.end(), envs.begin(), envs.end());
    return result;
}

// posix_spawn takes null-terminated arrays of C strings, and does not modify them despite the
// non-const signature.
std::vector<const char*> to_c_strings(const std::vector<std::string>& strings)
{
    std::vector<const char*> result;
    result.reserve(strings.size() + 1);
    migraphx::transform(
        strings, std::back_inserter(result), [](const std::string& s) { return s.c_str(); });
    result.push_back(nullptr);
    return result;
}

// glibc 2.29 added posix_spawn_file_actions_addchdir_np; older C libraries go through /bin/sh.
#if defined(__GLIBC__) && (__GLIBC__ > 2 || (__GLIBC__ == 2 && __GLIBC_MINOR__ >= 29))
#define MIGRAPHX_HAS_SPAWN_ADDCHDIR 1
#else
#define MIGRAPHX_HAS_SPAWN_ADDCHDIR 0
#endif

// Starts cmd with args as its argv, never through a shell, and returns its pid. PATH is searched
// for a cmd without a slash, as a shell would; the default src_compiler is a bare "c++". Use
// posix_spawn rather than fork: the parent has the HIP runtime loaded and many threads running, and
// fork in that state is a hazard.
pid_t spawn_child(const std::string& cmd,
                  const std::vector<std::string>& args,
                  const spawn_options& options)
{
    std::vector<std::string> argv{cmd};
    argv.insert(argv.end(), args.begin(), args.end());
#if !MIGRAPHX_HAS_SPAWN_ADDCHDIR
    if(not options.cwd.empty())
    {
        // A fixed script changes directory and then becomes the command. The cwd and the argv reach
        // it as positional parameters, never as script text, so none of them is interpreted.
        argv.insert(argv.begin(),
                    {"/bin/sh", "-c", R"(cd -- "$0" && exec "$@")", options.cwd.string()});
    }
#endif
    auto child_argv = to_c_strings(argv);

    std::vector<std::string> env_storage;
    std::vector<const char*> child_envp;
    char* const* envp = environ;
    if(not options.envs.empty())
    {
        env_storage = make_child_environment(options.envs);
        child_envp  = to_c_strings(env_storage);
        envp        = const_cast<char* const*>(child_envp.data()); // NOLINT
    }

    // posix_spawn's dup2 action clears O_CLOEXEC on the duplicate, so the child gets these as its
    // stdin/stdout while our other fds stay O_CLOEXEC; stderr is inherited untouched.
    posix_spawn_file_actions_t file_actions;
    if(posix_spawn_file_actions_init(&file_actions) != 0)
        MIGRAPHX_THROW("Failed to allocate spawn file actions");

    // Separate statements: the actions are applied in the order they are added, so the order must
    // not be left to the compiler.
    int action_result = 0;
    if(options.stdin_fd != -1)
    {
        action_result =
            posix_spawn_file_actions_adddup2(&file_actions, options.stdin_fd, STDIN_FILENO);
    }
    if(action_result == 0 and options.stdout_fd != -1)
    {
        action_result =
            posix_spawn_file_actions_adddup2(&file_actions, options.stdout_fd, STDOUT_FILENO);
    }
#if MIGRAPHX_HAS_SPAWN_ADDCHDIR
    if(action_result == 0 and not options.cwd.empty())
        action_result = posix_spawn_file_actions_addchdir_np(&file_actions, options.cwd.c_str());
#endif

    pid_t pid        = 0;
    int spawn_result = action_result;
    if(spawn_result == 0)
    {
        spawn_result = posix_spawnp(&pid,
                                    argv.front().c_str(),
                                    &file_actions,
                                    nullptr,
                                    const_cast<char* const*>(child_argv.data()), // NOLINT
                                    envp);
    }
    posix_spawn_file_actions_destroy(&file_actions);
    if(spawn_result != 0)
        MIGRAPHX_THROW("Failed to spawn process " + cmd + " (errno " +
                       std::to_string(spawn_result) + ")");
    return pid;
}

// Runs the command with its stdout collected through a pipe, handing each chunk to std_out as it
// arrives. Returns the child's exit code as exit_code_of reports it.
int exec_read(const std::string& cmd,
              const std::vector<std::string>& args,
              spawn_options options,
              const process::writer& std_out)
{
    // Declared before the pipe so it is destroyed after it; see child_reaper's comment.
    child_reaper child;

    auto stdout_fds = make_pipe("stdout");
    fd_wrapper child_stdout_read{stdout_fds[0]};
    fd_wrapper child_stdout_write{stdout_fds[1]};

    options.stdout_fd = child_stdout_write.get();
    child.reset(spawn_child(cmd, args, options));
    // Drop our copy of the child's end, or stdout never reports EOF.
    child_stdout_write.close();

    std::vector<char> buffer(read_chunk_size);
    for(;;)
    {
        auto bytes_read = ::read(child_stdout_read.get(), buffer.data(), buffer.size());
        if(bytes_read < 0)
        {
            if(errno == EINTR)
                continue;
            MIGRAPHX_THROW("Failed to read from child stdout");
        }
        // A zero-length read on a pipe means every write end is closed.
        if(bytes_read == 0)
            break;
        std_out(buffer.data(), static_cast<std::size_t>(bytes_read));
    }
    child_stdout_read.close();
    return exit_code_of(child.wait());
}

// Writes all n bytes, picking up after short writes and signals. A child that stops reading is not
// an error here: the rest is dropped and the child's exit status decides the outcome, as it does
// for read_write.
void write_all(fd_wrapper& fd, const char* buffer, std::size_t n)
{
    while(n > 0 and fd.get() != -1)
    {
        auto written = ::write(fd.get(), buffer, n);
        if(written < 0)
        {
            if(errno == EINTR)
                continue;
            if(errno != EPIPE)
                MIGRAPHX_THROW("Failed to write to child stdin");
            fd.close();
            return;
        }
        buffer += written;
        n -= static_cast<std::size_t>(written);
    }
}

// Runs the command with std_in feeding its stdin through a pipe; the child's stdout is shared with
// ours. Returns the child's exit code as exit_code_of reports it.
int exec_write(const std::string& cmd,
               const std::vector<std::string>& args,
               spawn_options options,
               const std::function<void(process::writer)>& std_in)
{
    // Declared before the pipe so it is destroyed after it; see child_reaper's comment.
    child_reaper child;

    auto stdin_fds = make_pipe("stdin");
    fd_wrapper child_stdin_read{stdin_fds[0]};
    fd_wrapper child_stdin_write{stdin_fds[1]};

    options.stdin_fd = child_stdin_read.get();
    child.reset(spawn_child(cmd, args, options));
    // Drop our copy of the child's end, or a dead child never shows up as EPIPE.
    child_stdin_read.close();

    {
        sigpipe_blocker no_sigpipe;
        std_in([&](const char* buffer, std::size_t n) { write_all(child_stdin_write, buffer, n); });
    }
    // The child sees EOF only once this is closed, so it must happen before the wait.
    child_stdin_write.close();
    return exit_code_of(child.wait());
}

exec_result exec_read_write(const std::string& cmd,
                            const std::vector<std::string>& argv,
                            const std::vector<char>& stdin_data)
{
    // Declared before the pipes so it is destroyed after them; see child_reaper's comment.
    child_reaper child;

    auto stdin_fds = make_pipe("stdin");
    fd_wrapper child_stdin_read{stdin_fds[0]};
    fd_wrapper child_stdin_write{stdin_fds[1]};

    auto stdout_fds = make_pipe("stdout");
    fd_wrapper child_stdout_read{stdout_fds[0]};
    fd_wrapper child_stdout_write{stdout_fds[1]};

    spawn_options options;
    options.stdin_fd  = child_stdin_read.get();
    options.stdout_fd = child_stdout_write.get();
    child.reset(spawn_child(cmd, argv, options));

    // Drop our copies of the child's ends, or stdout never reports EOF and a dead child never shows
    // up as an error on the write side.
    child_stdin_read.close();
    child_stdout_write.close();

    pipe_pump pump{stdin_data, child_stdin_write, child_stdout_read};

    sigpipe_blocker no_sigpipe;
    for(;;)
    {
        if(poll(pump.fds.data(), static_cast<nfds_t>(pump.fds.size()), -1) < 0)
        {
            if(errno == EINTR)
                continue;
            MIGRAPHX_THROW("Failed to poll child pipes for " + cmd);
        }
        pump.handle_write_event();
        if(not pump.handle_read_event())
            break;
    }
    // Both pipes must be closed before the wait. A child that closed stdout early but is still
    // reading would otherwise never see EOF, and we would wait on it forever.
    pump.finish_writing();
    child_stdout_read.close();

    return {exit_code_of(child.wait()), pump.take_stdout()};
}

#endif

} // namespace

struct process_impl
{
    std::vector<std::string> envs{};
    std::string command{};
    fs::path cwd{};
    // POSIX spawns the command directly instead of through a shell, so the arguments are kept
    // unjoined and the joined form Windows wants is built on demand.
    std::vector<std::string> arg_list{};

    std::string args() const { return join_strings(arg_list, " "); }

    // For messages and tracing only; nothing ever runs this string.
    std::string get_command() const
    {
        std::string result;
        if(not cwd.empty())
            result += "cd " + cwd.string() + "; ";
        if(not envs.empty())
            result += join_strings(envs, " ") + " ";
        result += command;
        if(not arg_list.empty())
            result += " " + args();
        return result;
    }

    // ec is the child's exit status, or negative when it did not exit normally at all.
    void check_exit_code(int ec) const
    {
        if(ec == 0)
            return;
        MIGRAPHX_THROW("Command " + get_command() +
                       (ec < 0 ? " terminated abnormally"
                               : " exited with status " + std::to_string(ec)));
    }

#ifdef _WIN32
    template <class... Ts>
    void check_exec(Ts&&... xs) const
    {
        check_exit_code(migraphx::exec(std::forward<Ts>(xs)...));
    }
#else
    spawn_options spawn_opts() const
    {
        spawn_options options;
        options.cwd  = cwd;
        options.envs = envs;
        return options;
    }
#endif
};

process::process(const std::string& cmd, const std::vector<std::string>& args)
    : impl(std::make_unique<process_impl>())
{
    impl->command  = cmd;
    impl->arg_list = args;
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
    impl->check_exec(impl->command, impl->cwd.string(), impl->args(), join_strings(impl->envs, " "),
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
    std::string result;
    auto append = [&](const char* data, std::size_t n) { result.append(data, n); };
    trace_command(impl->get_command());
    impl->check_exit_code(exec_read(impl->command, impl->arg_list, impl->spawn_opts(), append));
#endif
    output(result.data(), result.size());
}

void process::exec()
{
#ifndef _WIN32
    auto to_cout = [](const char* data, std::size_t n) {
        std::cout.write(data, static_cast<std::streamsize>(n));
    };
    trace_command(impl->get_command());
    impl->check_exit_code(exec_read(impl->command, impl->arg_list, impl->spawn_opts(), to_cout));
#else
    // clang-format off
    impl->check_exec(impl->command, impl->cwd.string(), impl->args(), join_strings(impl->envs, " "),
                     GetStdHandle(STD_OUTPUT_HANDLE));
    // clang-format on
#endif
}

void process::write(std::function<void(writer)> pipe_in)
{
#ifndef _WIN32
    trace_command(impl->get_command());
    impl->check_exit_code(exec_write(impl->command, impl->arg_list, impl->spawn_opts(), pipe_in));
#else
    // clang-format off
    impl->check_exec(impl->command, impl->cwd.string(),
                     impl->args(), join_strings(impl->envs, " "), std::move(pipe_in));
    // clang-format on
#endif
}

void process::read_write(const std::function<void(writer)>& pipe_in, const writer& output)
{
    // Neither is plumbed through the Windows bidirectional spawn, and no caller needs them. Reject
    // them on every platform rather than silently drop them on one.
    if(not impl->cwd.empty() or not impl->envs.empty())
        MIGRAPHX_THROW("Command " + impl->get_command() +
                       " uses read_write, which does not support cwd or env");

    // Collected up front rather than streamed: pipe_in pushes, and the pump has to hand the kernel
    // a contiguous buffer it can keep re-offsetting into across many poll iterations. Interleaving
    // the two would take a second thread.
    std::vector<char> stdin_data;
    pipe_in([&](const char* buffer, std::size_t n) {
        // An empty write may hand us a null pointer, which is not a valid range even for a count
        // of zero.
        if(n > 0)
            stdin_data.insert(stdin_data.end(), buffer, buffer + n);
    });

    trace_command(impl->get_command());
    auto result = exec_read_write(impl->command, impl->arg_list, stdin_data);
    impl->check_exit_code(result.exit_code);
    output(result.stdout_data.data(), result.stdout_data.size());
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
