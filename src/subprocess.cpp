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
// STARTUPINFOEX and the process-thread attribute list are Vista+.
// cppcheck-suppress definePrefix
#define _WIN32_WINNT 0x0600
// Windows.h defines min/max as macros, which breaks std::numeric_limits<int>::max().
// cppcheck-suppress definePrefix
#define NOMINMAX
// cppcheck-suppress definePrefix
#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#include <cstdint>
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
    // stderr, not stdout: a caller may be redirecting stdout to collect real output.
    std::cerr << exe.string();
    if(not argv.empty())
        std::cerr << " " << join_strings(argv, " ");
    std::cerr << std::endl;
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
    const std::string pipe_name = "\\\\.\\pipe\\migraphx_subprocess_" +
                                  std::to_string(GetCurrentProcessId()) + "_" +
                                  std::to_string(GetCurrentThreadId()) + "_" +
                                  std::to_string(reinterpret_cast<std::uintptr_t>(&read)); // NOLINT

    SECURITY_ATTRIBUTES sa;
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
        // std::allocator<char> goes through ::operator new, which is aligned for any fundamental
        // type, so the attribute list is suitably aligned.
        storage.resize(size);
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

    std::vector<char> storage{};
    LPPROC_THREAD_ATTRIBUTE_LIST list = nullptr;
};

// Cancels a pending overlapped operation and waits for the kernel to release its buffer. Without
// this, an exception unwinding out of run() frees memory the kernel is still writing into.
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

// Kills the child if run() exits without a clean wait, so a throw does not leave a compiler
// process running against pipes nobody holds.
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

subprocess_result
run(const fs::path& exe, const std::vector<std::string>& argv, const std::vector<char>& stdin_data)
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
        throw_error(GetLastError(), "Failed to create process " + exe_string);
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

    // stdout is deliberately first: WaitForMultipleObjects reports the lowest signalled index, so a
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
        stdout_data.resize(total_bytes_read + read_chunk_size);
        assert(stdout_data.size() - total_bytes_read >= read_chunk_size);
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
    // Report those the same way POSIX reports a signalled child.
    if(exit_code > static_cast<DWORD>((std::numeric_limits<int>::max)()))
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

    int get() const { return fd; }

    int fd = -1;
};

// Reaps the child on every path out of run(), so a throw between posix_spawn and the explicit wait
// does not leave a zombie behind. Declared before the pipe wrappers so that it is destroyed after
// them -- waiting while we still hold the pipes open would deadlock against a child blocked on a
// full stdout pipe or on a stdin that never reaches EOF.
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
        blocked         = pthread_sigmask(SIG_BLOCK, &block_set, &old_set) == 0;
        already_blocked = sigismember(&old_set, SIGPIPE) == 1;
    }
    sigpipe_blocker(const sigpipe_blocker&)            = delete;
    sigpipe_blocker& operator=(const sigpipe_blocker&) = delete;
    ~sigpipe_blocker()
    {
        if(not blocked)
            return;
        // Consume a SIGPIPE raised while it was blocked, so unblocking does not deliver it. Only
        // safe to do when we are the ones who blocked it.
        if(not already_blocked)
        {
            sigset_t pending;
            sigemptyset(&pending);
            if(sigpending(&pending) == 0 and sigismember(&pending, SIGPIPE) == 1)
            {
                sigset_t wait_set;
                sigemptyset(&wait_set);
                sigaddset(&wait_set, SIGPIPE);
                int sig = 0;
                sigwait(&wait_set, &sig);
            }
        }
        pthread_sigmask(SIG_SETMASK, &old_set, nullptr);
    }

    sigset_t old_set{};
    bool blocked         = false;
    bool already_blocked = false;
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

subprocess_result
run(const fs::path& exe, const std::vector<std::string>& argv, const std::vector<char>& stdin_data)
{
    // Declared before the pipes so it is destroyed after them; see child_reaper's comment.
    child_reaper child;

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
    if(posix_spawn_file_actions_init(&file_actions) != 0)
        MIGRAPHX_THROW("Failed to allocate spawn file actions");

    // Separate statements: the actions are applied in the order they are added, so the order must
    // not be left to the compiler.
    int action_result =
        posix_spawn_file_actions_adddup2(&file_actions, child_stdin_read.get(), STDIN_FILENO);
    if(action_result == 0)
    {
        action_result = posix_spawn_file_actions_adddup2(
            &file_actions, child_stdout_write.get(), STDOUT_FILENO);
    }

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
        MIGRAPHX_THROW("Failed to spawn process " + exe_string + " (errno " +
                       std::to_string(spawn_result) + ")");
    child.reset(pid);

    // Drop our copies of the child's ends, or stdout never reports EOF and a dead child never shows
    // up as an error on the write side.
    child_stdin_read.close();
    child_stdout_write.close();

    std::array<pollfd, 2> fds{};
    fds[0].fd     = child_stdin_write.get();
    fds[0].events = POLLOUT;
    fds[1].fd     = child_stdout_read.get();
    fds[1].events = POLLIN;

    // Stop feeding the child and let it see EOF. A negative fd is ignored by poll.
    auto finish_writing = [&] {
        child_stdin_write.close();
        fds[0].fd = -1;
    };

    if(stdin_data.empty())
        finish_writing();

    sigpipe_blocker no_sigpipe;
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

        // An error on the write side only means the child stopped reading; keep draining stdout so
        // whatever it already produced is not thrown away.
        if(has_event(fds[0].revents, POLLERR) or has_event(fds[0].revents, POLLHUP))
            finish_writing();
        else if(has_event(fds[0].revents, POLLOUT))
        {
            assert(total_bytes_written < stdin_data.size());
            auto remaining = stdin_data.size() - total_bytes_written;
            auto written   = write(child_stdin_write.get(),
                                   stdin_data.data() + total_bytes_written,
                                   std::min(remaining, write_chunk_size));
            if(written < 0 and errno == EINTR)
                continue;
            if(written <= 0)
                finish_writing();
            else
            {
                total_bytes_written += static_cast<std::size_t>(written);
                if(total_bytes_written >= stdin_data.size())
                    finish_writing();
            }
        }

        if(has_event(fds[1].revents, POLLIN))
        {
            auto bytes_read = read(child_stdout_read.get(), buffer.data(), buffer.size());
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
        else if(has_event(fds[1].revents, POLLHUP) or has_event(fds[1].revents, POLLERR))
        {
            break;
        }
    }
    // Both pipes must be closed before the wait. A child that closed stdout early but is still
    // reading would otherwise never see EOF, and we would wait on it forever.
    finish_writing();
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
