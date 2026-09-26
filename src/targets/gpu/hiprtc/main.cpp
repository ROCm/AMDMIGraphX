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
#include <migraphx/gpu/compile_hip.hpp>
#include <migraphx/gpu/mlir.hpp>
#include <migraphx/process.hpp>
#include <migraphx/program.hpp>
#include <migraphx/serialize.hpp>
#include <migraphx/time.hpp>
#include <migraphx/value.hpp>
#include <migraphx/msgpack.hpp>
#include <migraphx/errors.hpp>
#include <migraphx/ranges.hpp>
#include <array>
#include <cassert>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <iostream>
#include <cstdio>
#include <mutex>
#include <thread>

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#else
#include <unistd.h>
#endif

// CRLF translation would corrupt both the msgpack request and the msgpack reply. Nothing restores
// the old mode: the streams carry nothing else before the process exits.
// https://learn.microsoft.com/en-us/cpp/c-runtime-library/reference/setmode?view=msvc-170
static void set_binary_mode([[maybe_unused]] FILE* stream)
{
#ifdef _WIN32
    if(_setmode(_fileno(stream), _O_BINARY) == -1)
        MIGRAPHX_THROW("Failed to set stream to binary mode");
#endif
}

static std::vector<char> read_stdin()
{
    set_binary_mode(stdin);
    std::vector<char> result;
    std::array<char, 1024> buffer{};
    std::size_t len = 0;
    while((len = std::fread(buffer.data(), 1, buffer.size(), stdin)) > 0)
        result.insert(result.end(), buffer.data(), buffer.data() + len);
    // Checked after the loop: a short read is how both EOF and an error end it, and only ferror
    // tells them apart. Inside the loop the check can never see the read that terminated it.
    if(std::ferror(stdin) != 0)
        MIGRAPHX_THROW("Failed reading request from stdin");
    return result;
}

// stdout is a binary channel: nothing else in this process may write to it, or the parent reads a
// corrupted reply. Takes a producer rather than a buffer so the reply, which carries the whole code
// object, is written as it is serialized instead of being materialized twice.
template <class F>
static void write_stdout(const F& produce)
{
    set_binary_mode(stdout);
    produce([](const char* data, std::size_t n) {
        if(std::fwrite(data, 1, n, stdout) != n)
            MIGRAPHX_THROW("Failed writing reply to stdout");
    });
    if(std::fflush(stdout) != 0)
        MIGRAPHX_THROW("Failed flushing stdout");
}

// stderr, not stdout: stdout is reserved for the reply.
static void print_usage()
{
    std::cerr << "USAGE:" << std::endl;
    std::cerr << "    ";
    std::cerr << "Used internally by migraphx to compile hip programs out-of-process." << std::endl;
    std::cerr << "    ";
    std::cerr << "Reads a msgpack compile request on stdin and writes a msgpack reply, carrying "
                 "the code object, to stdout."
              << std::endl;
    std::cerr << "    ";
    std::cerr
        << "With --serve, answers MLIR compile requests one after another until stdin closes, "
           "each request and reply being one message in process::write_message framing."
        << std::endl;
}

using milliseconds = std::chrono::duration<double, std::milli>;

// How often a compile's CPU time is checked against its budget, which bounds the overshoot
constexpr std::chrono::milliseconds watchdog_period{10};

static void write_reply(std::ostream& replies, const migraphx::value& reply)
{
    migraphx::process::write_message(replies, migraphx::to_msgpack(reply));
}

// Replies "timeout" for a compile that runs out of CPU budget and then ends the process, since a
// running compile can't be canceled. One watchdog serves the whole session and sleeps between
// compiles.
struct compile_watchdog
{
    explicit compile_watchdog(std::ostream& out) : replies(out), thread([this] { watch(); }) {}
    compile_watchdog(const compile_watchdog&)            = delete;
    compile_watchdog& operator=(const compile_watchdog&) = delete;
    ~compile_watchdog()
    {
        {
            std::lock_guard<std::mutex> lock(mutex);
            stopping = true;
        }
        wake.notify_one();
        thread.join();
    }

    void arm(std::chrono::milliseconds cpu_budget)
    {
        {
            std::lock_guard<std::mutex> lock(mutex);
            armed  = true;
            budget = cpu_budget;
            timer  = migraphx::cpu_timer{};
        }
        wake.notify_one();
    }

    // Once this returns, the watchdog can no longer reply for the compile
    void disarm()
    {
        std::lock_guard<std::mutex> lock(mutex);
        armed = false;
    }

    private:
    void watch()
    {
        std::unique_lock<std::mutex> lock(mutex);
        while(not stopping)
        {
            wake.wait(lock, [&] { return armed or stopping; });
            auto used = timer.record<milliseconds>();
            if(armed and used >= budget.count())
            {
                // The lock is held until the process ends, so disarm() blocks and the compile
                // can't reply as well
                write_reply(replies, {{"timeout", true}, {"cpu_ms", used}});
                std::_Exit(0);
            }
            wake.wait_for(lock, watchdog_period, [&] { return not armed or stopping; });
        }
    }

    std::ostream& replies;
    std::mutex mutex;
    std::condition_variable wake;
    bool armed    = false;
    bool stopping = false;
    milliseconds budget{0};
    migraphx::cpu_timer timer{};
    std::thread thread;
};

// The request is decoded before the watchdog is armed, so only the compile counts against the
// budget.
static migraphx::value serve_request(const std::vector<char>& request, compile_watchdog& watchdog)
{
    try
    {
        auto v = migraphx::from_msgpack(request);
        migraphx::program p;
        p.from_value(v.at("program"));
        auto inputs = migraphx::from_value<std::vector<migraphx::shape>>(v.at("inputs"));
        auto props  = migraphx::from_value<migraphx::gpu::mlir_gpu_properties>(v.at("gpu"));
        migraphx::timer wall{};
        migraphx::cpu_timer cpu{};
        watchdog.arm(std::chrono::milliseconds{v.at("cpu_budget_ms").to<std::int64_t>()});
        auto mco =
            migraphx::gpu::compile_mlir(props, *p.get_main_module(), inputs, v.at("solution"));
        watchdog.disarm();
        return {{"mlir_code_object", migraphx::to_value(mco)},
                {"cpu_ms", cpu.record<milliseconds>()},
                {"wall_ms", wall.record<milliseconds>()}};
    }
    catch(const std::exception& e)
    {
        watchdog.disarm();
        return {{"error", e.what()}};
    }
}

static int serve()
{
    set_binary_mode(stdin);
    try
    {
        // Everything else printed to stdout, such as MIGRAPHX_TRACE_MLIR output, goes to stderr
        auto replies = migraphx::process::take_stdout();
        migraphx::gpu::warm_up_mlir();
        compile_watchdog watchdog{*replies};
        while(auto request = migraphx::process::read_message(std::cin))
            write_reply(*replies, serve_request(*request, watchdog));
    }
    catch(const std::exception& err)
    {
        std::cerr << err.what() << std::endl;
        return 1;
    }
    // The parent waits for the driver to exit, and the static destructors of the MLIR and HIP
    // libraries take tens of milliseconds. Every reply has already been flushed.
    std::_Exit(0);
}

// Without this, a bare invocation would silently block in fread waiting for a human to type
// msgpack.
static bool stdin_is_interactive()
{
#ifdef _WIN32
    return _isatty(_fileno(stdin)) != 0;
#else
    return isatty(fileno(stdin)) != 0;
#endif
}

int main(int argc, char const* argv[])
{
    // Request and reply are both msgpack on stdin/stdout, so no arguments are expected in normal
    // operation.
    if(argc > 1)
    {
        std::string arg = argv[1];
        if(arg == "--serve")
            return serve();
        print_usage();
        std::exit(migraphx::contains({"-h", "--help", "-v", "--version"}, arg) ? 0 : 1);
    }
    if(stdin_is_interactive())
    {
        print_usage();
        std::exit(1);
    }
    bool quiet = false;
    try
    {
        auto v = migraphx::from_msgpack(read_stdin());
        quiet  = v.at("quiet").to<bool>();
        std::vector<migraphx::gpu::hiprtc_src_file> srcs;
        migraphx::from_value(v.at("srcs"), srcs);
        auto out =
            migraphx::gpu::compile_hip_src_with_hiprtc(std::move(srcs),
                                                       v.at("params").to_vector<std::string>(),
                                                       v.at("arch").to<std::string>(),
                                                       quiet);
        // compile_hip_src_with_hiprtc throws on an empty code object and returns exactly one.
        assert(not out.empty());
        // A msgpack map rather than the raw bytes, so the reply can carry more than the code object
        // later on without the parent having to guess at what it received.
        migraphx::value reply;
        reply["code_object"] = migraphx::value::binary{out.front()};
        write_stdout([&](auto writer) { migraphx::to_msgpack(reply, writer); });
    }
    catch(const std::exception& err)
    {
        if(not quiet)
            std::cerr << err.what() << std::endl;
        // Exit status is the parent's only failure signal.
        return 1;
    }
    return 0;
}
