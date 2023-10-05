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
#include <unistd.h>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_TRACE_CMD_EXECUTE)

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

void process::exec() { impl->check_exec(impl->get_command(), redirect_to(std::cout)); }

void process::write(std::function<void(process::writer)> pipe_in)
{
    impl->check_exec(impl->get_command(), std::move(pipe_in));
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
