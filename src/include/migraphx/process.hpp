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
#ifndef MIGRAPHX_GUARD_MIGRAPHX_PROCESS_HPP
#define MIGRAPHX_GUARD_MIGRAPHX_PROCESS_HPP

#include <migraphx/config.hpp>
#include <migraphx/filesystem.hpp>
#include <migraphx/optional.hpp>
#include <functional>
#include <iosfwd>
#include <string>
#include <memory>
#include <vector>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct process_impl;
struct process_session_impl;

struct MIGRAPHX_EXPORT process
{
    using writer = std::function<void(const char*, std::size_t)>;
    explicit process(const std::string& cmd, const std::vector<std::string>& args = {});

    explicit process(const fs::path& cmd, const std::vector<std::string>& args = {})
        : process{cmd.string(), args}
    {
    }

    // move constructor
    process(process&&) noexcept;

    // copy assignment operator
    process& operator=(process rhs);

    ~process() noexcept;

    process& cwd(const fs::path& p);
    process& env(const std::vector<std::string>& envs);

    void exec();
    void write(std::function<void(writer)> pipe_in);
    void read(const writer& output) const;

    /// Writes the bytes `pipe_in` produces to the child's stdin while concurrently draining its
    /// stdout; done in sequence, a request larger than the pipe buffer would deadlock. `output` is
    /// invoked exactly once, with the whole of stdout. Spawns directly rather than through a shell
    /// and never merges stderr into stdout, so stdout stays a clean binary channel; cwd() and env()
    /// are unsupported. Throws if the child cannot be spawned, exits non-zero, or terminates
    /// abnormally.
    void read_write(const std::function<void(writer)>& pipe_in, const writer& output);

    /// A child kept running to answer requests one at a time over its stdin and stdout. Every
    /// request and every reply is one message in the framing of read_message and write_message.
    struct MIGRAPHX_EXPORT session
    {
        session(session&&) noexcept;
        session& operator=(session&&) noexcept;
        /// Closes the child's stdin and waits for it to exit. A child whose request was cut short
        /// by an exception may still be working on it, so it is killed first.
        ~session() noexcept;

        /// Sends the bytes `pipe_in` produces as one message and returns the payload of the
        /// child's next message. Throws if the child closes stdout or exits before the reply is
        /// complete. After a throw the session is broken: every later request throws too.
        std::vector<char> request(const std::function<void(writer)>& pipe_in);

        private:
        friend struct process;
        explicit session(std::unique_ptr<process_session_impl> p);
        std::unique_ptr<process_session_impl> impl;
    };

    /// Spawns the child the same way read_write does, but leaves it running to serve requests.
    /// cwd() and env() are unsupported.
    session start() const;

    /// The child side of a session. read_message returns the next message on `in`, or nullopt at
    /// EOF before a message starts, and throws on a message cut short. write_message writes one
    /// message to `out` and flushes it. Both streams must be in binary mode.
    static optional<std::vector<char>> read_message(std::istream& in);
    static void write_message(std::ostream& out, const std::vector<char>& data);

    /// Also for the child: returns a binary stream on a private copy of stdout to write the
    /// replies to, and points stdout itself at stderr. Libraries print to stdout through printf
    /// and the file descriptor as well as through std::cout, and a stray byte among the replies
    /// would leave the parent waiting for a message that never ends.
    static std::unique_ptr<std::ostream> take_stdout();

    private:
    std::unique_ptr<process_impl> impl;
};

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
#endif // MIGRAPHX_GUARD_MIGRAPHX_PROCESS_HPP
