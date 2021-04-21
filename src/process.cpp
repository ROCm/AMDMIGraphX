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

int exec(const std::string& cmd, const std::function<void(const char*)>& std_out)
{
    int ec = 0;
    if(enabled(MIGRAPHX_TRACE_CMD_EXECUTE{}))
        std::cout << cmd << std::endl;
    std::array<char, 128> buffer;
    auto closer = [&](FILE* stream) {
        auto status = pclose(stream);
        ec          = WIFEXITED(status) ? 0 : WEXITSTATUS(status); // NOLINT
    };
    {
        // TODO: Use execve instead of popen
        std::unique_ptr<FILE, decltype(closer)> pipe(popen(cmd.c_str(), "r"), closer); // NOLINT
        if(!pipe)
            MIGRAPHX_THROW("popen() failed: " + cmd);
        while(fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr)
            std_out(buffer.data());
    }
    return ec;
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
    auto ec = migraphx::exec(impl->get_command(), redirect_to(std::cout));
    if(ec != 0)
        MIGRAPHX_THROW("Command " + impl->get_command() + " exited with status " +
                       std::to_string(ec));
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
