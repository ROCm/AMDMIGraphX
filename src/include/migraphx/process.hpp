#ifndef MIGRAPHX_GUARD_MIGRAPHX_PROCESS_HPP
#define MIGRAPHX_GUARD_MIGRAPHX_PROCESS_HPP

#include <migraphx/config.hpp>
#include <migraphx/filesystem.hpp>
#include <string>
#include <memory>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct process_impl;

struct process
{
    process(const std::string& cmd);

    // move constructor
    process(process&&) noexcept;

    // copy assignment operator
    process& operator=(process rhs);

    ~process() noexcept;

    process& cwd(const fs::path& p);

    void exec();

    private:
    std::unique_ptr<process_impl> impl;
};

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
#endif // MIGRAPHX_GUARD_MIGRAPHX_PROCESS_HPP
