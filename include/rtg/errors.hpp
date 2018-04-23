#ifndef RTG_GUARD_ERRORS_HPP
#define RTG_GUARD_ERRORS_HPP

#include <exception>
#include <stdexcept>
#include <string>

namespace rtg {

struct exception : std::runtime_error
{
    exception(std::string msg = "")
    : std::runtime_error(msg)
    {}
};

inline exception make_exception(std::string context, std::string message = "")
{
    return {context + ": " + message};
}

inline std::string make_source_context(const std::string& file, int line)
{
    return file + ":" + std::to_string(line);
}

#define RTG_THROW(...) \
    throw rtg::make_exception(rtg::make_source_context(__FILE__, __LINE__), __VA_ARGS__)

} // namespace rtg

#endif
