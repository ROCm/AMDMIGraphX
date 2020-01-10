#ifndef MIGRAPHX_GUARD_ERRORS_HPP
#define MIGRAPHX_GUARD_ERRORS_HPP

#include <exception>
#include <stdexcept>
#include <string>
#include <migraphx/config.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

/// Represents exceptions that can be thrown by migraphxlib
struct exception : std::runtime_error
{
    unsigned int error;
    exception(unsigned int e = 0, const std::string& msg = "") : std::runtime_error(msg), error(e)
    {
    }
};

/**
 * @brief Create an exception object
 *
 * @param context A message that says where the exception occurred
 * @param message Custom message for the error
 * @return Exceptions
 */
inline exception make_exception(const std::string& context, const std::string& message = "")
{
    return {0, context + ": " + message};
}

inline exception
make_exception(const std::string& context, unsigned int e, const std::string& message = "")
{
    return {e, context + ": " + message};
}

/**
 * @brief Create a message of a file location
 *
 * @param file The filename
 * @param line The line number
 *
 * @return A string that represents the file location
 */
inline std::string make_source_context(const std::string& file, int line)
{
    return file + ":" + std::to_string(line);
}

/**
 * @brief Throw an exception with context information
 */
#define MIGRAPHX_THROW(...) \
    throw migraphx::make_exception(migraphx::make_source_context(__FILE__, __LINE__), __VA_ARGS__)

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
