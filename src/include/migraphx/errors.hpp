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
#ifndef MIGRAPHX_GUARD_ERRORS_HPP
#define MIGRAPHX_GUARD_ERRORS_HPP

#include <exception>
#include <stdexcept>
#include <string>
#include <migraphx/check_capture.hpp>
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
inline std::string make_source_context(const std::string& file, int line, const std::string& fname)
{
    return file + ":" + std::to_string(line) + ": " + fname;
}

// NOLINTNEXTLINE
#define MIGRAPHX_MAKE_SOURCE_CTX() migraphx::make_source_context(__FILE__, __LINE__, __func__)

/**
 * @brief Throw an exception with context information
 */
#define MIGRAPHX_THROW(...) throw migraphx::make_exception(MIGRAPHX_MAKE_SOURCE_CTX(), __VA_ARGS__)

/**
 * @brief Append the failing-check expression to a user-supplied message.
 *
 * Produces a message in the same shape as the test framework's
 * `FAILED: <expr> [ <values> ]` output, but as a single line suitable for an
 * exception: `<message>: <check> [ <values> ]`. The bracketed values clause is
 * omitted when capture could not decompose the predicate (i.e. the printed
 * value is the same as the stringified expression, e.g. literals).
 */
inline std::string
make_failed_check_message(const std::string& message, const char* check, const std::string& values)
{
    if(check == nullptr or *check == '\0')
        return message;
    std::string result = message.empty() ? std::string{} : message + ": ";
    result += check;
    if(not values.empty() and values != check)
        result += " [ " + values + " ]";
    return result;
}

/**
 * @brief Throw an exception when @p cond does not hold.
 *
 * Like the test framework's `EXPECT`, the argument is the predicate that is
 * *expected to be true*. The condition is evaluated through an
 * operator-decomposing capture so that, on failure, the thrown exception
 * carries the user-supplied message together with the stringified condition
 * and the actual operand values, matching the test framework's
 * `<expr> [ <values> ]` format (e.g. `"OP: mismatch: x == y [ 5 == 7 ]"`).
 * The condition is evaluated exactly once.
 *
 * Usage:
 *     MIGRAPHX_EXPECT(input_ndim == expected_ndim, "OP: input dimension mismatch");
 */
// NOLINTNEXTLINE
#define MIGRAPHX_EXPECT(cond, ...)                                                            \
    do                                                                                        \
    {                                                                                         \
        auto&& migraphx_check_expr_ = MIGRAPHX_CHECK_CAPTURE(cond);                           \
        if(not bool(migraphx_check_expr_.value()))                                            \
            MIGRAPHX_THROW(migraphx::make_failed_check_message(                               \
                __VA_ARGS__,                                                                  \
                #cond,                                                                        \
                migraphx::check_capture_detail::expression_to_string(migraphx_check_expr_))); \
    } while(0)

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
