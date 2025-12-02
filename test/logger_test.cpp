/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include <migraphx/logger.hpp>
#include "test.hpp"

TEST_CASE(logger_set_log_level)
{
    // Test setting different log levels
    migraphx::log::set_severity(migraphx::log::severity::error);
    EXPECT(migraphx::log::is_enabled(migraphx::log::severity::error));
    EXPECT(not migraphx::log::is_enabled(migraphx::log::severity::warn));
    EXPECT(not migraphx::log::is_enabled(migraphx::log::severity::info));

    migraphx::log::set_severity(migraphx::log::severity::info);
    EXPECT(migraphx::log::is_enabled(migraphx::log::severity::error));
    EXPECT(migraphx::log::is_enabled(migraphx::log::severity::warn));
    EXPECT(migraphx::log::is_enabled(migraphx::log::severity::info));
    EXPECT(not migraphx::log::is_enabled(migraphx::log::severity::debug));

    migraphx::log::set_severity(migraphx::log::severity::trace);
    EXPECT(migraphx::log::is_enabled(migraphx::log::severity::error));
    EXPECT(migraphx::log::is_enabled(migraphx::log::severity::warn));
    EXPECT(migraphx::log::is_enabled(migraphx::log::severity::info));
    EXPECT(migraphx::log::is_enabled(migraphx::log::severity::debug));
    EXPECT(migraphx::log::is_enabled(migraphx::log::severity::trace));
}

TEST_CASE(logger_is_enabled)
{
    // Set to INFO level
    migraphx::log::set_severity(migraphx::log::severity::info);

    // Check severity ordering: ERROR < WARN < INFO < DEBUG < TRACE
    EXPECT(migraphx::log::is_enabled(migraphx::log::severity::error));
    EXPECT(migraphx::log::is_enabled(migraphx::log::severity::warn));
    EXPECT(migraphx::log::is_enabled(migraphx::log::severity::info));
    EXPECT(not migraphx::log::is_enabled(migraphx::log::severity::debug));
    EXPECT(not migraphx::log::is_enabled(migraphx::log::severity::trace));
}

TEST_CASE(logger_basic_logging)
{
    // Test that logging doesn't crash or throw
    migraphx::log::set_severity(migraphx::log::severity::trace);

    // Test stream operator
    migraphx::log::error() << "Test error message";
    migraphx::log::warn() << "Test warn message";
    migraphx::log::info() << "Test info message";
    migraphx::log::debug() << "Test debug message";
    migraphx::log::trace() << "Test trace message";

    // Test function call operator
    migraphx::log::error()("Error message");
    migraphx::log::warn()("Warn message");
    migraphx::log::info()("Info message");
    migraphx::log::debug()("Debug message");
    migraphx::log::trace()("Trace message");
}

TEST_CASE(logger_stream_composition)
{
    migraphx::log::set_severity(migraphx::log::severity::info);

    // Test composing multiple values in a single log message
    int value        = 42;
    double pi        = 3.14;
    std::string text = "hello";

    migraphx::log::info() << "Multiple values: " << value << ", " << pi << ", " << text;
}

TEST_CASE(logger_function_call_multiple_args)
{
    migraphx::log::set_severity(migraphx::log::severity::info);

    // Test function call operator with multiple arguments
    migraphx::log::info()("Multiple", " ", "arguments");
    migraphx::log::error()("Error code: ", 404);
}

TEST_CASE(logger_disabled_levels)
{
    // Set to ERROR level - only errors should be enabled
    migraphx::log::set_severity(migraphx::log::severity::error);

    EXPECT(migraphx::log::is_enabled(migraphx::log::severity::error));
    EXPECT(not migraphx::log::is_enabled(migraphx::log::severity::warn));
    EXPECT(not migraphx::log::is_enabled(migraphx::log::severity::info));
    EXPECT(not migraphx::log::is_enabled(migraphx::log::severity::debug));
    EXPECT(not migraphx::log::is_enabled(migraphx::log::severity::trace));

    // These should not cause any output or crash
    migraphx::log::warn() << "This should be suppressed";
    migraphx::log::info() << "This should be suppressed";
    migraphx::log::debug() << "This should be suppressed";
}

TEST_CASE(logger_none_level)
{
    // Set to NONE - nothing should be enabled
    migraphx::log::set_severity(migraphx::log::severity::none);

    EXPECT(not migraphx::log::is_enabled(migraphx::log::severity::error));
    EXPECT(not migraphx::log::is_enabled(migraphx::log::severity::warn));
    EXPECT(not migraphx::log::is_enabled(migraphx::log::severity::info));
    EXPECT(not migraphx::log::is_enabled(migraphx::log::severity::debug));
    EXPECT(not migraphx::log::is_enabled(migraphx::log::severity::trace));
}

TEST_CASE(logger_severity_ordering)
{
    // Test that severity levels are properly ordered
    // NONE(0) < ERROR(1) < WARN(2) < INFO(3) < DEBUG(4) < TRACE(5)

    EXPECT(static_cast<int>(migraphx::log::severity::none) <
           static_cast<int>(migraphx::log::severity::error));
    EXPECT(static_cast<int>(migraphx::log::severity::error) <
           static_cast<int>(migraphx::log::severity::warn));
    EXPECT(static_cast<int>(migraphx::log::severity::warn) <
           static_cast<int>(migraphx::log::severity::info));
    EXPECT(static_cast<int>(migraphx::log::severity::info) <
           static_cast<int>(migraphx::log::severity::debug));
    EXPECT(static_cast<int>(migraphx::log::severity::debug) <
           static_cast<int>(migraphx::log::severity::trace));
}

TEST_CASE(logger_empty_messages)
{
    migraphx::log::set_severity(migraphx::log::severity::info);

    // Test logging empty messages doesn't crash
    migraphx::log::info() << "";
    migraphx::log::error()("");
}

TEST_CASE(logger_special_characters)
{
    migraphx::log::set_severity(migraphx::log::severity::info);

    // Test logging special characters
    migraphx::log::info() << "Special chars: \n\t\\";
    migraphx::log::info()("Unicode: ", "日本語");
}

TEST_CASE(logger_long_messages)
{
    migraphx::log::set_severity(migraphx::log::severity::info);

    // Test logging a long message
    std::string long_msg(1000, 'x');
    migraphx::log::info() << "Long message: " << long_msg;
}

TEST_CASE(logger_conditional_logging)
{
    migraphx::log::set_severity(migraphx::log::severity::info);

    // Test conditional logging based on is_enabled
    if(migraphx::log::is_enabled(migraphx::log::severity::debug))
    {
        // This should not execute
        migraphx::log::debug() << "Should not appear";
    }

    if(migraphx::log::is_enabled(migraphx::log::severity::info))
    {
        // This should execute
        migraphx::log::info() << "Should appear";
    }
}

TEST_CASE(logger_custom_sink)
{
    migraphx::log::set_severity(migraphx::log::severity::info);

    // Track messages received by custom sink
    std::vector<std::string> messages;

    // Add a custom sink
    auto sink_id = migraphx::log::add_sink(
        [&](migraphx::log::severity, std::string_view msg, migraphx::source_location) {
            messages.push_back(std::string(msg));
        },
        migraphx::log::severity::info);

    // Sink ID should be > 0 (stderr is 0)
    EXPECT(sink_id > 0);

    // Log a message
    migraphx::log::info() << "Test custom sink";

    // Verify the custom sink received the message
    EXPECT(not messages.empty());
    EXPECT(messages.back() == "Test custom sink");

    // Remove the sink
    migraphx::log::remove_sink(sink_id);

    // Log another message
    messages.clear();
    migraphx::log::info() << "After removal";

    // Custom sink should not receive this message
    EXPECT(messages.empty());
}

TEST_CASE(logger_sink_level)
{
    migraphx::log::set_severity(migraphx::log::severity::trace);

    std::vector<std::string> messages;

    // Add a custom sink with ERROR level only
    auto sink_id = migraphx::log::add_sink(
        [&](migraphx::log::severity, std::string_view msg, migraphx::source_location) {
            messages.push_back(std::string(msg));
        },
        migraphx::log::severity::error);

    // INFO message should not go to this sink
    migraphx::log::info() << "Info message";
    EXPECT(messages.empty());

    // ERROR message should go to this sink
    migraphx::log::error() << "Error message";
    EXPECT(not messages.empty());
    EXPECT(messages.back() == "Error message");

    // Change sink level to INFO
    messages.clear();
    migraphx::log::set_severity(migraphx::log::severity::info, sink_id);

    // Now INFO should work
    migraphx::log::info() << "Info after level change";
    EXPECT(not messages.empty());
    EXPECT(messages.back() == "Info after level change");

    migraphx::log::remove_sink(sink_id);
}

TEST_CASE(logger_set_severity_default)
{
    // set_severity with default ID should change stderr sink (ID 0)
    migraphx::log::set_severity(migraphx::log::severity::error);

    // Now only ERROR should go to stderr
    // (This just tests it doesn't crash - actual filtering is internal)
    migraphx::log::error() << "Error after set_severity";

    // Reset for other tests
    migraphx::log::set_severity(migraphx::log::severity::info);
}

TEST_CASE(logger_file_sink_returns_id)
{
    migraphx::log::set_severity(migraphx::log::severity::info);

    // add_file_logger should return an ID > 0
    auto file_id =
        migraphx::log::add_file_logger("test_sink_log.txt", migraphx::log::severity::info);
    EXPECT(file_id > 0);

    // Log something
    migraphx::log::info() << "File sink test";

    // Can modify the file sink level
    migraphx::log::set_severity(migraphx::log::severity::error, file_id);

    // Can remove the file sink
    migraphx::log::remove_sink(file_id);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
