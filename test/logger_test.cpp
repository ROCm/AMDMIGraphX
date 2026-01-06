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

#include <migraphx/logger.hpp>
#include "test.hpp"
#include <fstream>

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
    migraphx::log::info()("Multiple arguments with different types: ", 42, ", ", "hello");
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

    // Track messages to verify suppression
    std::vector<std::string> messages;
    auto sink_id = migraphx::log::add_sink(
        [&](migraphx::log::severity, std::string_view msg, migraphx::source_location) {
            messages.push_back(std::string(msg));
        },
        migraphx::log::severity::error);

    // These should not cause any output
    migraphx::log::warn() << "This should be suppressed";
    migraphx::log::info() << "This should be suppressed";
    migraphx::log::debug() << "This should be suppressed";

    // Verify no messages were captured (they were suppressed)
    EXPECT(messages.empty());

    migraphx::log::remove_sink(sink_id);
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

    EXPECT(migraphx::log::severity::none < migraphx::log::severity::error);
    EXPECT(migraphx::log::severity::error < migraphx::log::severity::warn);
    EXPECT(migraphx::log::severity::warn < migraphx::log::severity::info);
    EXPECT(migraphx::log::severity::info < migraphx::log::severity::debug);
    EXPECT(migraphx::log::severity::debug < migraphx::log::severity::trace);
}

TEST_CASE(logger_set_severity_default)
{
    // set_severity with default ID should change stderr sink (ID 0)
    migraphx::log::set_severity(migraphx::log::severity::error);

    // Now only ERROR should go to stderr
    // (This just tests it doesn't crash - actual filtering is internal)
    migraphx::log::error() << "This message should appear";
    migraphx::log::info() << "This message should not appear";
}

TEST_CASE(logger_empty_messages)
{
    migraphx::log::set_severity(migraphx::log::severity::info);

    // Test logging empty messages doesn't crash
    migraphx::log::info() << "";
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
        migraphx::log::debug() << "This message should not appear";
    }

    if(migraphx::log::is_enabled(migraphx::log::severity::info))
    {
        // This should execute
        migraphx::log::info() << "This message should appear";
    }
}

TEST_CASE(logger_custom_sink)
{
    // Prevent stderr output
    migraphx::log::set_severity(migraphx::log::severity::none);

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
    // Prevent stderr output
    migraphx::log::set_severity(migraphx::log::severity::none);

    std::vector<std::string> messages;

    // Add a custom sink with ERROR level only
    auto sink_id = migraphx::log::add_sink(
        [&](migraphx::log::severity, std::string_view msg, migraphx::source_location) {
            messages.push_back(std::string(msg));
        },
        migraphx::log::severity::error);

    // INFO message should not go to this sink
    migraphx::log::info() << "This message should not appear";
    EXPECT(messages.empty());

    // ERROR message should go to this sink
    migraphx::log::error() << "This message should appear";
    EXPECT(not messages.empty());
    EXPECT(messages.back() == "This message should appear");

    // Change sink level to INFO
    messages.clear();
    migraphx::log::set_severity(migraphx::log::severity::info, sink_id);

    // Now INFO should work
    migraphx::log::info() << "This second message should appear";
    EXPECT(not messages.empty());
    // cppcheck-suppress containerOutOfBounds
    EXPECT(messages.back() ==
           "This second message should appear"); // suppression is needed due to false positive

    migraphx::log::remove_sink(sink_id);
}

TEST_CASE(logger_file_sink)
{
    // Prevent stderr output
    migraphx::log::set_severity(migraphx::log::severity::none);

    // add_file_logger should return an ID > 0
    auto file_id =
        migraphx::log::add_file_logger("/tmp/migraphx_test_log.txt", migraphx::log::severity::info);
    EXPECT(file_id > 0);

    // Log something
    migraphx::log::info() << "File sink test";

    // Log a debug message that should not be written to the file
    migraphx::log::debug() << "This message should not be written to the file";

    // Can modify the file sink level
    migraphx::log::set_severity(migraphx::log::severity::debug, file_id);

    // Log a debug message that should be written to the file
    migraphx::log::debug() << "This message should be written to the file";

    // Verify the file has two messages
    std::ifstream file("/tmp/migraphx_test_log.txt");
    std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    EXPECT(content.find("File sink test") != std::string::npos);
    EXPECT(content.find("This message should be written to the file") != std::string::npos);
    EXPECT(content.find("This message should not be written to the file") == std::string::npos);

    // Remove the file
    std::remove("/tmp/migraphx_test_log.txt");

    // Can remove the file sink
    migraphx::log::remove_sink(file_id);
}

TEST_CASE(logger_file_sink_existing_file)
{
    // Prevent stderr output
    migraphx::log::set_severity(migraphx::log::severity::none);

    const char* log_path = "/tmp/migraphx_test_existing_log.txt";

    // Create a file logger and write some content
    auto file_id1 = migraphx::log::add_file_logger(log_path, migraphx::log::severity::info);
    EXPECT(file_id1 > 0);
    migraphx::log::info() << "First message";
    migraphx::log::remove_sink(file_id1);

    // Add a file logger to the same path (file now exists)
    auto file_id2 = migraphx::log::add_file_logger(log_path, migraphx::log::severity::info);
    EXPECT(file_id2 > 0);

    // Log another message
    migraphx::log::info() << "Second message";

    // Verify the file has two messages
    std::ifstream file(log_path);
    std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    EXPECT(content.find("First message") != std::string::npos);
    EXPECT(content.find("Second message") != std::string::npos);

    // Remove the file
    std::remove(log_path);

    // Clean up
    migraphx::log::remove_sink(file_id2);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
