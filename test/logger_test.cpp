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
    migraphx::log::set_log_level(migraphx::log::severity::ERROR);
    EXPECT(migraphx::log::is_enabled(migraphx::log::severity::ERROR));
    EXPECT(not migraphx::log::is_enabled(migraphx::log::severity::WARN));
    EXPECT(not migraphx::log::is_enabled(migraphx::log::severity::INFO));

    migraphx::log::set_log_level(migraphx::log::severity::INFO);
    EXPECT(migraphx::log::is_enabled(migraphx::log::severity::ERROR));
    EXPECT(migraphx::log::is_enabled(migraphx::log::severity::WARN));
    EXPECT(migraphx::log::is_enabled(migraphx::log::severity::INFO));
    EXPECT(not migraphx::log::is_enabled(migraphx::log::severity::DEBUG));

    migraphx::log::set_log_level(migraphx::log::severity::TRACE);
    EXPECT(migraphx::log::is_enabled(migraphx::log::severity::ERROR));
    EXPECT(migraphx::log::is_enabled(migraphx::log::severity::WARN));
    EXPECT(migraphx::log::is_enabled(migraphx::log::severity::INFO));
    EXPECT(migraphx::log::is_enabled(migraphx::log::severity::DEBUG));
    EXPECT(migraphx::log::is_enabled(migraphx::log::severity::TRACE));
}

TEST_CASE(logger_is_enabled)
{
    // Set to INFO level
    migraphx::log::set_log_level(migraphx::log::severity::INFO);
    
    // Check severity ordering: ERROR < WARN < INFO < DEBUG < TRACE
    EXPECT(migraphx::log::is_enabled(migraphx::log::severity::ERROR));
    EXPECT(migraphx::log::is_enabled(migraphx::log::severity::WARN));
    EXPECT(migraphx::log::is_enabled(migraphx::log::severity::INFO));
    EXPECT(not migraphx::log::is_enabled(migraphx::log::severity::DEBUG));
    EXPECT(not migraphx::log::is_enabled(migraphx::log::severity::TRACE));
}

TEST_CASE(logger_basic_logging)
{
    // Test that logging doesn't crash or throw
    migraphx::log::set_log_level(migraphx::log::severity::TRACE);
    
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
    migraphx::log::set_log_level(migraphx::log::severity::INFO);
    
    // Test composing multiple values in a single log message
    int value = 42;
    double pi = 3.14;
    std::string text = "hello";
    
    migraphx::log::info() << "Multiple values: " << value << ", " << pi << ", " << text;
}

TEST_CASE(logger_function_call_multiple_args)
{
    migraphx::log::set_log_level(migraphx::log::severity::INFO);
    
    // Test function call operator with multiple arguments
    migraphx::log::info()("Multiple", " ", "arguments");
    migraphx::log::error()("Error code: ", 404);
}

TEST_CASE(logger_disabled_levels)
{
    // Set to ERROR level - only errors should be enabled
    migraphx::log::set_log_level(migraphx::log::severity::ERROR);
    
    EXPECT(migraphx::log::is_enabled(migraphx::log::severity::ERROR));
    EXPECT(not migraphx::log::is_enabled(migraphx::log::severity::WARN));
    EXPECT(not migraphx::log::is_enabled(migraphx::log::severity::INFO));
    EXPECT(not migraphx::log::is_enabled(migraphx::log::severity::DEBUG));
    EXPECT(not migraphx::log::is_enabled(migraphx::log::severity::TRACE));
    
    // These should not cause any output or crash
    migraphx::log::warn() << "This should be suppressed";
    migraphx::log::info() << "This should be suppressed";
    migraphx::log::debug() << "This should be suppressed";
}

TEST_CASE(logger_none_level)
{
    // Set to NONE - nothing should be enabled
    migraphx::log::set_log_level(migraphx::log::severity::NONE);
    
    EXPECT(not migraphx::log::is_enabled(migraphx::log::severity::ERROR));
    EXPECT(not migraphx::log::is_enabled(migraphx::log::severity::WARN));
    EXPECT(not migraphx::log::is_enabled(migraphx::log::severity::INFO));
    EXPECT(not migraphx::log::is_enabled(migraphx::log::severity::DEBUG));
    EXPECT(not migraphx::log::is_enabled(migraphx::log::severity::TRACE));
}

TEST_CASE(logger_severity_ordering)
{
    // Test that severity levels are properly ordered
    // NONE(0) < ERROR(1) < WARN(2) < INFO(3) < DEBUG(4) < TRACE(5)
    
    EXPECT(static_cast<int>(migraphx::log::severity::NONE) < 
           static_cast<int>(migraphx::log::severity::ERROR));
    EXPECT(static_cast<int>(migraphx::log::severity::ERROR) < 
           static_cast<int>(migraphx::log::severity::WARN));
    EXPECT(static_cast<int>(migraphx::log::severity::WARN) < 
           static_cast<int>(migraphx::log::severity::INFO));
    EXPECT(static_cast<int>(migraphx::log::severity::INFO) < 
           static_cast<int>(migraphx::log::severity::DEBUG));
    EXPECT(static_cast<int>(migraphx::log::severity::DEBUG) < 
           static_cast<int>(migraphx::log::severity::TRACE));
}

TEST_CASE(logger_empty_messages)
{
    migraphx::log::set_log_level(migraphx::log::severity::INFO);
    
    // Test logging empty messages doesn't crash
    migraphx::log::info() << "";
    migraphx::log::error()("");
}

TEST_CASE(logger_special_characters)
{
    migraphx::log::set_log_level(migraphx::log::severity::INFO);
    
    // Test logging special characters
    migraphx::log::info() << "Special chars: \n\t\\";
    migraphx::log::info()("Unicode: ", "日本語");
}

TEST_CASE(logger_long_messages)
{
    migraphx::log::set_log_level(migraphx::log::severity::INFO);
    
    // Test logging a very long message
    std::string long_msg(1000, 'x');
    migraphx::log::info() << "Long message: " << long_msg;
}

TEST_CASE(logger_conditional_logging)
{
    migraphx::log::set_log_level(migraphx::log::severity::INFO);
    
    // Test conditional logging based on is_enabled
    if(migraphx::log::is_enabled(migraphx::log::severity::DEBUG))
    {
        // This should not execute
        migraphx::log::debug() << "Should not appear";
    }
    
    if(migraphx::log::is_enabled(migraphx::log::severity::INFO))
    {
        // This should execute
        migraphx::log::info() << "Should appear";
    }
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }

