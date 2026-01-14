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
#include <atomic>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <map>
#include <mutex>
#include <thread>
#include <vector>

// Helper to verify per-thread message ordering in concurrent logging tests.
// Messages must follow the format "Thread %u message %d".
// Verifies that messages from each thread arrive in strictly increasing order.
// If verify_complete is true, also verifies that each thread sent exactly msgs_per_thread messages.
void verify_per_thread_ordering(const std::vector<std::string>& messages,
                                unsigned int num_threads,
                                int msgs_per_thread  = -1,
                                bool verify_complete = false)
{
    std::vector<int> last_index(num_threads, -1);
    std::vector<int> msg_count(num_threads, 0);

    for(const auto& msg : messages)
    {
        unsigned int thread_id = 0;
        int msg_index          = 0;
        int parsed = std::sscanf(msg.c_str(), "Thread %u message %d", &thread_id, &msg_index);

        EXPECT(parsed == 2);
        EXPECT(thread_id < num_threads);
        if(msgs_per_thread > 0)
        {
            EXPECT(msg_index >= 0);
            EXPECT(msg_index < msgs_per_thread);
        }
        EXPECT(msg_index > last_index[thread_id]);
        last_index[thread_id] = msg_index;
        msg_count[thread_id]++;
    }

    if(verify_complete)
    {
        for(unsigned int t = 0; t < num_threads; ++t)
        {
            EXPECT(msg_count[t] == msgs_per_thread);
        }
    }
}

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

// =============================================================================
// Multithreading Tests
// =============================================================================

static unsigned int get_thread_count(unsigned int min_threads = 8)
{
    static unsigned int hw = std::thread::hardware_concurrency();
    return hw == 0 ? min_threads : std::max(min_threads, hw);
}

// Test multiple threads logging simultaneously to the same sink
TEST_CASE(logger_concurrent_logging)
{
    migraphx::log::set_severity(migraphx::log::severity::none);

    std::vector<std::string> messages;

    auto sink_id = migraphx::log::add_sink(
        [&](migraphx::log::severity, std::string_view msg, migraphx::source_location) {
            messages.push_back(std::string(msg));
        },
        migraphx::log::severity::trace);

    const unsigned int num_threads    = get_thread_count(8);
    constexpr int messages_per_thread = 100;

    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    for(unsigned int t = 0; t < num_threads; ++t)
    {
        threads.emplace_back([t]() {
            for(int i = 0; i < messages_per_thread; ++i)
            {
                migraphx::log::info() << "Thread " << t << " message " << i;
            }
        });
    }

    for(auto& thread : threads)
        thread.join();

    migraphx::log::remove_sink(sink_id);

    // Verify all messages were received
    EXPECT(messages.size() == num_threads * messages_per_thread);

    // Verify message content, per-thread ordering, and completeness
    verify_per_thread_ordering(messages, num_threads, messages_per_thread, true);
}

TEST_CASE(logger_concurrent_is_enabled)
{
    // Test is_enabled() called from multiple threads while severity changes
    migraphx::log::set_severity(migraphx::log::severity::info);

    std::atomic<bool> stop{false};
    std::atomic<int> enabled_count{0};
    std::atomic<int> check_count{0};

    const unsigned int num_reader_threads = get_thread_count(4);
    std::vector<std::thread> readers;
    readers.reserve(num_reader_threads);

    // Reader threads continuously call is_enabled()
    for(unsigned int t = 0; t < num_reader_threads; ++t)
    {
        readers.emplace_back([&]() {
            while(not stop.load())
            {
                if(migraphx::log::is_enabled(migraphx::log::severity::info))
                    enabled_count.fetch_add(1);
                check_count.fetch_add(1);
            }
        });
    }

    // Writer thread changes severity levels
    std::thread writer([&]() {
        for(int i = 0; i < 100; ++i)
        {
            migraphx::log::set_severity(migraphx::log::severity::trace);
            migraphx::log::set_severity(migraphx::log::severity::error);
            migraphx::log::set_severity(migraphx::log::severity::info);
        }
    });

    writer.join();
    stop.store(true);

    for(auto& reader : readers)
        reader.join();

    // Verify substantial work was done by all threads
    // Each thread should have performed many checks during the 100 severity change iterations
    EXPECT(check_count.load() >= num_reader_threads * 100);

    // Some checks should have returned true (when severity was info or trace)
    // and some false (when severity was error), so enabled_count should be
    // between 0 and check_count (exclusive on both ends in practice)
    EXPECT(enabled_count.load() > 0);
    EXPECT(enabled_count.load() < check_count.load());
}

TEST_CASE(logger_concurrent_add_remove_sink)
{
    // Test adding and removing sinks while other threads are logging
    migraphx::log::set_severity(migraphx::log::severity::none);

    std::atomic<bool> stop{false};
    std::atomic<int> log_count{0};

    // Base sink that always exists - captures all messages for verification
    std::vector<std::string> base_messages;
    std::vector<migraphx::log::severity> base_severities;
    auto base_sink_id = migraphx::log::add_sink(
        [&](migraphx::log::severity s, std::string_view msg, migraphx::source_location) {
            base_messages.push_back(std::string(msg));
            base_severities.push_back(s);
        },
        migraphx::log::severity::info);

    // Counter for messages received by dynamic sinks
    std::atomic<int> dynamic_sink_count{0};

    // Logger threads continuously log messages
    const unsigned int num_logger_threads = get_thread_count(4);
    std::vector<std::thread> loggers;
    loggers.reserve(num_logger_threads);

    for(unsigned int t = 0; t < num_logger_threads; ++t)
    {
        loggers.emplace_back([&, t]() {
            int i = 0;
            while(not stop.load())
            {
                migraphx::log::info() << "Thread " << t << " message " << i++;
                log_count.fetch_add(1);
            }
        });
    }

    // Sink manager thread adds and removes sinks
    std::thread sink_manager([&]() {
        using namespace std::chrono_literals;
        for(int i = 0; i < 25; ++i)
        {
            auto id = migraphx::log::add_sink(
                [&](migraphx::log::severity, std::string_view, migraphx::source_location) {
                    dynamic_sink_count.fetch_add(1);
                },
                migraphx::log::severity::info);
            // Brief pause to let some logging happen (yield is insufficient on fast CI)
            std::this_thread::sleep_for(1ms);
            migraphx::log::remove_sink(id);
        }
    });

    sink_manager.join();
    stop.store(true);

    for(auto& logger : loggers)
        logger.join();

    // Remove base sink before assertions
    migraphx::log::remove_sink(base_sink_id);

    // Verify logging occurred
    EXPECT(log_count.load() > 0);

    // Verify base sink received all messages
    EXPECT(base_messages.size() == log_count.load());

    // Verify all messages have correct severity
    for(auto s : base_severities)
    {
        EXPECT(s == migraphx::log::severity::info);
    }

    // Verify message format and per-thread ordering
    verify_per_thread_ordering(base_messages, num_logger_threads);

    // Verify dynamic sinks received some messages while they were active
    EXPECT(dynamic_sink_count.load() > 0);
}

TEST_CASE(logger_concurrent_set_severity)
{
    // Test changing severity on sinks while logging
    migraphx::log::set_severity(migraphx::log::severity::none);

    std::vector<std::string> messages;
    std::atomic<int> total_info_logged{0};
    std::atomic<int> total_debug_logged{0};

    auto sink_id = migraphx::log::add_sink(
        [&](migraphx::log::severity, std::string_view msg, migraphx::source_location) {
            messages.push_back(std::string(msg));
        },
        migraphx::log::severity::info);

    std::atomic<bool> stop{false};

    // Logger threads
    const unsigned int num_logger_threads = get_thread_count(4);
    std::vector<std::thread> loggers;
    loggers.reserve(num_logger_threads);

    for(unsigned int t = 0; t < num_logger_threads; ++t)
    {
        loggers.emplace_back([&, t]() {
            int i = 0;
            while(not stop.load())
            {
                migraphx::log::info() << "Thread " << t << " message " << i++;
                total_info_logged.fetch_add(1);
                migraphx::log::debug() << "Debug " << t;
                total_debug_logged.fetch_add(1);
            }
        });
    }

    // Severity changer thread cycles through: trace -> error -> info
    // - trace: allows info AND debug messages
    // - error: blocks both info and debug
    // - info: allows info, blocks debug
    std::thread changer([&]() {
        using namespace std::chrono_literals;
        for(int i = 0; i < 25; ++i)
        {
            migraphx::log::set_severity(migraphx::log::severity::trace, sink_id);
            std::this_thread::sleep_for(1ms);
            migraphx::log::set_severity(migraphx::log::severity::error, sink_id);
            std::this_thread::sleep_for(1ms);
            migraphx::log::set_severity(migraphx::log::severity::info, sink_id);
        }
    });

    changer.join();
    stop.store(true);

    for(auto& logger : loggers)
        logger.join();

    // Remove sink first to ensure cleanup even if test fails
    migraphx::log::remove_sink(sink_id);

    // Count info vs debug messages that were actually received
    std::vector<std::string> info_messages;
    std::vector<std::string> debug_messages;
    for(const auto& msg : messages)
    {
        if(msg.find("Thread ") == 0)
            info_messages.push_back(msg);
        else if(msg.find("Debug ") == 0)
            debug_messages.push_back(msg);
    }

    // Verify severity changes had effect:
    // 1. Some info messages received (when severity was trace or info)
    EXPECT(not info_messages.empty());

    // 2. Some debug messages received (only when severity was trace)
    EXPECT(not debug_messages.empty());

    // 3. Not all info messages received (some filtered when severity was error)
    EXPECT(info_messages.size() < total_info_logged.load());

    // 4. Not all debug messages received (filtered when severity was error or info)
    EXPECT(debug_messages.size() < total_debug_logged.load());

    // 5. Fewer debug than info messages (debug only passes at trace level)
    EXPECT(debug_messages.size() < info_messages.size());

    // 6. Per-thread ordering preserved for info messages
    verify_per_thread_ordering(info_messages, num_logger_threads);
}

TEST_CASE(logger_concurrent_multiple_sinks)
{
    // Test logging to multiple sinks concurrently
    migraphx::log::set_severity(migraphx::log::severity::none);

    const unsigned int num_sinks = std::min(8u, get_thread_count(4));
    std::vector<std::vector<std::string>> sink_messages(num_sinks);
    std::vector<size_t> sink_ids(num_sinks);

    // Add multiple sinks
    for(unsigned int s = 0; s < num_sinks; ++s)
    {
        sink_ids[s] = migraphx::log::add_sink(
            [&, s](migraphx::log::severity, std::string_view msg, migraphx::source_location) {
                sink_messages[s].push_back(std::string(msg));
            },
            migraphx::log::severity::info);
    }

    const unsigned int num_threads = get_thread_count(4);
    constexpr int msgs_per_thread  = 50;

    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    for(unsigned int t = 0; t < num_threads; ++t)
    {
        threads.emplace_back([t]() {
            for(int i = 0; i < msgs_per_thread; ++i)
            {
                migraphx::log::info() << "Thread " << t << " message " << i;
            }
        });
    }

    for(auto& thread : threads)
        thread.join();

    // Clean up first to ensure cleanup even if test fails
    for(unsigned int s = 0; s < num_sinks; ++s)
        migraphx::log::remove_sink(sink_ids[s]);

    // Each sink should have received all messages with correct per-thread ordering and completeness
    for(unsigned int s = 0; s < num_sinks; ++s)
    {
        EXPECT(sink_messages[s].size() == num_threads * msgs_per_thread);
        verify_per_thread_ordering(sink_messages[s], num_threads, msgs_per_thread, true);
    }

    // All sinks should have received identical messages (same content, same order)
    for(unsigned int s = 1; s < num_sinks; ++s)
    {
        EXPECT(sink_messages[s] == sink_messages[0]);
    }
}

TEST_CASE(logger_concurrent_file_sink)
{
    // Test concurrent logging to a file sink
    migraphx::log::set_severity(migraphx::log::severity::none);

    const char* log_path = "/tmp/migraphx_concurrent_test.log";

    // Remove any existing file
    std::remove(log_path);

    auto file_id = migraphx::log::add_file_logger(log_path, migraphx::log::severity::info);

    const unsigned int num_threads    = get_thread_count(4);
    constexpr int messages_per_thread = 25;

    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    for(unsigned int t = 0; t < num_threads; ++t)
    {
        threads.emplace_back([t]() {
            for(int i = 0; i < messages_per_thread; ++i)
            {
                migraphx::log::info() << "Thread " << t << " message " << i;
            }
        });
    }

    for(auto& thread : threads)
        thread.join();

    migraphx::log::remove_sink(file_id);

    // Read file and verify format and content
    std::ifstream file(log_path);
    std::vector<std::string> messages;
    std::vector<std::string> raw_lines;
    std::string line;
    while(std::getline(file, line))
    {
        raw_lines.push_back(line);
        // Extract message portion after timestamp/severity prefix
        auto pos = line.find("Thread ");
        if(pos != std::string::npos)
            messages.push_back(line.substr(pos));
    }

    EXPECT(messages.size() == num_threads * messages_per_thread);

    // Verify file format: each line should have timestamp, severity, location, message
    // Format: "YYYY-MM-DD HH:MM:SS.microsec [INFO] [file:line] Thread X message Y"
    for(const auto& raw_line : raw_lines)
    {
        // Should contain timestamp pattern (starts with year)
        EXPECT(raw_line.size() > 26);
        EXPECT(raw_line[4] == '-');  // YYYY-
        EXPECT(raw_line[7] == '-');  // MM-
        EXPECT(raw_line[10] == ' '); // DD
        EXPECT(raw_line[13] == ':'); // HH:

        // Should contain severity marker
        EXPECT(raw_line.find("[INFO]") != std::string::npos);

        // Should contain source location
        EXPECT(raw_line.find("logger_test.cpp:") != std::string::npos);
    }

    // Verify per-thread ordering
    verify_per_thread_ordering(messages, num_threads, messages_per_thread, true);

    std::remove(log_path);
}

TEST_CASE(logger_stress_test)
{
    // High-contention stress test - use more threads for higher contention
    migraphx::log::set_severity(migraphx::log::severity::none);

    std::vector<std::string> messages;
    std::vector<migraphx::log::severity> severities;

    auto sink_id = migraphx::log::add_sink(
        [&](migraphx::log::severity s, std::string_view msg, migraphx::source_location) {
            messages.push_back(std::string(msg));
            severities.push_back(s);
        },
        migraphx::log::severity::trace);

    const unsigned int num_threads = get_thread_count(16);
    constexpr int iterations       = 100;

    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    for(unsigned int t = 0; t < num_threads; ++t)
    {
        threads.emplace_back([t]() {
            for(int i = 0; i < iterations; ++i)
            {
                // Mix of different severity levels, all with consistent format
                switch(i % 5)
                {
                case 0: migraphx::log::error() << "Thread " << t << " message " << i; break;
                case 1: migraphx::log::warn() << "Thread " << t << " message " << i; break;
                case 2: migraphx::log::info() << "Thread " << t << " message " << i; break;
                case 3: migraphx::log::debug() << "Thread " << t << " message " << i; break;
                case 4: migraphx::log::trace() << "Thread " << t << " message " << i; break;
                }
            }
        });
    }

    for(auto& thread : threads)
        thread.join();

    // Remove sink first to ensure cleanup even if test fails
    migraphx::log::remove_sink(sink_id);

    EXPECT(messages.size() == num_threads * iterations);
    EXPECT(severities.size() == num_threads * iterations);

    // Verify per-thread ordering and completeness
    verify_per_thread_ordering(messages, num_threads, iterations, true);

    // Verify severity distribution: each level should appear exactly 20% of the time
    // (iterations % 5 == 0, so distribution is deterministic)
    std::map<migraphx::log::severity, int> severity_counts;
    for(auto s : severities)
        severity_counts[s]++;

    const int expected_per_level = (num_threads * iterations) / 5;
    EXPECT(severity_counts[migraphx::log::severity::error] == expected_per_level);
    EXPECT(severity_counts[migraphx::log::severity::warn] == expected_per_level);
    EXPECT(severity_counts[migraphx::log::severity::info] == expected_per_level);
    EXPECT(severity_counts[migraphx::log::severity::debug] == expected_per_level);
    EXPECT(severity_counts[migraphx::log::severity::trace] == expected_per_level);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
