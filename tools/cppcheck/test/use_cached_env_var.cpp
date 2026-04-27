/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
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
 *
 */

// cppcheck-suppress-file definePrefix
#include <utility>
#include <vector>
#include <string>
#include <map>

#define DECLARE_ENV_VAR(x)                        \
    struct x /* NOLINT */                         \
    {                                             \
        static const char* value() { return #x; } \
    };

bool enabled(const char* name);
bool disabled(const char* name);
std::vector<std::string> env(const char* name);
std::size_t value_of(const char* name, std::size_t fallback = 0);
std::string string_value_of(const char* name, std::string fallback = "");
std::map<std::string, std::string> get_all_envs();

template <class T>
bool enabled(T)
{
    static const bool result = enabled(T::value());
    return result;
}

template <class T>
bool disabled(T)
{
    static const bool result = disabled(T::value());
    return result;
}

template <class T>
std::size_t value_of(T, std::size_t fallback = 0)
{
    static const std::size_t result = value_of(T::value(), fallback);
    return result;
}

template <class T>
std::string string_value_of(T, std::string fallback = "")
{
    static const std::string result = string_value_of(T::value(), std::move(fallback));
    return result;
}

DECLARE_ENV_VAR(TEST_VAR)

void test_env_direct()
{
    // cppcheck-suppress migraphx-UseCachedEnvVar
    auto e = env(TEST_VAR::value());
    (void)e;
}

void test_enabled_direct()
{
    // cppcheck-suppress migraphx-UseCachedEnvVar
    auto e = enabled(TEST_VAR::value());
    (void)e;
}

void test_value_of_direct()
{
    // cppcheck-suppress migraphx-UseCachedEnvVar
    auto e = value_of(TEST_VAR::value());
    (void)e;
}

void test_value_of_direct_fallback()
{
    // cppcheck-suppress migraphx-UseCachedEnvVar
    auto e = value_of(TEST_VAR::value(), 1);
    (void)e;
}
