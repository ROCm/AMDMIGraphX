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
#include <migraphx/env.hpp>
#include <test.hpp>
#include <cstdlib>
#include <string>

MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_TEST_ENV_UNSET_VALUE)
MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_TEST_ENV_UNSET_STRING)
MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_TEST_ENV_SET_VALUE)
MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_TEST_ENV_SET_STRING)

TEST_CASE(value_of_unset_returns_the_fallback_of_each_call)
{
    EXPECT(migraphx::value_of(MIGRAPHX_TEST_ENV_UNSET_VALUE{}, 1) == 1);
    EXPECT(migraphx::value_of(MIGRAPHX_TEST_ENV_UNSET_VALUE{}, 4) == 4);
    EXPECT(migraphx::value_of(MIGRAPHX_TEST_ENV_UNSET_VALUE{}) == 0);
}

TEST_CASE(string_value_of_unset_returns_the_fallback_of_each_call)
{
    EXPECT(migraphx::string_value_of(MIGRAPHX_TEST_ENV_UNSET_STRING{}, "a") == "a");
    EXPECT(migraphx::string_value_of(MIGRAPHX_TEST_ENV_UNSET_STRING{}, "b") == "b");
    EXPECT(migraphx::string_value_of(MIGRAPHX_TEST_ENV_UNSET_STRING{}).empty());
}

TEST_CASE(value_of_set_overrides_every_fallback)
{
    EXPECT(migraphx::value_of(MIGRAPHX_TEST_ENV_SET_VALUE{}, 1) == 16);
    EXPECT(migraphx::value_of(MIGRAPHX_TEST_ENV_SET_VALUE{}, 4) == 16);
    EXPECT(migraphx::get_all_envs().at("MIGRAPHX_TEST_ENV_SET_VALUE") == "16");
}

TEST_CASE(string_value_of_set_overrides_every_fallback)
{
    EXPECT(migraphx::string_value_of(MIGRAPHX_TEST_ENV_SET_STRING{}, "a") == "x");
    EXPECT(migraphx::string_value_of(MIGRAPHX_TEST_ENV_SET_STRING{}, "b") == "x");
    EXPECT(migraphx::get_all_envs().at("MIGRAPHX_TEST_ENV_SET_STRING") == "x");
}

int main(int argc, const char* argv[])
{
#ifdef _WIN32
    _putenv_s("MIGRAPHX_TEST_ENV_SET_VALUE", "16");
    _putenv_s("MIGRAPHX_TEST_ENV_SET_STRING", "x");
#else
    setenv("MIGRAPHX_TEST_ENV_SET_VALUE", "16", 1);
    setenv("MIGRAPHX_TEST_ENV_SET_STRING", "x", 1);
#endif
    test::run(argc, argv);
}
