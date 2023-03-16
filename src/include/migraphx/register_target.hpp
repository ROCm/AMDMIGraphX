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
#ifndef MIGRAPHX_GUARD_RTGLIB_REGISTER_TARGET_HPP
#define MIGRAPHX_GUARD_RTGLIB_REGISTER_TARGET_HPP

#include <memory>
#include <migraphx/config.hpp>
#include <migraphx/target.hpp>
#include <migraphx/auto_register.hpp>
#include <migraphx/dynamic_loader.hpp>
#include <cstring>
#include <vector>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

void store_target_lib(dynamic_loader lib);

void target_map_init();

void register_target(const target& t);
void unregister_target(const std::string& name);
target make_target(const std::string& name);
std::vector<std::string> get_targets();

struct target_handler
{
    target t;
    target_handler(const target& t_r) : t(t_r) {}
    ~target_handler() { unregister_target(t.name()); }
};

template <class T>
void register_target()
{
    target_map_init();
    static auto t_h = target_handler(T{});
    register_target(t_h.t);
}

struct register_target_action
{
    template <class T>
    static void apply()
    {
        register_target<T>();
    }
};

template <class T>
using auto_register_target = auto_register<register_target_action, T>;

#define MIGRAPHX_REGISTER_TARGET(...) MIGRAPHX_AUTO_REGISTER(register_target_action, __VA_ARGS__)

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
