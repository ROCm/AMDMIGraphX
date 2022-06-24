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
#ifndef MIGRAPHX_GUARD_MIGRAPHLIB_PASS_MANAGER_HPP
#define MIGRAPHX_GUARD_MIGRAPHLIB_PASS_MANAGER_HPP

#include <migraphx/config.hpp>
#include <migraphx/pass.hpp>
#include <migraphx/tracer.hpp>
#include <vector>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct module_pass_manager
{
    module_pass_manager()                                  = default;
    module_pass_manager(const module_pass_manager&)        = delete;
    virtual module& get_module()                           = 0;
    virtual module* create_module(const std::string& name) = 0;
    virtual module* get_common_parent()                    = 0;
    virtual void run_pass(const pass& p)                   = 0;

    protected:
    virtual ~module_pass_manager() {}
};

void run_passes(module& mod, const std::vector<pass>& passes, tracer trace = tracer{});
void run_passes(program& prog, const std::vector<pass>& passes, tracer trace = tracer{});

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
