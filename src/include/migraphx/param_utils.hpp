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
 *
 */
#ifndef MIGRAPHX_GUARD_MIGRAPHX_PARAM_UTILS_HPP
#define MIGRAPHX_GUARD_MIGRAPHX_PARAM_UTILS_HPP

#include <migraphx/config.hpp>
#include <migraphx/instruction_ref.hpp>
#include <migraphx/module_ref.hpp>
#include <unordered_set>
#include <vector>
#include <string>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

MIGRAPHX_EXPORT std::string param_name(std::size_t i, const std::string& prefix = "x");

void sort_params(std::vector<instruction_ref>& params);

// Find the inputs for a module by finding instructions that are mapped to the
// parameters in the module
MIGRAPHX_EXPORT std::vector<instruction_ref>
find_inputs(const std::unordered_map<instruction_ref, instruction_ref>& map_ins,
            const_module_ref parent,
            const_module_ref sub);

MIGRAPHX_EXPORT std::vector<instruction_ref>
find_inputs(const std::unordered_map<instruction_ref, instruction_ref>& map_ins,
            const std::unordered_set<instruction_ref>& parent_instructions,
            const_module_ref sub);

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
#endif // MIGRAPHX_GUARD_MIGRAPHX_PARAM_UTILS_HPP
