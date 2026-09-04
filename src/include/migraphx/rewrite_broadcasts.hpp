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
#ifndef MIGRAPHX_GUARD_MIGRAPHX_REWRITE_BROADCASTS_HPP
#define MIGRAPHX_GUARD_MIGRAPHX_REWRITE_BROADCASTS_HPP

#include <migraphx/config.hpp>
#include <migraphx/instruction_ref.hpp>
#include <cstddef>
#include <string>
#include <vector>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct module_pass_manager;
struct shape;

// True if the broadcast output shape only expands the given reduce axes, which
// allows the broadcast to be fused into the reduction
MIGRAPHX_EXPORT bool is_valid_broadcast(const shape& s, std::vector<std::size_t> reduce_axes);

// True if an input other than the broadcast spans the reduce axes. Fusing a
// broadcast into a reduction requires such an input since the codegen cannot
// reduce purely broadcasted data.
MIGRAPHX_EXPORT bool has_spanning_input(const std::vector<instruction_ref>& inputs,
                                        instruction_ref broadcast,
                                        const std::vector<std::size_t>& reduce_axes);

// Move a broadcast or multibroadcast between a pointwise producer and a consumer
// of the given op onto the pointwise inputs so the two can be fused.
MIGRAPHX_EXPORT void rewrite_broadcasts(module_pass_manager& mpm, const std::string& op);

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
#endif // MIGRAPHX_GUARD_MIGRAPHX_REWRITE_BROADCASTS_HPP
