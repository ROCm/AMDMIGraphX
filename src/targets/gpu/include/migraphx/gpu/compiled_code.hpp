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
#ifndef MIGRAPHX_GUARD_GPU_COMPILED_CODE_HPP
#define MIGRAPHX_GUARD_GPU_COMPILED_CODE_HPP

#include <migraphx/gpu/config.hpp>
#include <migraphx/program.hpp>
#include <migraphx/module.hpp>
#include <migraphx/instruction_ref.hpp>
#include <migraphx/reflect.hpp>
#include <string>
#include <unordered_map>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

/**
 * The result of compiling one precompile_op, in a form that can be serialized and reused.
 *
 * The fragment holds the compiled code objects along with any prefills and allocations they
 * need. Its parameters correspond by position to the inputs of the instruction it replaces, so
 * the fragment does not refer to the instruction that produced it and can be reused for every
 * instruction that compiles to the same code. It is kept as a program because that is what
 * already converts to and from a value.
 */
struct MIGRAPHX_GPU_EXPORT compiled_code
{
    program fragment = {};
    /// Values used to prefill inputs while benchmarking, keyed by type and size string.
    std::unordered_map<std::string, double> fill_map = {};

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.fragment, "fragment"), f(self.fill_map, "fill_map"));
    }

    bool empty() const { return fragment.get_main_module()->size() == 0; }

    /// Splice the fragment into m in place of ins, mapping each fragment parameter to the
    /// input of ins in the same position.
    void replace(module& m, instruction_ref ins) const;

    /// Name of the fragment parameter standing in for the input at index i.
    static std::string input_name(std::size_t i);
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif // MIGRAPHX_GUARD_GPU_COMPILED_CODE_HPP
