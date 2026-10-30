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
#ifndef MIGRAPHX_GUARD_ONNX_RESHAPE_HELPERS_HPP
#define MIGRAPHX_GUARD_ONNX_RESHAPE_HELPERS_HPP

#include <migraphx/onnx/onnx_parser.hpp>
#include <migraphx/dim_like.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/runtime_shape.hpp>

#include <algorithm>
#include <optional>
#include <vector>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {

template <class T>
std::optional<instruction_ref>
try_add_runtime_reshape(const onnx_parser::node_info& info,
                        instruction_ref input,
                        const std::vector<T>& reshape_dims,
                        const std::vector<instruction_ref>& shape_sources)
{
    return migraphx::try_add_runtime_reshape(*info.mod, input, reshape_dims, shape_sources);
}

inline instruction_ref
add_symbolic_reshape(const onnx_parser::node_info& info,
                     instruction_ref input,
                     const std::vector<shape::dynamic_dimension>& dims,
                     const std::vector<instruction_ref>& extra_shape_sources = {})
{
    std::vector<instruction_ref> shape_sources = info.mod->get_parameters();
    auto append_source                         = [&](instruction_ref source) {
        if(std::find(shape_sources.begin(), shape_sources.end(), source) == shape_sources.end())
            shape_sources.push_back(source);
    };
    append_source(input);
    for(auto source : extra_shape_sources)
        append_source(source);

    auto result = migraphx::try_add_runtime_reshape(*info.mod, input, dims, shape_sources);
    if(not result.has_value())
        MIGRAPHX_THROW("Reshape: symbolic dimensions require direct runtime shape sources");
    return *result;
}

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
