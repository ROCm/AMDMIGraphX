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

#include <Windows.h>

#include "migraphx_kernels.hpp"
#include "resource.h"

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

namespace resource {
    std::string_view read(int id)
    {
        HMODULE handle{ ::GetModuleHandle(nullptr) };
        HRSRC rc{ ::FindResource(handle, MAKEINTRESOURCE(id), MAKEINTRESOURCE(TEXTFILE)) };
        HGLOBAL data{ ::LoadResource(handle, rc) };
        return { static_cast<char const *>(::LockResource(data)),
                ::SizeofResource(handle, rc) };
    }
}

std::vector<src_file> migraphx_kernels()
{
    static src_file _kernels_[] = {
        { "migraphx/kernels/algorithm.hpp", resource::read(IDR_ALGORITHM_HPP) },
        { "migraphx/kernels/args.hpp", resource::read(IDR_ARGS_HPP) },
        { "migraphx/kernels/array.hpp", resource::read(IDR_ARRAY_HPP) },
        { "migraphx/kernels/concat.hpp", resource::read(IDR_CONCAT_HPP) },
        { "migraphx/kernels/debug.hpp", resource::read(IDR_DEBUG_HPP) },
        { "migraphx/kernels/dfor.hpp", resource::read(IDR_DFOR_HPP) },
        { "migraphx/kernels/dpp.hpp", resource::read(IDR_DPP_HPP) },
        { "migraphx/kernels/functional.hpp", resource::read(IDR_FUNCTIONAL_HPP) },
        { "migraphx/kernels/gather.hpp", resource::read(IDR_GATHER_HPP) },
        { "migraphx/kernels/gathernd.hpp", resource::read(IDR_GATHERND_HPP) },
        { "migraphx/kernels/generic_constant.hpp", resource::read(IDR_GENERIC_CONSTANT_HPP) },
        { "migraphx/kernels/hip.hpp", resource::read(IDR_HIP_HPP) },
        { "migraphx/kernels/index.hpp", resource::read(IDR_INDEX_HPP) },
        { "migraphx/kernels/integral_constant.hpp", resource::read(IDR_INTEGRAL_CONSTANT_HPP) },
        { "migraphx/kernels/iota_iterator.hpp", resource::read(IDR_IOTA_ITERATOR_HPP) },
        { "migraphx/kernels/layernorm.hpp", resource::read(IDR_LAYERNORM_HPP) },
        { "migraphx/kernels/math.hpp", resource::read(IDR_MATH_HPP) },
        { "migraphx/kernels/ops.hpp", resource::read(IDR_OPS_HPP) },
        { "migraphx/kernels/pad.hpp", resource::read(IDR_PAD_HPP) },
        { "migraphx/kernels/pointwise.hpp", resource::read(IDR_POINTWISE_HPP) },
        { "migraphx/kernels/preload.hpp", resource::read(IDR_PRELOAD_HPP) },
        { "migraphx/kernels/print.hpp", resource::read(IDR_PRINT_HPP) },
        { "migraphx/kernels/ranges.hpp", resource::read(IDR_RANGES_HPP) },
        { "migraphx/kernels/reduce.hpp", resource::read(IDR_REDUCE_HPP) },
        { "migraphx/kernels/roialign.hpp", resource::read(IDR_ROIALIGN_HPP) },
        { "migraphx/kernels/scatternd.hpp", resource::read(IDR_SCATTERND_HPP) },
        { "migraphx/kernels/shape.hpp", resource::read(IDR_SHAPE_HPP) },
        { "migraphx/kernels/softmax.hpp", resource::read(IDR_SOFTMAX_HPP) },
        { "migraphx/kernels/tensor_view.hpp", resource::read(IDR_TENSOR_VIEW_HPP) },
        { "migraphx/kernels/type_traits.hpp", resource::read(IDR_TYPE_TRAITS_HPP) },
        { "migraphx/kernels/types.hpp", resource::read(IDR_TYPES_HPP) },
        { "migraphx/kernels/vec.hpp", resource::read(IDR_VEC_HPP) },
        { "migraphx/kernels/vectorize.hpp", resource::read(IDR_VECTORIZE_HPP) }
    };
    return { std::begin(_kernels_), std::end(_kernels_) };
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
