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

namespace {
std::string_view __read(int id)
{
    HMODULE handle{::GetModuleHandle(nullptr)};
    HRSRC rc{::FindResource(handle, MAKEINTRESOURCE(id), MAKEINTRESOURCE(MIGRAPHX_TEXTFILE))};
    HGLOBAL data{::LoadResource(handle, rc)};
    return {static_cast<char const*>(::LockResource(data)), ::SizeofResource(handle, rc)};
}
}

std::unordered_map<std::string_view, std::string_view> migraphx_kernels()
{
    static std::unordered_map<std::string_view, std::string_view> kernels = {
        {"migraphx/kernels/algorithm.hpp", __read(MIGRAPHX_IDR_ALGORITHM_HPP)},
        {"migraphx/kernels/args.hpp", __read(MIGRAPHX_IDR_ARGS_HPP)},
        {"migraphx/kernels/array.hpp", __read(MIGRAPHX_IDR_ARRAY_HPP)},
        {"migraphx/kernels/concat.hpp", __read(MIGRAPHX_IDR_CONCAT_HPP)},
        {"migraphx/kernels/debug.hpp", __read(MIGRAPHX_IDR_DEBUG_HPP)},
        {"migraphx/kernels/dfor.hpp", __read(MIGRAPHX_IDR_DFOR_HPP)},
        {"migraphx/kernels/dpp.hpp", __read(MIGRAPHX_IDR_DPP_HPP)},
        {"migraphx/kernels/functional.hpp", __read(MIGRAPHX_IDR_FUNCTIONAL_HPP)},
        {"migraphx/kernels/gather.hpp", __read(MIGRAPHX_IDR_GATHER_HPP)},
        {"migraphx/kernels/gathernd.hpp", __read(MIGRAPHX_IDR_GATHERND_HPP)},
        {"migraphx/kernels/generic_constant.hpp", __read(MIGRAPHX_IDR_GENERIC_CONSTANT_HPP)},
        {"migraphx/kernels/hip.hpp", __read(MIGRAPHX_IDR_HIP_HPP)},
        {"migraphx/kernels/index.hpp", __read(MIGRAPHX_IDR_INDEX_HPP)},
        {"migraphx/kernels/integral_constant.hpp", __read(MIGRAPHX_IDR_INTEGRAL_CONSTANT_HPP)},
        {"migraphx/kernels/iota_iterator.hpp", __read(MIGRAPHX_IDR_IOTA_ITERATOR_HPP)},
        {"migraphx/kernels/layernorm.hpp", __read(MIGRAPHX_IDR_LAYERNORM_HPP)},
        {"migraphx/kernels/math.hpp", __read(MIGRAPHX_IDR_MATH_HPP)},
        {"migraphx/kernels/ops.hpp", __read(MIGRAPHX_IDR_OPS_HPP)},
        {"migraphx/kernels/pad.hpp", __read(MIGRAPHX_IDR_PAD_HPP)},
        {"migraphx/kernels/pointwise.hpp", __read(MIGRAPHX_IDR_POINTWISE_HPP)},
        {"migraphx/kernels/preload.hpp", __read(MIGRAPHX_IDR_PRELOAD_HPP)},
        {"migraphx/kernels/print.hpp", __read(MIGRAPHX_IDR_PRINT_HPP)},
        {"migraphx/kernels/ranges.hpp", __read(MIGRAPHX_IDR_RANGES_HPP)},
        {"migraphx/kernels/reduce.hpp", __read(MIGRAPHX_IDR_REDUCE_HPP)},
        {"migraphx/kernels/roialign.hpp", __read(MIGRAPHX_IDR_ROIALIGN_HPP)},
        {"migraphx/kernels/scatternd.hpp", __read(MIGRAPHX_IDR_SCATTERND_HPP)},
        {"migraphx/kernels/shape.hpp", __read(MIGRAPHX_IDR_SHAPE_HPP)},
        {"migraphx/kernels/softmax.hpp", __read(MIGRAPHX_IDR_SOFTMAX_HPP)},
        {"migraphx/kernels/tensor_view.hpp", __read(MIGRAPHX_IDR_TENSOR_VIEW_HPP)},
        {"migraphx/kernels/type_traits.hpp", __read(MIGRAPHX_IDR_TYPE_TRAITS_HPP)},
        {"migraphx/kernels/types.hpp", __read(MIGRAPHX_IDR_TYPES_HPP)},
        {"migraphx/kernels/vec.hpp", __read(MIGRAPHX_IDR_VEC_HPP)},
        {"migraphx/kernels/vectorize.hpp", __read(MIGRAPHX_IDR_VECTORIZE_HPP)}};
    return kernels;
}
