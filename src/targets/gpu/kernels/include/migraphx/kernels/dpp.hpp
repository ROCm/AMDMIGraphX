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
#ifndef MIGRAPHX_GUARD_KERNELS_DPP_HPP
#define MIGRAPHX_GUARD_KERNELS_DPP_HPP

#include <migraphx/kernels/hip.hpp>
#include <migraphx/kernels/types.hpp>
#include <migraphx/kernels/debug.hpp>

namespace migraphx {

#ifndef MIGRAPHX_HAS_DPP
#define MIGRAPHX_HAS_DPP 1
#endif

#if MIGRAPHX_HAS_DPP
constexpr unsigned int dpp_row_shr(unsigned int x) { return 0x110u | x; }

constexpr unsigned int dpp_row_bcast(unsigned int x)
{
    unsigned int y = 0;
    switch(x)
    {
    case 15: y = 0x142; break;
    case 31: y = 0x143; break;
    default: MIGRAPHX_UNREACHABLE();
    }
    return y;
}

template <unsigned int DppCtrl,
          unsigned int RowMask  = 0xf,
          unsigned int BankMask = 0xf,
          bool BoundCtrl        = false,
          class T>
__device__ T dpp_mov(T& x)
{
    static const index_int n = sizeof(T) < 4 ? 1 : sizeof(T) / 4;
    union type
    {
        uint32_t reg[n];
        T data;
    };
    type output{};
    type input{};
    // cppcheck-suppress unreadVariable
    input.data = x;
    for(index_int i = 0; i < n; i++)
    {
#if defined(__HCC__)
        output.reg[i] = __llvm_amdgcn_move_dpp(input.reg[i], DppCtrl, RowMask, BankMask, BoundCtrl);
#else
        output.reg[i] = __hip_move_dpp(input.reg[i], DppCtrl, RowMask, BankMask, BoundCtrl);
#endif
    }
    return output.data;
}
#endif // MIGRAPHX_HAS_DPP

} // namespace migraphx
#endif // MIGRAPHX_GUARD_KERNELS_DPP_HPP
