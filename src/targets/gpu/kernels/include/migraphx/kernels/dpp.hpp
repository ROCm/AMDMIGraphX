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
#endif

} // namespace migraphx
#endif // MIGRAPHX_GUARD_KERNELS_DPP_HPP
