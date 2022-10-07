
#ifndef MIGRAPHX_GUARD_GEMM_INSTANCE_HPP
#define MIGRAPHX_GUARD_GEMM_INSTANCE_HPP
#include <migraphx/kernels/ck_gemm_includes.hpp>

namespace migraphx {

    using gemm = 

       CKDeviceGemm< Row, Row, Row, F16, F16, F16, F32, F16, PassThrough, PassThrough, PassThrough, GemmDefault, 1, 256, 128, 128, 32, 8, 8, 32, 32, 2, 2, S<4, 64, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 1, S<4, 64, 1>, S<0, 2, 1>, S<0, 2, 1>, 1, 2, 8, 1, 1, 1, S<1, 32, 1, 8>, 8>;

} // namespace migraphx
#endif
