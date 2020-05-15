#include <migraphx/shape.hpp>
#include <migraphx/argument.hpp>
#include <migraphx/dfor.hpp>
#include <migraphx/gpu/device/add_transpose.hpp>
#include <migraphx/gpu/device/reduce.hpp>
#include <migraphx/gpu/device/tensor.hpp>
#include <migraphx/gpu/device/launch.hpp>
#include <migraphx/gpu/device/types.hpp>
#include <migraphx/gpu/device/srtc.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace device {

// the operator performed in this kernel is:
// shape of arg is {1, 128, 2304}
// slice {1536, 2304) of arg to generate shape of {1, 128, 768}
// reshape to shape of ({batch_size, 128, 12, 64}, sum_arg)
// transpose to shape of ([0, 2, 1, 3], rs_arg)
void add_transpose_arg0(hipStream_t stream,
                        const argument& result,
                        const argument& arg,
                        int slice_start)
{
    slice_reshape_transpose<0, 2, 1, 3>(stream, result, arg, slice_start);
}

// the operator performed in this kernel is:
// shape of arg is {1, 128, 2304}
// slice {768, 1536) of arg to generate shape of {1, 128, 768}
// reshape to shape of ({batch_size, 128, 12, 64}, sum_arg)
// transpose to shape of ([0, 2, 3, 1], rs_arg)
void add_transpose_arg1(hipStream_t stream,
                        const argument& result,
                        const argument& arg,
                        int slice_start)
{
    slice_reshape_transpose<0, 2, 3, 1>(stream, result, arg, slice_start);
}

} // namespace device
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
