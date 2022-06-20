
#include <migraphx/program.hpp>

namespace migraphx {
namespace driver {
inline namespace MIGRAPHX_INLINE_NS {

migraphx::program resnet50(unsigned batch);
migraphx::program inceptionv3(unsigned batch);
migraphx::program alexnet(unsigned batch);

} // namespace MIGRAPHX_INLINE_NS
} // namespace driver
} // namespace migraphx
