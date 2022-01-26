#pragma once

#include <migraphx/program.hpp>
#include <migraphx/config.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace fpga {

struct lowering
{
    std::string name() const { return "fpga::lowering"; }
    void apply(module& m) const;
};

} // namespace fpga
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
