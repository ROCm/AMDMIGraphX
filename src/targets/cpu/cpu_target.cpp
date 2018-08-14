
#include <migraph/cpu/cpu_target.hpp>
#include <migraph/cpu/cpu_lowering.hpp>

namespace migraph {
namespace cpu {

std::string cpu_target::name() const { return "cpu"; }

std::vector<pass> cpu_target::get_passes(context&) const { return {cpu_lowering{}}; }

} // namespace cpu

} // namespace migraph
