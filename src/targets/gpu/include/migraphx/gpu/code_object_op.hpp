#ifndef MIGRAPHX_GUARD_AMDMIGRAPHX_CODE_OBJECT_OP_HPP
#define MIGRAPHX_GUARD_AMDMIGRAPHX_CODE_OBJECT_OP_HPP

#include <migraphx/config.hpp>
#include <migraphx/value.hpp>
#include <migraphx/argument.hpp>
#include <migraphx/functional.hpp>
#include <migraphx/gpu/kernel.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

struct context;

struct code_object_op
{
    value::binary code_object;
    std::string symbol_name;
    std::size_t global;
    std::size_t local;
    std::vector<shape> expected_inputs;
    shape output;
    kernel k{};

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.code_object, "code_object"),
                    f(self.symbol_name, "symbol_name"),
                    f(self.global, "global"),
                    f(self.local, "local"),
                    f(self.expected_inputs, "expected_inputs"),
                    f(self.output, "output"));
    }

    std::string name() const { return "gpu::code_object"; }
    shape compute_shape(std::vector<shape> inputs) const;
    argument
    compute(context& ctx, const shape& output_shape, const std::vector<argument>& args) const;
    void finalize(context&, const shape&, const std::vector<shape>&);
    std::ptrdiff_t output_alias(const std::vector<shape>& shapes) const
    {
        return shapes.size() - 1;
    }

    friend std::ostream& operator<<(std::ostream& os, const code_object_op& op)
    {
        os << op.name() << "[";
        os << "code_object=" << op.code_object.size() << ",";
        os << "symbol_name=" << op.symbol_name << ",";
        os << "global=" << op.global << ",";
        os << "local=" << op.local << ",";
        return os;
    }
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
