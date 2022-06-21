#include <migraphx/cpu/write_literals.hpp>
#include <migraphx/module.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/iterator_for.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace cpu {

struct cpu_literal
{
    argument data;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.data, "data"));
    }

    std::string name() const { return "cpu::literal"; }

    shape compute_shape(const std::vector<shape>&) const { return data.get_shape(); }

    argument compute(const shape&, const std::vector<argument>&) const { return data; }

    friend std::ostream& operator<<(std::ostream& os, const cpu_literal& x)
    {
        os << x.name();
        return os;
    }
};

void write_literals::apply(module& m) const
{
    for(auto ins : iterator_for(m))
    {
        if(ins->name() != "@literal")
            continue;
        m.replace_instruction(ins, cpu_literal{ins->get_literal().get_argument()});
    }
}

} // namespace cpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
