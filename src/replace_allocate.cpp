#include <migraphx/replace_allocate.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/program.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/op/allocate.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

void replace_allocate::apply(module& p) const
{
    for(auto ins : iterator_for(p))
    {
        if(ins->get_operator().name() != "allocate")
            continue;
        auto op = ins->get_operator();
        auto v  = op.to_value();
        assert(v.contains("tag"));
        auto alloc_ins = p.insert_instruction(
            ins, make_op(model.name(), {{"shape", to_value(ins->get_shape())}, v.at("tag")}));
        p.replace_instruction(ins, alloc_ins);
    }
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
