#include <migraphx/dom_info.hpp>
#include <migraphx/program.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/erase.hpp>
#include <migraphx/ranges.hpp>
#include <unordered_set>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct program_visitor
{
    program* prog;
    program& get_nodes() { return *prog; }

    const std::vector<instruction_ref>& get_children(instruction_ref ins) { return ins->inputs(); }
};

template <class Visitor>
dominator_info compute_dominator_generic(Visitor v)
{
    dominator_info info;
    std::unordered_map<instruction_ref, std::unordered_set<instruction_ref>> instr2_doms;
    for(instruction_ref ins : iterator_for(v.get_nodes()))
    {
        const std::vector<instruction_ref>& children = v.get_children(ins);
        if(not children.empty())
        {
            auto arg         = children.front();
            instr2_doms[ins] = instr2_doms[arg];
            std::for_each(children.begin() + 1, children.end(), [&](instruction_ref child) {
                auto&& child_doms = instr2_doms[child];
                erase_if(instr2_doms[ins], [&](auto x) { return contains(child_doms, x); });
            });

            if(children.size() == 1)
            {
                info.ins2idom[ins] = arg;
            }
            else
            {
                for(auto x : instr2_doms[ins])
                {
                    if(!std::any_of(instr2_doms[ins].begin(), instr2_doms[ins].end(), [&](auto y) {
                           return info.strictly_dominate(x, y);
                       }))
                    {
                        assert(info.ins2idom.find(ins) == info.ins2idom.end());
                        info.ins2idom[ins] = x;
                    }
                }
            }
        }

        instr2_doms[ins].insert(ins);
    }
    return info;
}

dominator_info compute_dominator(program& p)
{
    return compute_dominator_generic(program_visitor{&p});
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
