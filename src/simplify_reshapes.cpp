#include <migraphx/simplify_reshapes.hpp>
#include <migraphx/program.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/op/as_shape.hpp>
#include <migraphx/op/transpose.hpp>
#include <migraphx/op/concat.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/matcher.hpp>
#include <migraphx/permutation.hpp>
#include <unordered_set>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

const auto& reshaper_names()
{
    // clang-format off
    static const std::unordered_set<std::string> names = {
        "reshape",
        "contiguous",
        "squeeze",
        "unsqueeze"
    };
    // clang-format on
    return names;
}

bool is_reshaper(instruction_ref ins) { return contains(reshaper_names(), ins->name()); }

instruction_ref find_transpose_input(instruction_ref ins)
{
    if(ins->inputs().size() != 1)
        return ins;
    if(ins->inputs().front()->name() == "contiguous")
        return find_transpose_input(ins->inputs().front());
    if(ins->inputs().front()->name() == "transpose")
        return ins->inputs().front();
    return ins;
}

auto get_transpose_dims(instruction_ref ins)
{
    return any_cast<const op::transpose&>(ins->get_operator()).dims;
}

bool is_no_transpose(const std::vector<int64_t>& dims)
{
    if(dims.empty())
        return true;
    if(dims.front() != 0)
        return false;
    return std::adjacent_find(
               dims.begin(), dims.end(), [](auto x, auto y) { return (y - x) != 1; }) == dims.end();
}

struct find_reshaper
{
    auto matcher() const
    {
        return match::name(reshaper_names())(
            match::any_of[match::outputs()](match::name(reshaper_names())));
    }

    void apply(program& p, const match::matcher_result& mr) const
    {
        auto ins = mr.result;
        std::vector<instruction_ref> reshapes{ins};
        while(is_reshaper(reshapes.back()))
        {
            assert(!reshapes.back()->inputs().empty());
            assert(p.has_instruction(reshapes.back()->inputs().front()));
            auto input = reshapes.back()->inputs().front();
            reshapes.push_back(input);
        }

        std::pair<instruction_ref, instruction_ref> r{p.end(), p.end()};
        for(auto start : iterator_for(reshapes))
        {
            auto last = std::find_if(reshapes.rbegin(), reshapes.rend(), [&](auto&& i) {
                return i->get_shape() == (*start)->get_shape() and i != (*start);
            });
            if(last != reshapes.rend())
            {
                r = std::make_pair(*start, *last);
                break;
            }
        }
        if(r.first != r.second)
        {
            p.replace_instruction(r.first, r.second);
        }
    }
};

struct find_nop_reshapes
{
    auto matcher() const
    {
        auto reshapes = reshaper_names();
        reshapes.insert("transpose");
        reshapes.insert("slice");
        return match::name(reshapes)(match::same_shape(match::arg(0)));
    }

    void apply(program& p, const match::matcher_result& mr) const
    {
        auto ins = mr.result;
        p.replace_instruction(ins, ins->inputs().front());
    }
};

struct find_transpose
{
    auto matcher() const
    {
        return match::name("transpose")(match::none_of(
            match::skip_output(match::name("contiguous"))(match::name("transpose"))));
    }

    void apply(program& p, const match::matcher_result& mr) const
    {
        auto ins = mr.result;
        auto x   = ins;
        auto t   = ins;
        std::vector<std::int64_t> dims(ins->get_shape().lens().size());
        std::iota(dims.begin(), dims.end(), 0);
        do
        {
            dims = reorder_dims(get_transpose_dims(t), dims);
            x    = t;
            t    = find_transpose_input(x);
        } while(x != t and t->name() == "transpose");
        if(t == ins or t->name() != "transpose")
            return;
        if(is_no_transpose(dims))
        {
            p.replace_instruction(ins, t->inputs().front());
        }
        else
        {
            p.replace_instruction(ins, op::transpose{{dims}}, t->inputs().front());
        }
    }
};

struct find_concat_transpose
{
    auto matcher() const
    {
        return match::name("concat")(match::all_of[match::inputs()](match::transpose_shape()));
    }

    void apply(program& p, const match::matcher_result& mr) const
    {
        auto ins = mr.result;
        auto s   = ins->inputs().front()->get_shape();
        assert(s.transposed());
        auto op           = any_cast<op::concat>(ins->get_operator());
        auto permutation  = find_permutation(s);
        auto ipermutation = invert_permutation(permutation);
        op.axis           = ipermutation[op.axis];

        std::vector<instruction_ref> inputs;
        std::transform(
            ins->inputs().begin(), ins->inputs().end(), std::back_inserter(inputs), [&](auto i) {
                return p.insert_instruction(ins, op::transpose{permutation}, i);
            });
        auto concat = p.insert_instruction(ins, op, inputs);
        auto t      = p.insert_instruction(ins, op::transpose{ipermutation}, concat);
        assert(ins->get_shape().lens() == t->get_shape().lens());
        p.replace_instruction(ins, t);
    }
};

struct find_nested_concat
{
    auto matcher() const
    {
        return match::name("concat")(match::any_of[match::inputs()](match::name("concat")));
    }

    static std::size_t get_axis(instruction_ref ins)
    {
        auto op = any_cast<op::concat>(ins->get_operator());
        return op.axis;
    }

    void apply(program& p, const match::matcher_result& mr) const
    {
        auto ins  = mr.result;
        auto axis = get_axis(ins);
        std::vector<instruction_ref> args;
        fix([&](auto self, auto&& inputs) {
            for(auto&& i : inputs)
            {
                if(i->name() == "concat" and get_axis(i) == axis and i->outputs().size() == 1)
                    self(i->inputs());
                else
                    args.push_back(i);
            }

        })(ins->inputs());
        p.replace_instruction(ins, ins->get_operator(), args);
    }
};

void simplify_reshapes::apply(program& p) const
{
    for(int i = 0; i < 2; i++)
    {
        auto end = std::prev(p.end());
        for(auto ins : iterator_for(p))
        {
            if(ins == end and ins->name() == "contiguous")
                continue;
            // Skip possible dead instructions
            if(ins->outputs().empty() and ins != end)
                continue;
            match::find_matches(p,
                                ins,
                                find_nop_reshapes{},
                                find_reshaper{},
                                find_transpose{},
                                find_concat_transpose{},
                                find_nested_concat{});
        }
    }
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
