#include <iterator>
#include <migraphx/simplify_reshapes.hpp>
#include <migraphx/program.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/op/as_shape.hpp>
#include <migraphx/op/transpose.hpp>
#include <migraphx/op/concat.hpp>
#include <migraphx/op/slice.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/matcher.hpp>
#include <migraphx/permutation.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <unordered_set>
#include <migraphx/make_op.hpp>
#include <migraphx/tune_axis.hpp>

#include <map>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

const auto& reshaper_names()
{
    // clang-format off
    static const std::unordered_set<std::string> names = {
        "flatten",
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

    void apply(module& p, const match::matcher_result& mr) const
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
        reshapes.insert("as_shape");
        reshapes.insert("broadcast");
        reshapes.insert("concat");
        reshapes.insert("convert");
        reshapes.insert("multibroadcast");
        reshapes.insert("pad");
        reshapes.insert("slice");
        reshapes.insert("transpose");
        return match::name(reshapes)(match::same_shape(match::arg(0)));
    }

    void apply(module& p, const match::matcher_result& mr) const
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

    void apply(module& p, const match::matcher_result& mr) const
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
            p.replace_instruction(
                ins, make_op("transpose", {{"permutation", dims}}), t->inputs().front());
        }
    }
};

struct find_nested_convert
{
    auto matcher() const { return match::name("convert")(match::arg(0)(match::name("convert"))); }

    void apply(module& m, const match::matcher_result& mr) const
    {
        auto ins   = mr.result;
        auto x     = ins->inputs().front();
        auto input = x->inputs().front();

        if(ins->get_shape() != input->get_shape())
            return;

        m.replace_instruction(ins, input);
    }
};

struct find_nested_slice
{
    auto matcher() const { return match::name("slice")(match::arg(0)(match::name("slice"))); }

    using axes_map = std::map<std::size_t, std::pair<std::size_t, std::size_t>>;

    static axes_map get_axes(instruction_ref ins)
    {
        axes_map result;
        auto op = any_cast<op::slice>(ins->get_operator());
        for(std::size_t i = 0; i < op.axes.size(); i++)
        {
            result[op.axes[i]] = std::make_pair(op.starts[i], op.ends[i]);
        }
        return result;
    }

    static axes_map merge(const axes_map& m1, const axes_map& m2)
    {
        axes_map result;
        // Non overlapping
        for(auto&& p : m1)
        {
            if(contains(m2, p.first))
                continue;
            result[p.first] = p.second;
        }
        for(auto&& p : m2)
        {
            if(contains(m1, p.first))
                continue;
            result[p.first] = p.second;
        }
        // Overlapping
        for(auto&& p1 : m1)
        {
            if(not contains(m2, p1.first))
                continue;
            auto&& v1        = p1.second;
            auto&& v2        = m2.at(p1.first);
            auto start       = v1.first + v2.first;
            auto end         = start + (v2.second - v2.first);
            result[p1.first] = std::make_pair(start, end);
        }
        return result;
    }

    void apply(module& p, const match::matcher_result& mr) const
    {
        auto ins   = mr.result;
        auto slice = ins->inputs().front();
        auto input = slice->inputs().front();

        auto a1 = get_axes(ins);
        auto a2 = get_axes(slice);

        auto axes = merge(a2, a1);

        auto op = op::slice{};
        for(auto&& pp : axes)
        {
            op.axes.push_back(pp.first);
            op.starts.push_back(pp.second.first);
            op.ends.push_back(pp.second.second);
        }
        p.replace_instruction(ins, op, input);
    }
};

struct find_concat_transpose
{
    auto matcher() const
    {
        return match::name("concat")(match::all_of[match::inputs()](match::transpose_shape()));
    }

    void apply(module& p, const match::matcher_result& mr) const
    {
        auto ins          = mr.result;
        auto trans_inputs = ins->inputs();
        auto s            = trans_inputs.front()->get_shape();
        assert(s.transposed());
        auto op          = any_cast<op::concat>(ins->get_operator());
        auto permutation = find_permutation(s);

        // permutation should be the same for all inputs
        if(!std::all_of(trans_inputs.begin(), trans_inputs.end(), [&](auto in) {
               return (find_permutation(in->get_shape()) == permutation);
           }))
        {
            return;
        }

        // axis could be a negative value
        int64_t n_dim = static_cast<int64_t>(s.lens().size());
        op.axis       = tune_axis(n_dim, op.axis, op.name());

        auto ipermutation = invert_permutation(permutation);
        op.axis           = ipermutation[op.axis];

        std::vector<instruction_ref> inputs;
        std::transform(
            ins->inputs().begin(), ins->inputs().end(), std::back_inserter(inputs), [&](auto i) {
                return p.insert_instruction(
                    ins, make_op("transpose", {{"permutation", permutation}}), i);
            });
        auto concat = p.insert_instruction(ins, op, inputs);
        auto t      = p.insert_instruction(
            ins, make_op("transpose", {{"permutation", ipermutation}}), concat);
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

    void apply(module& p, const match::matcher_result& mr) const
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

struct find_resize
{
    auto matcher() const
    {
        return match::name("gather")(
            match::args(match::name("reshape").bind("data"), match::is_constant().bind("ind")));
    }

    void apply(module& p, match::matcher_result r) const
    {
        auto ins     = r.result;
        auto ins_rsp = r.instructions["data"];
        auto ins_ind = r.instructions["ind"];

        // resize input shape
        if(ins_rsp->get_shape().lens().size() != 1)
        {
            return;
        }

        // resize output shape
        const auto& in_shape  = ins_rsp->inputs().front()->get_shape();
        const auto& out_shape = ins->get_shape();
        // check if output shape is multiple of input shape
        const auto& in_lens  = in_shape.lens();
        const auto& out_lens = out_shape.lens();
        if(in_lens.size() != out_lens.size())
        {
            return;
        }

        // output shape must be multiple of input shape
        std::vector<bool> is_multi(in_lens.size());
        std::transform(
            in_lens.begin(), in_lens.end(), out_lens.begin(), is_multi.begin(), [](auto x, auto y) {
                return (y % x == 0);
            });
        if(not std::all_of(is_multi.begin(), is_multi.end(), [](auto b) { return b; }))
        {
            return;
        }

        // output must be multiple of inputs
        std::vector<std::size_t> scales(in_lens.size());
        std::transform(
            in_lens.begin(), in_lens.end(), out_lens.begin(), scales.begin(), [](auto x, auto y) {
                return y / x;
            });

        // if ind is not constant, cannot optimize
        std::vector<int> vec_ind;
        auto arg_ind = ins_ind->eval();
        if(arg_ind.empty())
        {
            return;
        }
        arg_ind.visit([&](auto v) { vec_ind.assign(v.begin(), v.end()); });
        if(not all_of(range(out_shape.elements()), [&](auto i) {
               auto out_idx = out_shape.multi(i);
               auto in_idx  = out_idx;
               std::transform(out_idx.begin(),
                              out_idx.end(),
                              scales.begin(),
                              in_idx.begin(),
                              [&](auto io, auto scale) { return io - (io % scale); });
               return vec_ind[i] == vec_ind[out_shape.index(in_idx)];
           }))
        {
            return;
        }

        // wrap up shapes for multibroadcast
        std::vector<std::pair<std::size_t, std::size_t>> dim_scales;
        std::transform(in_lens.begin(),
                       in_lens.end(),
                       out_lens.begin(),
                       std::back_inserter(dim_scales),
                       [](auto x, auto y) { return std::make_pair(x, y / x); });

        std::vector<int64_t> in_dims;
        std::vector<int64_t> out_dims;
        for(auto& isp : dim_scales)
        {
            in_dims.push_back(isp.first);
            out_dims.push_back(isp.first * isp.second);
            if(isp.first == 1 or isp.second == 1)
            {
                continue;
            }

            out_dims.back() = isp.first;
            in_dims.push_back(1);
            out_dims.push_back(isp.second);
        }

        auto in_rsp   = ins_rsp->inputs().front();
        auto rsp_data = p.insert_instruction(
            ins_rsp, migraphx::make_op("reshape", {{"dims", in_dims}}), in_rsp);
        auto mb_rsp = p.insert_instruction(
            ins_rsp, migraphx::make_op("multibroadcast", {{"out_lens", out_dims}}), rsp_data);
        auto std_mb = p.insert_instruction(ins, migraphx::make_op("contiguous"), mb_rsp);
        std::vector<int64_t> rsp_dims(out_lens.begin(), out_lens.end());
        p.replace_instruction(ins, migraphx::make_op("reshape", {{"dims", rsp_dims}}), std_mb);
    }
};

struct find_where_op
{
    auto matcher() const
    {
        return match::name("gather")(
            match::args(match::name("reshape")(match::arg(0)(match::name("concat").bind("data"))),
                        match::is_constant().bind("ind")));
    }

    void apply(module& p, match::matcher_result r) const
    {
        auto ins     = r.result;
        auto concat  = r.instructions["data"];
        auto ins_ind = r.instructions["ind"];
        std::vector<bool> vec_ind;
        auto arg_ind = ins_ind->eval();
        arg_ind.visit([&](auto v) { vec_ind.assign(v.begin(), v.end()); });
        // ind has to be the same value
        auto val = vec_ind.front();
        if(not std::all_of(vec_ind.begin(), vec_ind.end(), [&](auto v) { return (v == val); }))
        {
            return;
        }

        // concat axis must be 0
        auto op = any_cast<op::concat>(concat->get_operator());
        if(op.axis != 0)
        {
            return;
        }

        // check concat inputs, it has to be 2 and have the same shape
        const auto& inputs = concat->inputs();
        if(inputs.size() != 2)
        {
            return;
        }
        if(inputs.at(0)->get_shape() != inputs.at(1)->get_shape())
        {
            return;
        }
        if(inputs.at(0)->get_shape().lens() != ins_ind->get_shape().lens())
        {
            return;
        }

        if(val)
        {
            p.replace_instruction(ins, inputs.at(0));
        }
        else
        {
            p.replace_instruction(ins, inputs.at(1));
        }
    }
};

struct find_reshape_cont
{
    auto matcher() const
    {
        return match::pointwise(
            match::nargs(2),
            match::either_arg(0, 1)(
                match::name("reshape")(match::args(match::name("contiguous").bind("cont")))
                    .bind("rsp"),
                match::any()));
    }

    void apply(module& p, match::matcher_result r) const
    {
        auto ins      = r.result;
        auto ins_cont = r.instructions["cont"];
        auto in_ins   = r.instructions["rsp"];

        auto cont_input = ins_cont->inputs().front();
        auto lens       = cont_input->get_shape().lens();
        std::vector<int64_t> dims(lens.begin(), lens.end());

        if(in_ins->get_shape() != ins->get_shape())
        {
            return;
        }

        if(not std::all_of(ins->inputs().begin(), ins->inputs().end(), [](auto i) {
               return i->get_shape().standard();
           }))
        {
            return;
        }

        auto out_lens = ins->get_shape().lens();
        std::vector<int64_t> out_dims(out_lens.begin(), out_lens.end());
        std::vector<instruction_ref> inputs;
        for(const auto& in : ins->inputs())
        {
            if(in == in_ins)
            {
                inputs.push_back(cont_input);
            }
            else
            {
                inputs.push_back(
                    p.insert_instruction(ins, make_op("reshape", {{"dims", dims}}), in));
            }
        }
        auto out = p.insert_instruction(ins, ins->get_operator(), inputs);
        p.replace_instruction(ins, make_op("reshape", {{"dims", out_dims}}), out);
    }
};

// match sequence of transpose --> contiguous --> reshaper_op
auto match_transpose_contiguous_reshaper()
{
    return match::name({"reshape", "squeeze", "unsqueeze"})(
               match::used_once(),
               match::args(
                   match::name("contiguous")(
                       match::used_once(), match::args(match::transpose_shape().bind("trans_ins")))
                       .bind("cont_ins")))
        .bind("reshaper_ins");
};

// finds the pattern of transpose --> contiguous --> reshaper_op --> unary
// application of this matcher moves the unary operation before the contiguous so it becomes
// transpose --> unary --> contiguous --> reshaper_op. later pointwise sub-module can be created out
// of unary --> contiguous --> reshaper_op. Such pattern appears in depthToSpace or spaceToDepth
// operator.
struct find_transpose_contiguous_reshaper_unary
{
    auto matcher() const
    {
        return pointwise(match::used_once(),
                         match::nargs(1),
                         match::args(match_transpose_contiguous_reshaper()));
    }

    void apply(module& p, match::matcher_result r) const
    {
        auto ins           = r.result;
        auto reshaper_ins  = r.instructions["reshaper_ins"];
        auto trans_ins     = r.instructions["trans_ins"];
        auto cont_ins      = r.instructions["cont_ins"];
        auto unary_op_name = ins->get_operator().name();
        auto unary_ins     = p.insert_instruction(cont_ins, make_op(unary_op_name), trans_ins);
        auto new_cont_ins  = p.insert_instruction(cont_ins, make_op("contiguous"), unary_ins);
        // older cont and reshape are removed by deadcode elimination
        p.replace_instruction(ins, reshaper_ins->get_operator(), new_cont_ins);
    }
};

void simplify_reshapes::apply(module& p) const
{
    for(int i = 0; i < 2; i++)
    {
        match::find_matches(p,
                            find_where_op{},
                            find_resize{},
                            find_reshape_cont{},
                            find_nop_reshapes{},
                            find_reshaper{},
                            find_transpose{},
                            find_concat_transpose{},
                            find_nested_convert{},
                            find_nested_slice{},
                            find_nested_concat{},
                            find_transpose_contiguous_reshaper_unary{});
        dead_code_elimination{}.apply(p);
    }
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
