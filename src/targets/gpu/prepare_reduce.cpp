#include <migraphx/gpu/prepare_reduce.hpp>
#include <migraphx/matcher.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/algorithm.hpp>
#include <migraphx/register_op.hpp>
#include <migraphx/make_op.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

struct parallel_reduce
{
    operation op;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.op, "op"));
    }

    std::string name() const { return "gpu::parallel_reduce"; }

    shape compute_shape(const std::vector<shape>& inputs) const
    {
        std::vector<shape> result;
        std::transform(inputs.begin(), inputs.end(), std::back_inserter(result), [&](auto input) {
            return op.compute_shape({input});
        });
        return {result};
    }
};
MIGRAPHX_REGISTER_OP(parallel_reduce);

namespace {

optional<instruction_ref> get_reduce(instruction_ref ins)
{
    if(ins->name() == "gpu::parallel_reduce")
        return nullopt;
    if(contains(ins->name(), "reduce"))
        return ins;
    if(ins->name() == "pointwise")
    {
        if(ins->inputs().size() == 1 and ins->outputs().size() == 1)
            return get_reduce(ins->outputs().front());
    }
    return nullopt;
}

MIGRAPHX_PRED_MATCHER(split_reduce, instruction_ref ins)
{
    if(ins->outputs().size() < 2)
        return false;
    auto n = std::count_if(ins->outputs().begin(),
                           ins->outputs().end(),
                           [](instruction_ref output) { return get_reduce(output).has_value(); });
    return n > 1;
}

struct find_multi_reduce
{
    auto matcher() const { return split_reduce(); }

    void apply(module& m, const match::matcher_result& r) const
    {
        auto ins = r.result;
        std::vector<instruction_ref> reduces;
        for(auto output : ins->outputs())
        {
            if(output->outputs().empty())
                continue;
            auto reduce = get_reduce(output);
            if(reduce.has_value())
                reduces.push_back(*reduce);
        }

        auto each = [&](auto start, auto last) {
            if(std::distance(start, last) < 2)
                return;
            std::vector<instruction_ref> inputs;
            std::transform(start, last, std::back_inserter(inputs), [&](auto reduce) {
                return reduce->inputs().front();
            });
            auto op        = (*start)->get_operator();
            auto insertion = std::next(ins);
            for(auto input : inputs)
            {
                if(input == ins)
                    continue;
                m.move_instruction(input, insertion);
            }
            auto preduce = m.insert_instruction(insertion, parallel_reduce{op}, inputs);
            int i        = 0;
            std::for_each(start, last, [&](auto reduce) {
                m.replace_instruction(reduce, make_op("get_tuple_elem", {{"index", i}}), preduce);
                i++;
            });
        };

        group_by(reduces.begin(), reduces.end(), each, by(std::equal_to<>{}, [](instruction_ref i) {
                     return std::make_tuple(i->name(), i->get_shape());
                 }));
    }
};

} // namespace

void prepare_reduce::apply(module& m) const { match::find_matches(m, find_multi_reduce{}); }

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
