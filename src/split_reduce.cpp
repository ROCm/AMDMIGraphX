#include <migraphx/split_reduce.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/module.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/liveness.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/check_shapes.hpp>
#include <migraphx/matcher.hpp>
#include <migraphx/register_op.hpp>
#include <migraphx/functional.hpp>
#include <migraphx/algorithm.hpp>
#include <migraphx/param_utils.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct split_fused_reduce
{
    std::vector<std::int64_t> axes{};
    std::string assign = "assign_none";

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.axes, "axes"), f(self.assign, "assign"));
    }

    shape compute_shape(const std::vector<shape>& inputs, std::vector<module_ref> mods) const
    {
        if(mods.size() != 1)
            MIGRAPHX_THROW("should have one submodule.");
        const auto* sm = mods.front();
        if(sm->get_output_shapes().size() != 1)
            MIGRAPHX_THROW("Only one output supported");
        auto names = sm->get_parameter_names();
        check_shapes{inputs, *this}.has(names.size()).same_ndims();
        std::sort(names.begin(), names.end());
        auto shapes = sm->get_parameter_shapes();
        // Check dimension matches for each input
        if(not equal(names, inputs, [&](const auto& name, const auto& input) {
               return shapes.at(name).lens() == input.lens();
           }))
            MIGRAPHX_THROW("Dimenstion does not match the submodule.");
        const auto& s = inputs.at(0);
        auto lens     = s.lens();
        if(lens != sm->get_output_shapes().front().lens())
        {
            for(const auto& axis : axes)
            {
                lens[axis] = 1;
            }
        }

        return shape::from_permutation(
            sm->get_output_shapes().front().type(), lens, find_permutation(inputs));
    }

    std::string name() const { return "split_fused_reduce"; }
};
MIGRAPHX_REGISTER_OP(split_fused_reduce);

static bool is_reduce(const instruction& ins) { return contains(ins.name(), "reduce"); }

static std::vector<instruction_ref> find_split(module_ref rm)
{
    std::vector<instruction_ref> result;
    auto reduce_ins = std::find_if(rm->begin(), rm->end(), &is_reduce);
    if(reduce_ins == rm->end())
        return result;
    // Bail if there is more than one reduce for now
    if(std::any_of(std::next(reduce_ins), rm->end(), &is_reduce))
        return result;
    result.push_back(reduce_ins);
    // TODO: Find instructions that are used again in the module
    return result;
}

static std::vector<instruction_ref> get_alive(module_ref rm,
                                              const std::vector<instruction_ref>& splits)
{
    std::vector<instruction_ref> result;
    bool stop = false;
    liveness(*rm, [&](auto ins, const auto& live_set) {
        if(stop)
            return;
        if(not contains(splits, ins))
            return;
        std::copy_if(live_set.begin(),
                     live_set.end(),
                     std::back_inserter(result),
                     [](instruction_ref live) { return live->name() != "@param"; });
        stop = true;
    });
    return result;
}

static std::string assign_op(const std::vector<instruction_ref>& splits)
{
    static std::unordered_map<std::string, std::string> m = {
        {"reduce_sum", "assign_add"},
        {"reduce_mean", "assign_add"},
        {"reduce_prod", "assign_mul"},
        {"reduce_max", "assign_max"},
        {"reduce_min", "assign_min"},
    };
    return m.at(splits.front()->name());
}

static std::vector<instruction_ref>
insert_module_inline(module& m, instruction_ref ins, const module::with_inputs& mwi)
{
    auto param_map = mwi.mod.get_ins_param_map(mwi.inputs, true);
    return m.insert_instructions(ins, &mwi.mod, &param_map);
}

static std::size_t get_reduce_size(module_ref rm)
{
    auto ins = std::find_if(rm->begin(), rm->end(), &is_reduce);
    assert(ins != rm->end());
    return ins->inputs().front()->get_shape().elements() / ins->get_shape().elements();
}

void split_reduce::apply(module_pass_manager& mpm) const
{
    for(auto ins : iterator_for(mpm.get_module()))
    {
        if(ins->name() != "fused_reduce")
            continue;
        auto* rm    = ins->module_inputs().front();
        if(get_reduce_size(rm) < split_size)
            continue;
        auto splits = find_split(rm);
        if(splits.empty())
            continue;
        // Only use split reduce with float for now
        if(not std::all_of(splits.begin(), splits.end(), [](instruction_ref split) {
               return split->get_shape().type() == shape::float_type;
           }))
            continue;
        auto v    = ins->get_operator().to_value();
        auto axes = v["axes"].to_vector<std::int64_t>();

        auto alive = get_alive(rm, splits);

        std::array<module::with_inputs, 2> mods;
        if(not alive.empty())
        {
            auto mods3 = rm->split(ins->inputs(), alive, splits);
            auto r     = insert_module_inline(mpm.get_module(), ins, mods3[0]);
            mods3[1].replace(alive, r);
            mods3[2].replace(alive, r);
            mods = {std::move(mods3[1]), std::move(mods3[2])};
        }
        else
        {
            mods = rm->split(ins->inputs(), splits);
        }

        auto* splitm = mpm.create_module(rm->name() + "_split", std::move(mods[0].mod));
        splitm->set_bypass();

        // Insert split reduce
        auto split_reduce = mpm.get_module().insert_instruction(
            ins,
            make_op("split_fused_reduce", {{"axes", axes}, {"assign", assign_op(splits)}}),
            mods[0].inputs,
            {splitm});

        mods[1].replace(splits.front(), split_reduce);
        auto replaced = insert_module_inline(mpm.get_module(), ins, mods[1]);
        assert(replaced.size() == 1);
        mpm.get_module().replace_instruction(ins, replaced.front());
    }
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
