#include <migraphx/instruction.hpp>
#include <migraphx/builtin.hpp>
#include <migraphx/erase.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

instruction::instruction(operation o, shape r, std::vector<instruction_ref> args)
    : op(std::move(o)), result(std::move(r)), arguments(std::move(args))
{
}

instruction::instruction(literal l)
    : op(builtin::literal{}), result(l.get_shape()), lit(std::move(l))
{
}

void instruction::replace(const shape& r)
{
    if(r != result)
    {
        result = r;
        for(auto&& ins : output)
        {
            assert(ins->name().front() != '@');
            ins->recompute_shape();
        }
    }
}

void instruction::replace(operation o)
{
    op = std::move(o);
    recompute_shape();
}

void instruction::recompute_shape() { replace(compute_shape(op, arguments)); }

void instruction::clear_arguments()
{
    for(auto&& arg : arguments)
    {
        arg->remove_output(*this);
    }
    arguments.clear();
}

bool operator==(const instruction& i, instruction_ref ref)
{
    return std::addressof(i) == std::addressof(*ref);
}

bool instruction::valid(instruction_ref start) const
{
    return valid() && std::all_of(arguments.begin(), arguments.end(), [&](instruction_ref i) {
               auto self = std::find(i->outputs().begin(), i->outputs().end(), *this);
               return self != i->outputs().end() &&
                      std::distance(start, i) < std::distance(start, *self);
           });
}

bool instruction::valid() const
{
    shape computed;
    if(op.name() == "@literal")
    {
        computed = lit.get_shape();
    }
    else if(op.name() == "@param")
    {
        computed = result;
    }
    else if(op.name() == "add_return")
    {
        computed = {};
    }
    else
    {
        try
        {
            computed = compute_shape(op, arguments);
        }
        catch(migraphx::exception&)
        {
            return false;
        }
    }

    return result == computed && std::all_of(output.begin(), output.end(), [&](instruction_ref i) {
               return std::find(i->inputs().begin(), i->inputs().end(), *this) != i->inputs().end();
           });
}

shape instruction::get_shape() const { return result; }
const literal& instruction::get_literal() const
{
    assert(op.name() == "@literal");
    return lit;
}

const operation& instruction::get_operator() const { return op; }

std::string instruction::name() const { return op.name(); }

const std::vector<instruction_ref>& instruction::inputs() const { return arguments; }

const std::vector<instruction_ref>& instruction::outputs() const { return output; }

bool operator==(const instruction& x, const instruction& y)
{
    if(std::tie(x.result, x.op, x.arguments) != std::tie(y.result, y.op, y.arguments))
        return false;
    if(x.name() == "@literal")
        return x.lit == y.lit;
    return true;
}

bool operator!=(const instruction& x, const instruction& y) { return !(x == y); }

bool operator==(instruction_ref ref, const instruction& i) { return i == ref; }

bool operator!=(const instruction& i, instruction_ref ref) { return !(i == ref); }

bool operator!=(instruction_ref ref, const instruction& i) { return !(i == ref); }

void instruction::add_output(instruction_ref ins)
{
    if(std::find(output.begin(), output.end(), ins) == output.end())
        output.push_back(ins);
}

void instruction::backreference(instruction_ref ref)
{
    for(auto&& arg : ref->inputs())
        arg->add_output(ref);
}

void instruction::replace_argument(instruction_ref ins,
                                   instruction_ref old,
                                   instruction_ref new_ins)
{
    ins->replace_argument(old, new_ins);
    backreference(ins);
    ins->recompute_shape();
}

void instruction::replace(instruction_ref ins,
                          operation o,
                          const shape& r,
                          std::vector<instruction_ref> args)
{
    ins->replace(std::move(o), r, std::move(args));
    backreference(ins);
}

void instruction::replace(operation o, const shape& r, std::vector<instruction_ref> args)
{
    op = std::move(o);
    replace(r);
    replace(std::move(args));
}

void instruction::replace(std::vector<instruction_ref> args)
{
    clear_arguments();
    arguments = std::move(args);
}

void instruction::replace_argument(instruction_ref old, instruction_ref new_ins)
{
    assert(std::any_of(arguments.begin(), arguments.end(), [&](auto i) { return i == old; }));
    std::replace(arguments.begin(), arguments.end(), old, new_ins);
    old->remove_output(*this);
}

bool instruction::can_eval() const
{
    if(op.name() == "@literal")
    {
        return true;
    }
    else if(is_context_free(op))
    {
        return std::all_of(
            this->inputs().begin(), this->inputs().end(), [](auto arg) { return arg->can_eval(); });
    }
    else
    {
        return false;
    }
}

argument instruction::eval(bool check_eval) const
{
    if(op.name() == "@literal")
    {
        return this->get_literal().get_argument();
    }
    if(is_context_free(op))
    {
        if(check_eval and not this->can_eval())
            return {};
        std::vector<argument> args;
        std::transform(this->inputs().begin(),
                       this->inputs().end(),
                       std::back_inserter(args),
                       [](auto arg) { return arg->eval(false); });
        return op.compute(result, args);
    }
    return {};
}

void instruction::finalize(context& ctx)
{
    if(has_finalize(this->op))
        this->op.finalize(ctx, this->get_shape(), to_shapes(this->inputs()));
}

instruction_ref instruction::get_output_alias(instruction_ref ins, bool shallow)
{
    auto i = ins->get_operator().output_alias(to_shapes(ins->inputs()));
    if(i < 0)
        return ins;
    if(shallow)
        return ins->inputs().at(i);
    return get_output_alias(ins->inputs().at(i));
}

std::vector<shape> to_shapes(const std::vector<instruction_ref>& args)
{
    std::vector<shape> shapes(args.size());
    std::transform(
        args.begin(), args.end(), shapes.begin(), [](instruction_ref i) { return i->get_shape(); });
    return shapes;
}

shape compute_shape(const operation& op, const std::vector<instruction_ref>& args)
{
    return op.compute_shape(to_shapes(args));
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
