#ifndef MIGRAPH_GUARD_MIGRAPHLIB_INSTRUCTION_HPP
#define MIGRAPH_GUARD_MIGRAPHLIB_INSTRUCTION_HPP

#include <migraph/literal.hpp>
#include <migraph/shape.hpp>
#include <migraph/builtin.hpp>
#include <migraph/instruction_ref.hpp>
#include <migraph/operation.hpp>
#include <migraph/erase.hpp>
#include <string>
#include <utility>

namespace migraph {

shape compute_shape(const operation& op, const std::vector<instruction_ref>& args);

struct instruction
{
    instruction() {}

    instruction(operation o, shape r, std::vector<instruction_ref> args)
        : op(std::move(o)), result(std::move(r)), arguments(std::move(args))
    {
    }

    instruction(literal l) : op(builtin::literal{}), result(l.get_shape()), lit(std::move(l)) {}

    void replace(const shape& r)
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

    void recompute_shape() { replace(compute_shape(op, arguments)); }

    void clear_arguments()
    {
        for(auto&& arg : arguments)
        {
            arg->remove_output(*this);
        }
        arguments.clear();
    }

    friend bool operator==(const instruction& i, instruction_ref ref)
    {
        return std::addressof(i) == std::addressof(*ref);
    }

    bool valid(instruction_ref start) const
    {
        return valid() && std::all_of(arguments.begin(), arguments.end(), [&](instruction_ref i) {
                   auto self = std::find(i->outputs().begin(), i->outputs().end(), *this);
                   return self != i->outputs().end() &&
                          std::distance(start, i) < std::distance(start, *self);
               });
    }

    bool valid() const
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
        else
        {
            try
            {
                computed = compute_shape(op, arguments);
            }
            catch(migraph::exception&)
            {
                return false;
            }
        }
        return result == computed &&
               std::all_of(output.begin(), output.end(), [&](instruction_ref i) {
                   return std::find(i->inputs().begin(), i->inputs().end(), *this) !=
                          i->inputs().end();
               });
    }

    shape get_shape() const { return result; }
    const literal& get_literal() const
    {
        assert(op.name() == "@literal");
        return lit;
    }

    const operation& get_operator() const { return op; }

    std::string name() const { return op.name(); }

    const std::vector<instruction_ref>& inputs() const { return arguments; }

    const std::vector<instruction_ref>& outputs() const { return output; }

    friend bool operator==(instruction_ref ref, const instruction& i) { return i == ref; }

    friend bool operator!=(const instruction& i, instruction_ref ref) { return !(i == ref); }

    friend bool operator!=(instruction_ref ref, const instruction& i) { return !(i == ref); }

    void add_output(instruction_ref ins)
    {
        if(std::find(output.begin(), output.end(), ins) == output.end())
            output.push_back(ins);
    }

    template <class T>
    void remove_output(const T& ins)
    {
        migraph::erase(output, ins);
    }

    static void backreference(instruction_ref ref)
    {
        for(auto&& arg : ref->inputs())
            arg->add_output(ref);
    }

    static void replace_argument(instruction_ref ins, instruction_ref old, instruction_ref new_ins)
    {
        ins->replace_argument(old, new_ins);
        backreference(ins);
        ins->recompute_shape();
    }

    static void
    replace(instruction_ref ins, operation o, const shape& r, std::vector<instruction_ref> args)
    {
        ins->replace(std::move(o), r, std::move(args));
        backreference(ins);
    }

    private:
    // internal
    void replace(operation o, const shape& r, std::vector<instruction_ref> args)
    {
        op = std::move(o);
        replace(r);
        replace(std::move(args));
    }

    // internal
    void replace(std::vector<instruction_ref> args)
    {
        clear_arguments();
        arguments = std::move(args);
    }

    // internal
    void replace_argument(instruction_ref old, instruction_ref new_ins)
    {
        std::replace(arguments.begin(), arguments.end(), old, new_ins);
        old->remove_output(*this);
    }

    operation op;
    shape result;
    std::vector<instruction_ref> output;
    std::vector<instruction_ref> arguments;
    literal lit;
};

// TODO: Move to a cpp file
inline shape compute_shape(const operation& op, const std::vector<instruction_ref>& args)
{
    std::vector<shape> shapes(args.size());
    std::transform(
        args.begin(), args.end(), shapes.begin(), [](instruction_ref i) { return i->get_shape(); });
    return op.compute_shape(shapes);
}

} // namespace migraph

namespace std {
template <>
struct hash<migraph::instruction_ref>
{
    using argument_type = migraph::instruction_ref;
    using result_type   = std::size_t;
    result_type operator()(const argument_type& x) const noexcept
    {
        return std::hash<migraph::instruction*>{}(&*x);
    }
};
} // namespace std

#endif
