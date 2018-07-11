#ifndef MIGRAPH_GUARD_MIGRAPHLIB_INSTRUCTION_HPP
#define MIGRAPH_GUARD_MIGRAPHLIB_INSTRUCTION_HPP

#include <migraph/literal.hpp>
#include <migraph/shape.hpp>
#include <migraph/builtin.hpp>
#include <migraph/instruction_ref.hpp>
#include <migraph/erase.hpp>
#include <string>

namespace migraph {

shape compute_shape(operation op, std::vector<instruction_ref> args);

struct instruction
{
    instruction() {}

    instruction(operation o, shape r, std::vector<instruction_ref> args)
        : op(std::move(o)), result(std::move(r)), arguments(std::move(args))
    {
    }

    instruction(literal l) : op(builtin::literal{}), result(l.get_shape()), lit(std::move(l)) {}

    void replace(operation o, shape r, std::vector<instruction_ref> args)
    {
        op = o;
        replace(std::move(r));
        replace(std::move(args));
    }

    void replace(shape r)
    {
        if(r != result)
        {
            result = r;
            for(auto&& ins : output)
            {
                ins->replace(compute_shape(ins->op, ins->arguments));
            }
        }
    }

    void replace(std::vector<instruction_ref> args)
    {
        clear_arguments();
        arguments = std::move(args);
    }

    void clear_arguments()
    {
        for(auto&& arg : arguments)
        {
            migraph::erase(arg->output, *this);
        }
        arguments.clear();
    }

    friend bool operator==(const instruction& i, instruction_ref ref)
    {
        return std::addressof(i) == std::addressof(*ref);
    }

    bool valid(instruction_ref start) const
    {
        std::vector<shape> shapes(arguments.size());
        std::transform(arguments.begin(), arguments.end(), shapes.begin(), [](instruction_ref ins) {
            return ins->result;
        });
        shape computed;
        try
        {
            computed = op.compute_shape(shapes);
        }
        catch(migraph::exception&)
        {
            return false;
        }
        return result == computed &&
               std::all_of(output.begin(),
                           output.end(),
                           [&](instruction_ref i) {
                               return std::find(i->arguments.begin(), i->arguments.end(), *this) !=
                                      i->arguments.end();
                           }) &&
               std::all_of(arguments.begin(), arguments.end(), [&](instruction_ref i) {
                   auto self = std::find(i->output.begin(), i->output.end(), *this);
                   return self != i->output.end() &&
                          std::distance(start, i) < std::distance(start, *self);
               });
    }

    friend bool operator==(instruction_ref ref, const instruction& i) { return i == ref; }

    friend bool operator!=(const instruction& i, instruction_ref ref) { return !(i == ref); }

    friend bool operator!=(instruction_ref ref, const instruction& i) { return !(i == ref); }

    operation op;
    shape result;
    std::vector<instruction_ref> output;
    std::vector<instruction_ref> arguments;
    literal lit;
};

inline void backreference(instruction_ref ref)
{
    for(auto&& arg : ref->arguments)
        arg->output.push_back(ref);
}

// TODO: Move to a cpp file
// TODO: Use const ref for vector
inline shape compute_shape(operation op, std::vector<instruction_ref> args)
{
    std::vector<shape> shapes(args.size());
    std::transform(
        args.begin(), args.end(), shapes.begin(), [](instruction_ref i) { return i->result; });
    return op.compute_shape(shapes);
}

} // namespace migraph

#endif
