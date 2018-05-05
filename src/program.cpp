#include <rtg/program.hpp>
#include <rtg/stringutils.hpp>
#include <rtg/instruction.hpp>
#include <iostream>
#include <algorithm>

namespace rtg {

struct program_impl
{
    // A list is used to keep references to an instruction stable
    std::list<instruction> instructions;
};

program::program() : impl(std::make_unique<program_impl>()) {}

program::program(program&&) noexcept = default;
program& program::operator=(program&&) noexcept = default;
program::~program() noexcept                    = default;

instruction_ref program::add_instruction(operation op, std::vector<instruction_ref> args)
{
    return insert_instruction(impl->instructions.end(), std::move(op), std::move(args));
}
instruction_ref
program::insert_instruction(instruction_ref ins, operation op, std::vector<instruction_ref> args)
{
    assert(std::all_of(
               args.begin(), args.end(), [&](instruction_ref x) { return has_instruction(x); }) &&
           "Argument is not an exisiting instruction");
    std::vector<shape> shapes(args.size());
    std::transform(
        args.begin(), args.end(), shapes.begin(), [](instruction_ref i) { return i->result; });
    shape r     = op.compute_shape(shapes);
    auto result = impl->instructions.insert(ins, {op, r, args});
    assert(result->arguments == args);
    for(auto&& arg : args)
        arg->output.push_back(result);
    return result;
}

instruction_ref program::add_literal(literal l)
{
    impl->instructions.emplace_back(std::move(l));
    return std::prev(impl->instructions.end());
}

instruction_ref program::add_parameter(std::string name, shape s)
{
    impl->instructions.push_back({builtin::param{std::move(name)}, s, {}});
    return std::prev(impl->instructions.end());
}

bool program::has_instruction(instruction_ref ins) const
{
    return std::find_if(
               impl->instructions.begin(), impl->instructions.end(), [&](const instruction& x) {
                   return std::addressof(*ins) == std::addressof(x);
               }) != impl->instructions.end();
}

literal program::eval(std::unordered_map<std::string, argument> params) const
{
    std::unordered_map<const instruction*, argument> results;
    argument result;
    for(auto& ins : impl->instructions)
    {
        if(ins.op.name() == "@literal")
        {
            result = ins.lit.get_argument();
        }
        else if(starts_with(ins.op.name(), "@param"))
        {
            result = params.at(ins.op.name().substr(7));
        }
        else
        {
            std::vector<argument> values(ins.arguments.size());
            std::transform(ins.arguments.begin(),
                           ins.arguments.end(),
                           values.begin(),
                           [&](instruction_ref i) { return results.at(std::addressof(*i)); });
            result = ins.op.compute(values);
        }
        results.emplace(std::addressof(ins), result);
    }
    return literal{result.get_shape(), result.data()};
}

std::ostream& operator<<(std::ostream& os, const program& p)
{
    std::unordered_map<const instruction*, std::string> names;
    int count = 0;

    for(auto& ins : p.impl->instructions)
    {
        std::string var_name = "@" + std::to_string(count);
        if(starts_with(ins.op.name(), "@param"))
        {
            var_name = ins.op.name().substr(7);
        }

        os << var_name << " = ";

        os << ins.op.name();

        if(ins.op.name() == "@literal")
        {
            if(ins.lit.get_shape().elements() > 10)
                os << "{ ... }";
            else
                os << "{" << ins.lit << "}";
        }

        if(!ins.arguments.empty())
        {
            char delim = '(';
            for(auto&& arg : ins.arguments)
            {
                assert(p.has_instruction(arg) && "Instruction not found");
                os << delim << names.at(std::addressof(*arg));
                delim = ',';
            }
            os << ")";
        }

        os << " -> " << ins.result;

        os << std::endl;

        names.emplace(std::addressof(ins), var_name);
        count++;
    }
    return os;
}

} // namespace rtg
