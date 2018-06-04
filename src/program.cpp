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
    // TODO: Use move
    shape r     = compute_shape(op, args);
    auto result = impl->instructions.insert(ins, {op, r, args});
    backreference(result);
    assert(result->arguments == args);
    return result;
}

instruction_ref
program::replace_instruction(instruction_ref ins, operation op, std::vector<instruction_ref> args)
{
    assert(std::all_of(
               args.begin(), args.end(), [&](instruction_ref x) { return has_instruction(x); }) &&
           "Argument is not an exisiting instruction");

    shape r = compute_shape(op, args);
    ins->replace(op, r, args);
    backreference(ins);
    return ins;
}

instruction_ref program::add_literal(literal l)
{
    impl->instructions.emplace_front(std::move(l));
    return impl->instructions.begin();
}

instruction_ref program::add_outline(shape s)
{
    impl->instructions.push_front({builtin::outline{s}, s, {}});
    return impl->instructions.begin();
}

instruction_ref program::add_parameter(std::string name, shape s)
{
    impl->instructions.push_front({builtin::param{std::move(name)}, s, {}});
    return impl->instructions.begin();
}

shape program::get_parameter_shape(std::string name)
{
    auto ins = std::find_if(
               impl->instructions.begin(), impl->instructions.end(), [&](const instruction& x) {
                   if(x.op.name() == "@param") 
                   {
                        return any_cast<builtin::param>(x.op).parameter == name;
                   }
                   else
                   {
                        return false;
                   }
               });
    if (ins != this->end()) return ins->result;
    else return {};
}

bool program::has_instruction(instruction_ref ins) const
{
    return std::find_if(
               impl->instructions.begin(), impl->instructions.end(), [&](const instruction& x) {
                   return std::addressof(*ins) == std::addressof(x);
               }) != impl->instructions.end();
}

instruction_ref program::begin() { return impl->instructions.begin(); }
instruction_ref program::end() { return impl->instructions.end(); }

instruction_ref program::validate() const
{
    return std::find_if(impl->instructions.begin(),
                        impl->instructions.end(),
                        [](const instruction& i) { return i.valid(); });
}

void program::compile(const target& t)
{
    assert(this->validate() != impl->instructions.end());
    t.apply(*this);
    if(this->validate() == impl->instructions.end())
        RTG_THROW("Invalid program from compilation");
}

literal program::eval(std::unordered_map<std::string, argument> params) const
{
    assert(this->validate() != impl->instructions.end());
    std::unordered_map<const instruction*, argument> results;
    argument result;
    for(auto& ins : impl->instructions)
    {
        if(ins.op.name() == "@literal")
        {
            result = ins.lit.get_argument();
        }
        else if(ins.op.name() == "@param")
        {
            result = params.at(any_cast<builtin::param>(ins.op).parameter);
        }
        else if(ins.op.name() == "@outline")
        {
            result = argument{ins.result, nullptr};
        }
        else
        {
            std::vector<argument> values(ins.arguments.size());
            std::transform(ins.arguments.begin(),
                           ins.arguments.end(),
                           values.begin(),
                           [&](instruction_ref i) { return results.at(std::addressof(*i)); });
            result = ins.op.compute(ins.result, values);
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
        if(ins.op.name() == "@param")
        {
            var_name = any_cast<builtin::param>(ins.op).parameter;
        }

        os << var_name << " = ";

        os << ins.op;

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
