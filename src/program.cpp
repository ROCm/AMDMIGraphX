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

instruction* program::add_instruction(operation op, std::vector<instruction*> args)
{
    assert(
        std::all_of(args.begin(), args.end(), [&](instruction* x) { return has_instruction(x); }) &&
        "Argument is not an exisiting instruction");
    std::vector<shape> shapes(args.size());
    std::transform(
        args.begin(), args.end(), shapes.begin(), [](instruction* ins) { return ins->result; });
    shape r = op.compute_shape(shapes);
    impl->instructions.push_back({op, r, args});
    assert(impl->instructions.back().arguments == args);
    return std::addressof(impl->instructions.back());
}

instruction* program::add_literal(literal l)
{
    impl->instructions.emplace_back(std::move(l));
    return std::addressof(impl->instructions.back());
}

instruction* program::add_parameter(std::string name, shape s)
{
    impl->instructions.push_back({builtin::param{std::move(name)}, s, {}});
    return std::addressof(impl->instructions.back());
}

bool program::has_instruction(const instruction* ins) const
{
    return std::find_if(impl->instructions.begin(),
                        impl->instructions.end(),
                        [&](const instruction& x) { return ins == std::addressof(x); }) !=
           impl->instructions.end();
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
                           [&](instruction* i) { return results.at(i); });
            result = ins.op.compute(values);
        }
        results.emplace(std::addressof(ins), result);
    }
    return literal{result.get_shape(), result.data()};
}

void program::print() const
{
    std::unordered_map<const instruction*, std::string> names;
    int count = 0;

    for(auto& ins : impl->instructions)
    {
        std::string var_name = "@" + std::to_string(count);
        if(starts_with(ins.op.name(), "@param"))
        {
            var_name = ins.op.name().substr(7);
        }

        std::cout << var_name << " = ";

        std::cout << ins.op.name();

        if(ins.op.name() == "@literal")
        {
            if(ins.lit.get_shape().elements() > 10)
                std::cout << "{ ... }";
            else
                std::cout << "{" << ins.lit << "}";
        }

        if(!ins.arguments.empty())
        {
            char delim = '(';
            for(auto&& arg : ins.arguments)
            {
                assert(this->has_instruction(arg) && "Instruction not found");
                std::cout << delim << names.at(arg);
                delim = ',';
            }
            std::cout << ")";
        }

        std::cout << " -> " << ins.result;

        std::cout << std::endl;

        names.emplace(std::addressof(ins), var_name);
        count++;
    }
}

} // namespace rtg
