#include <rtg/program.hpp>
#include <rtg/stringutils.hpp>
#include <iostream>
#include <algorithm>

namespace rtg {

literal program::eval(std::unordered_map<std::string, argument> params) const
{
    std::unordered_map<const instruction*, argument> results;
    argument result;
    for(auto& ins:instructions)
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
            std::transform(ins.arguments.begin(), ins.arguments.end(), values.begin(), [&](instruction * i) {
                return results.at(i);
            });
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

    for(auto& ins:instructions)
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
            std::cout << "{" << ins.lit << "}";
        }

        if(!ins.arguments.empty())
        {
            char delim = '(';
            for(auto&& arg:ins.arguments)
            {
                std::cout << delim << names.at(arg);
                delim = ',';
            }
            std::cout << ")";
        }

        std::cout << " -> " << ins.result;

        std::cout << std::endl;

        names.emplace(std::addressof(ins), var_name);

    }
}

}

