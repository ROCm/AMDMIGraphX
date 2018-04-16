#include <rtg/program.hpp>
#include <rtg/stringutils.hpp>
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

}

