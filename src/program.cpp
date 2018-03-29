#include <rtg/program.hpp>
#include <algorithm>

namespace rtg {

literal program::eval() const
{
    std::unordered_map<const instruction*, argument> results;
    argument result;
    for(auto& ins:instructions)
    {
        if(ins.name == "literal")
        {
            result = ins.lit.get_argument();
        }
        else
        {
            auto&& op = ops.at(ins.name);
            std::vector<argument> values(ins.arguments.size());
            std::transform(ins.arguments.begin(), ins.arguments.end(), values.begin(), [&](instruction * i) {
                return results.at(i);
            });
            result = op.compute(values);
        }
        results.emplace(std::addressof(ins), result);
    }
    return literal{result.get_shape(), result.data()};
}

}

