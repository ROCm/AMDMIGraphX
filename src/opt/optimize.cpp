#include <migraph/optimize.hpp>
#include <migraph/program.hpp>
#include <migraph/instruction.hpp>
#include <migraph/operators.hpp>
#include <migraph/iterator_for.hpp>

namespace migraph {
void optimize::apply(program &p) const
{
    std::cout << p << std::endl;
    for(auto ins : iterator_for(p)) {
        
    }
}
} // namespace migraph
