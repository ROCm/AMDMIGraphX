#ifndef MIGRAPH_GUARD_MIGRAPHLIB_INSTRUCTION_HPP
#define MIGRAPH_GUARD_MIGRAPHLIB_INSTRUCTION_HPP

#include <migraph/literal.hpp>
#include <migraph/shape.hpp>
#include <migraph/instruction_ref.hpp>
#include <migraph/operation.hpp>
#include <string>
#include <utility>

namespace migraph {

shape compute_shape(const operation& op, const std::vector<instruction_ref>& args);

struct instruction
{
    instruction() {}

    instruction(operation o, shape r, std::vector<instruction_ref> args);

    instruction(literal l);

    void replace(const shape& r);

    void recompute_shape();

    void clear_arguments();

    friend bool operator==(const instruction& i, instruction_ref ref);

    bool valid(instruction_ref start) const;

    bool valid() const;

    shape get_shape() const;
    const literal& get_literal() const;

    const operation& get_operator() const;

    std::string name() const;

    const std::vector<instruction_ref>& inputs() const;

    const std::vector<instruction_ref>& outputs() const;

    friend bool operator==(instruction_ref ref, const instruction& i);

    friend bool operator!=(const instruction& i, instruction_ref ref);

    friend bool operator!=(instruction_ref ref, const instruction& i);

    void add_output(instruction_ref ins);

    template <class T>
    void remove_output(const T& ins);

    static void backreference(instruction_ref ref);

    static void replace_argument(instruction_ref ins, instruction_ref old, instruction_ref new_ins);

    static void
    replace(instruction_ref ins, operation o, const shape& r, std::vector<instruction_ref> args);

    private:
    // internal
    void replace(operation o, const shape& r, std::vector<instruction_ref> args);

    // internal
    void replace(std::vector<instruction_ref> args);

    // internal
    void replace_argument(instruction_ref old, instruction_ref new_ins);

    operation op;
    shape result;
    std::vector<instruction_ref> output;
    std::vector<instruction_ref> arguments;
    literal lit;
};

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
