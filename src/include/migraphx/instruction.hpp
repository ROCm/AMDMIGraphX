#ifndef MIGRAPHX_GUARD_MIGRAPHLIB_INSTRUCTION_HPP
#define MIGRAPHX_GUARD_MIGRAPHLIB_INSTRUCTION_HPP

#include <migraphx/literal.hpp>
#include <migraphx/shape.hpp>
#include <migraphx/instruction_ref.hpp>
#include <migraphx/operation.hpp>
#include <migraphx/erase.hpp>
#include <migraphx/config.hpp>
#include <string>
#include <utility>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

shape compute_shape(const operation& op, const std::vector<instruction_ref>& args);
std::vector<shape> to_shapes(const std::vector<instruction_ref>& args);

struct instruction
{
    instruction() {}

    instruction(operation o, shape r, std::vector<instruction_ref> args);

    instruction(literal l);

    void replace(operation o);

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

    friend bool operator==(const instruction& x, const instruction& y);

    friend bool operator!=(const instruction& x, const instruction& y);

    friend bool operator==(instruction_ref ref, const instruction& i);

    friend bool operator!=(const instruction& i, instruction_ref ref);

    friend bool operator!=(instruction_ref ref, const instruction& i);

    void add_output(instruction_ref ins);

    template <class T>
    void remove_output(const T& ins)
    {
        migraphx::erase(output, ins);
    }

    static void backreference(instruction_ref ref);

    static void replace_argument(instruction_ref ins, instruction_ref old, instruction_ref new_ins);

    static void
    replace(instruction_ref ins, operation o, const shape& r, std::vector<instruction_ref> args);

    bool can_eval() const;

    argument eval(bool check_eval = true) const;

    void finalize(context& ctx);

    static instruction_ref get_output_alias(instruction_ref ins, bool shallow = false);

    private:
    // internal
    void replace(operation o, const shape& r, std::vector<instruction_ref> args);

    // internal
    void replace(std::vector<instruction_ref> args);

    // internal
    void replace_argument(instruction_ref old, instruction_ref new_ins);

    void replace(const shape& r);

    operation op;
    shape result{};
    std::vector<instruction_ref> output;
    std::vector<instruction_ref> arguments;
    literal lit;
};
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

namespace std {
template <>
struct hash<migraphx::instruction_ref>
{
    using argument_type = migraphx::instruction_ref;
    using result_type   = std::size_t;
    result_type operator()(const argument_type& x) const noexcept
    {
        return std::hash<migraphx::instruction*>{}(&*x);
    }
};

} // namespace std

#endif
