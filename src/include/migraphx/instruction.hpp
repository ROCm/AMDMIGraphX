#ifndef MIGRAPHX_GUARD_MIGRAPHLIB_INSTRUCTION_HPP
#define MIGRAPHX_GUARD_MIGRAPHLIB_INSTRUCTION_HPP

#include <migraphx/literal.hpp>
#include <migraphx/shape.hpp>
#include <migraphx/instruction_ref.hpp>
#include <migraphx/module_ref.hpp>
#include <migraphx/operation.hpp>
#include <migraphx/erase.hpp>
#include <migraphx/config.hpp>
#include <string>
#include <utility>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

shape compute_shape(const operation& op, const std::vector<instruction_ref>& args);
shape compute_shape(const operation& op,
                    const std::vector<instruction_ref>& args,
                    const std::vector<module_ref>& mods);
std::vector<shape> to_shapes(const std::vector<instruction_ref>& args);
std::vector<shape> try_compute_shape(const operation& op, const std::vector<shape>& inputs);

struct instruction
{
    instruction() {}

    instruction(operation o, shape r, std::vector<instruction_ref> args);

    instruction(operation o,
                shape r,
                std::vector<instruction_ref> args,
                std::vector<module_ref> modules);

    instruction(literal l);

    void replace(operation o);

    void recompute_shape();

    void clear_arguments();

    friend bool operator==(const instruction& i, instruction_ref ref);

    bool valid(instruction_ref start, bool check_order = false) const;

    bool valid() const;

    shape get_shape() const;
    const literal& get_literal() const;

    const operation& get_operator() const;

    std::string name() const;

    const std::vector<instruction_ref>& inputs() const;

    const std::vector<module_ref>& module_inputs() const;

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

    static void replace_refs(instruction_ref ins,
                             const std::unordered_map<instruction_ref, instruction_ref>& map_insts,
                             const std::unordered_map<module_ref, module_ref>& map_mods);

    static void backreference(instruction_ref ref);

    static void replace_argument(instruction_ref ins, instruction_ref old, instruction_ref new_ins);

    static void replace_mod_argument(instruction_ref ins, module_ref old, module_ref new_mod);

    static void
    replace(instruction_ref ins, operation o, const shape& r, std::vector<instruction_ref> args);

    static void replace(instruction_ref ins,
                        operation o,
                        const shape& r,
                        std::vector<instruction_ref> args,
                        std::vector<module_ref> module_args);

    bool can_eval() const;

    argument eval(bool check_eval = true) const;

    void finalize(context& ctx);

    static instruction_ref get_output_alias(instruction_ref ins, bool shallow = false);

    void set_normalized(bool value = true);
    bool is_normalized() const;

    bool need_normalization() const;

    operation normalized_operator() const;

    void debug_print() const;

    static void print(std::ostream& os,
                      instruction_ref ins,
                      const std::unordered_map<instruction_ref, std::string>& names);

    private:
    // internal
    void replace(operation o, const shape& r, std::vector<instruction_ref> args);

    // internal
    void replace(operation o,
                 const shape& r,
                 std::vector<instruction_ref> args,
                 std::vector<module_ref> mdl_args);

    // internal
    void replace(std::vector<instruction_ref> args);

    // internal
    void replace(std::vector<instruction_ref> args, std::vector<module_ref> mdl_args);

    // internal
    void replace_argument(instruction_ref old, instruction_ref new_ins);

    // internal
    void replace_mod_argument(module_ref old, module_ref new_mod);

    void replace(const shape& r);

    operation op;
    shape result{};
    std::vector<instruction_ref> output;
    std::vector<instruction_ref> arguments;
    std::vector<module_ref> module_args;
    literal lit;
    bool normalized = false;
};
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
