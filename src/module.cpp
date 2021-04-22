#include <migraphx/module.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/target.hpp>
#include <migraphx/env.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/time.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/register_target.hpp>
#include <migraphx/make_op.hpp>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <set>
#include <utility>
#include <unordered_set>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct module_impl
{
    // A list is used to keep references to an instruction stable
    std::list<instruction> instructions;
    std::vector<std::string> input_names;
    std::string name;
};

const operation& get_operation(instruction_ref ins) { return ins->get_operator(); }

module::module(const std::string& name) : impl(std::make_unique<module_impl>())
{
    impl->name = name;
}

module::module(module&&) noexcept = default;
module::~module() noexcept        = default;

// copy constructor
module::module(const module& m) { assign(m); }

// copy assignment operator
module& module::operator=(module m)
{
    std::swap(m.impl, this->impl);
    return *this;
}

std::string module::name() const { return impl->name; }

void module::assign(const module& m)
{
    // clean the current module
    if(!impl)
    {
        impl = std::make_unique<module_impl>();
    }
    else if(!impl->instructions.empty())
    {
        impl->instructions.clear();
    }
    impl->input_names = m.impl->input_names;
    impl->name        = m.impl->name;

    std::unordered_map<instruction_ref, instruction_ref> ins_map;
    for(auto ins : iterator_for(m))
    {
        instruction_ref copy_ins{};
        if(ins->name() == "@literal")
        {
            auto l   = ins->get_literal();
            copy_ins = impl->instructions.insert(impl->instructions.end(), instruction{l});
        }
        else if(ins->name() == "@param")
        {
            auto&& name = any_cast<builtin::param>(ins->get_operator()).parameter;
            auto s      = ins->get_shape();
            copy_ins    = impl->instructions.insert(impl->instructions.end(),
                                                 {builtin::param{name}, std::move(s), {}});
        }
        else if(ins->name() == "@outline")
        {
            auto s = ins->get_shape();
            copy_ins =
                impl->instructions.insert(impl->instructions.end(), {builtin::outline{s}, s, {}});
        }
        else
        {
            // if there are sub_module inputs, need to make a copy of the submodule
            auto module_args = ins->module_inputs();
            // retrieve its mapped input
            auto inputs = ins->inputs();
            std::vector<instruction_ref> copy_inputs(inputs.size());
            std::transform(inputs.begin(), inputs.end(), copy_inputs.begin(), [&](auto i) {
                return contains(ins_map, i) ? ins_map[i] : i;
            });
            if(ins->name() == "@return")
            {
                copy_ins = add_return(copy_inputs);
            }
            else
            {
                if(module_args.empty())
                {
                    copy_ins = add_instruction(ins->get_operator(), copy_inputs);
                }
                else
                {
                    copy_ins = add_instruction(ins->get_operator(), copy_inputs, module_args);
                }
            }
        }

        ins_map[ins] = copy_ins;
    }
}

instruction_ref module::add_instruction(const operation& op, std::vector<instruction_ref> args)
{
    return insert_instruction(impl->instructions.end(), op, std::move(args));
}
instruction_ref module::insert_instruction(instruction_ref ins,
                                           const operation& op,
                                           std::vector<instruction_ref> args)
{
    assert(not starts_with(op.name(), "@"));
    shape r     = compute_shape(op, args);
    auto result = impl->instructions.insert(ins, {op, r, std::move(args)});
    instruction::backreference(result);
    assert(result->valid(begin()));
    return result;
}

instruction_ref module::add_instruction(const operation& op,
                                        std::vector<instruction_ref> args,
                                        std::vector<module_ref> module_args)
{
    return insert_instruction(
        impl->instructions.end(), op, std::move(args), std::move(module_args));
}

instruction_ref module::insert_instruction(instruction_ref ins,
                                           const operation& op,
                                           std::vector<instruction_ref> args,
                                           std::vector<module_ref> module_args)
{
    assert(not starts_with(op.name(), "@"));
    auto out_shape = compute_shape(op, args, module_args);
    auto result =
        impl->instructions.insert(ins, {op, out_shape, std::move(args), std::move(module_args)});
    instruction::backreference(result);
    assert(result->valid(begin()));
    return result;
}

instruction_ref module::replace_instruction(instruction_ref ins,
                                            const operation& op,
                                            std::vector<instruction_ref> args) MIGRAPHX_TIDY_CONST
{
    assert(not starts_with(op.name(), "@"));
std::cout << "loc31" << std::endl;
    shape r = compute_shape(op, args);
std::cout << "loc32" << std::endl;
    instruction::replace(ins, op, r, std::move(args));
std::cout << "loc33" << std::endl;
    assert(ins->valid(begin()));
std::cout << "loc34" << std::endl;
    return ins;
}

instruction_ref module::replace_instruction(instruction_ref ins,
                                            const operation& op,
                                            std::vector<instruction_ref> args,
                                            std::vector<module_ref> module_args) MIGRAPHX_TIDY_CONST
{
    assert(not starts_with(op.name(), "@"));
    auto out_shape = compute_shape(op, args, module_args);
    instruction::replace(ins, op, out_shape, std::move(args), std::move(module_args));
    assert(ins->valid(begin()));
    return ins;
}

instruction_ref module::replace_instruction(instruction_ref ins, instruction_ref rep)
{
    assert(has_instruction(ins));
    assert(has_instruction(rep));
    assert(ins != rep);

    if(ins == std::prev(this->end()))
    {
        return replace_instruction(ins, make_op("identity"), rep);
    }

    // TODO: Should it be an error if the output is empty?
    if(ins->outputs().empty())
    {
        return rep;
    }
    // Make a copy of outputs which can be changed when calling replace_argument
    auto outputs = ins->outputs();
    for(auto out : outputs)
    {
        // TODO: Check for possible cycles
        if(out != rep)
        {
            instruction::replace_argument(out, ins, rep);
        }
        assert(out->valid(begin()));
    }
    // Replacement should not be dead code unless its the last instruction
    assert(!rep->outputs().empty() or rep == std::prev(end()));
    // Output of the original instruction should only be the replacement or empty
    assert(ins->outputs().empty() or std::all_of(ins->outputs().begin(),
                                                 ins->outputs().end(),
                                                 [&](auto i) { return i == rep; }));
    assert(ins->valid(begin()));
    assert(rep->valid(begin()));
    return rep;
}

instruction_ref module::remove_instruction(instruction_ref ins)
{
    assert(has_instruction(ins));
    assert(ins->outputs().empty());
    ins->clear_arguments();
    return impl->instructions.erase(ins);
}

instruction_ref module::remove_instructions(instruction_ref first, instruction_ref last)
{
    if(first == last)
        return first;
    // TODO: Check every element
    assert(has_instruction(first));
    std::for_each(first, last, [&](instruction& ins) { ins.clear_arguments(); });
    assert(std::all_of(first, last, [&](const instruction& ins) { return ins.outputs().empty(); }));
    return impl->instructions.erase(first, last);
}

instruction_ref module::move_instruction(instruction_ref src, instruction_ref dst)
{
    impl->instructions.splice(dst, impl->instructions, src);
    return src;
}

instruction_ref module::move_instructions(instruction_ref src, instruction_ref dst)
{
    this->move_instruction(src, dst);
    for(auto ins : src->inputs())
        this->move_instruction(ins, src);
    return src;
}

instruction_ref module::add_literal(literal l)
{
    impl->instructions.emplace_front(std::move(l));
    return impl->instructions.begin();
}

instruction_ref module::add_outline(const shape& s)
{
    impl->instructions.push_front({builtin::outline{s}, s, {}});
    return impl->instructions.begin();
}

instruction_ref module::add_parameter(std::string name, shape s)
{
    assert(get_parameter_shape(name) == shape{});
    impl->input_names.push_back(name);

    impl->instructions.push_front({builtin::param{std::move(name)}, std::move(s), {}});
    return impl->instructions.begin();
}

instruction_ref module::add_return(std::vector<instruction_ref> args)
{
    impl->instructions.push_back({builtin::returns{}, {}, std::move(args)});
    auto result = std::prev(impl->instructions.end());
    instruction::backreference(result);
    assert(result->valid(begin()));

    return result;
}

shape module::get_parameter_shape(std::string name) const
{
    auto ins = std::find_if(
        impl->instructions.begin(), impl->instructions.end(), [&](const instruction& x) {
            if(x.name() == "@param")
            {
                return any_cast<builtin::param>(x.get_operator()).parameter == name;
            }
            else
            {
                return false;
            }
        });
    if(ins != this->end())

        return ins->get_shape();
    else
        return {};
}

std::vector<std::string> module::get_parameter_names() const
{
    std::vector<std::string> result = impl->input_names;
    std::unordered_set<std::string> params;
    for(auto&& ins : impl->instructions)
    {
        if(ins.name() == "@param")
        {
            auto&& name = any_cast<builtin::param>(ins.get_operator()).parameter;
            params.insert(name);
        }
    }
    erase_if(result, [&](auto&& name) { return params.count(name) == 0; });
    return result;
}

instruction_ref module::get_parameter(std::string name) const
{
    auto ins = std::find_if(
        impl->instructions.begin(), impl->instructions.end(), [&](const instruction& x) {
            if(x.name() == "@param")
            {
                return any_cast<builtin::param>(x.get_operator()).parameter == name;
            }
            else
            {
                return false;
            }
        });
    if(ins != this->end())
        return ins;
    else
        return this->end();
}

std::unordered_map<std::string, shape> module::get_parameter_shapes() const
{
    std::unordered_map<std::string, shape> result;
    for(auto&& ins : impl->instructions)
    {
        if(ins.name() == "@param")
        {
            auto&& name  = any_cast<builtin::param>(ins.get_operator()).parameter;
            result[name] = ins.get_shape();
        }
    }
    return result;
}

bool module::has_instruction(instruction_ref ins) const
{
    return std::find_if(
               impl->instructions.begin(), impl->instructions.end(), [&](const instruction& x) {
                   return std::addressof(*ins) == std::addressof(x);
               }) != impl->instructions.end();
}

std::size_t module::size() const { return impl->instructions.size(); }
instruction_ref module::begin() const { return impl->instructions.begin(); }
instruction_ref module::end() const { return impl->instructions.end(); }

std::vector<shape> module::get_output_shapes() const
{
    auto last_ins = impl->instructions.back();
    if(last_ins.name() == "@return")
    {
        const auto& output_ins = last_ins.inputs();
        std::vector<shape> output_shapes;
        std::transform(output_ins.begin(),
                       output_ins.end(),
                       std::back_inserter(output_shapes),
                       [](auto& ins) { return ins->get_shape(); });

        return output_shapes;
    }
    // The else branch is to provide backward compatibility
    else
    {
        return {last_ins.get_shape()};
    }
}

instruction_ref module::validate() const
{
    return std::find_if(
        impl->instructions.begin(), impl->instructions.end(), [&](const instruction& i) {
            auto inputs      = i.inputs();
            bool check_order = std::all_of(inputs.begin(), inputs.end(), [&](auto in) {
                return contains(impl->instructions, *in);
            });

            return !i.valid(impl->instructions.begin(), check_order);
        });
}

void module::finalize(context& ctx)
{
    for(auto ins : iterator_for(*this))
    {
        ins->finalize(ctx);
        for(const auto& smod : ins->module_inputs())
        {
            smod->finalize(ctx);
        }
    }

    // Warn when an instruction is not normalized
    auto ins = std::find_if(begin(), end(), [](auto& i) { return i.need_normalization(); });
    if(ins != end())
        std::cerr << "WARNING: Instruction needs normalization, performance may be affected."
                  << std::endl;
}

void module::debug_print() const { std::cout << *this << std::endl; }

void module::debug_print(instruction_ref ins,
                         std::unordered_map<instruction_ref, std::string>& names) const
{
    if(ins == this->end())
    {
        std::cout << "End instruction" << std::endl;
        return;
    }
    if(not has_instruction(ins))
    {
        std::cout << "Instruction not part of module" << std::endl;
        return;
    }
    std::stringstream ss;
    names = this->print(
        [&](auto x, auto ins_names) {
            if(x == ins)
            {
                instruction::print(std::cout, x, ins_names);
                std::cout << std::endl;
            }
        },
        names);
}

void module::debug_print(instruction_ref ins) const
{
    std::unordered_map<instruction_ref, std::string> names;
    this->debug_print(ins, names);
}

void module::debug_print(const std::vector<instruction_ref>& inss) const
{
    for(auto ins : inss)
        this->debug_print(ins);
    std::cout << std::endl;
}

std::unordered_map<instruction_ref, std::string> module::print(
    const std::function<void(instruction_ref,
                             const std::unordered_map<instruction_ref, std::string>&)>& print_func,
    std::unordered_map<instruction_ref, std::string> names) const
{
    int count = 0;
    for(auto ins : iterator_for(*this))
    {
        std::string var_name;
        if(ins->name() == "@param")
        {
            var_name = any_cast<builtin::param>(ins->get_operator()).parameter;
        }
        else
        {
            var_name = this->name();
            var_name.append((this->name().empty() ? "@" : ":@"));
            var_name.append(std::to_string(count));
            count++;
        }
        names.emplace(ins, var_name);

        print_func(ins, names);
    }
    return names;
}

void module::print(const std::function<
                   void(instruction_ref, const std::unordered_map<instruction_ref, std::string>&)>&
                       print_func) const
{
    this->print(print_func, {});
}

static std::string enclose_name(const std::string& name)
{
    return '"' + replace_string(name, "\"", "\\\"") + '"';
}

void module::print_graph(std::ostream& os, bool brief) const
{
    os << "digraph {" << std::endl;
    os << "\trankdir=LR;" << std::endl;
    this->print([&](auto ins, auto ins_names) {
        std::string label;
        if(brief)
            label = ins->name();
        else
            label = to_string(ins->get_operator());
        os << "\t" << enclose_name(ins_names.at(ins)) << "[label=" << enclose_name(label) << "]";
        os << ";" << std::endl;
        if(!ins->inputs().empty())
        {
            for(auto&& arg : ins->inputs())
            {
                os << "\t" << enclose_name(ins_names.at(arg)) << " -> "
                   << enclose_name(ins_names.at(ins));
                if(not brief)
                    os << "[label=" << enclose_name(to_string(ins->get_shape())) << "]";
                os << ";" << std::endl;
            }
        }
    });
    os << "}" << std::endl;
}

static std::string cpp_var_name(const std::string& name)
{
    return "m" + replace_string(name, "@", "x");
}

static std::string cpp_op_var(const std::string& name, instruction_ref ins)
{
    return replace_string(name, "@", ins->name());
}

static void print_op_attributes(std::ostream& os, const std::string& name, const operation& op)
{
    std::string x = to_string(op);
    if(contains(x, "["))
    {
        auto start                 = x.find('[');
        auto end                   = x.find(']');
        std::string attribute_text = x.substr(start + 1, end - start - 1);
        std::vector<std::string> attributes;
        for(auto&& attribute : split_string(attribute_text, ','))
        {
            if(contains(attribute, '='))
                attributes.push_back(attribute);
            else
                attributes.back() += "," + attribute;
        }
        for(auto&& attribute : attributes)
        {
            auto p     = split_string(attribute, '=');
            auto key   = p.front();
            auto value = p.back();
            if(contains({"bn_mode", "padding_mode"}, key))
                continue;
            if(key == "mode")
                value = enclose_name(trim(value));
            os << name << "." << key << " = " << value << ";" << std::endl;
        }
    }
}

static void print_cpp_shape(std::ostream& os, const migraphx::shape& s)
{
    os << "migraphx::shape{migraphx::shape::" << s.type_string();
    os << ", {" << to_string_range(s.lens()) << "}";
    if(not s.standard())
        os << ", {" << to_string_range(s.strides()) << "}";
    os << "}";
}

std::unordered_map<instruction_ref, std::string>
module::print_cpp(std::ostream& os, std::unordered_map<instruction_ref, std::string> names) const
{
    os << "migraphx::module p;" << std::endl;
    // cppcheck-suppress variableScope
    unsigned long seed = 0;
    names              = this->print(
        [&](auto ins, auto ins_names) {
            auto op = cpp_op_var(ins_names.at(ins), ins);
            if(ins->name().front() != '@')
            {
                os << "migraphx::op::" << ins->name() << " " << op << ";" << std::endl;
                print_op_attributes(os, op, ins->get_operator());
            }
            os << "auto " << cpp_var_name(ins_names.at(ins)) << " = ";
            if(ins->name() == "@literal")
            {
                os << "p.add_literal(";
                bool use_abs = false;
                ins->get_literal().visit([&](auto v) {
                    use_abs = std::none_of(v.begin(), v.end(), [](auto x) { return x < 0; });
                });
                if(use_abs)
                    os << "migraphx::abs(";
                os << "migraphx::generate_literal(";
                print_cpp_shape(os, ins->get_shape());
                os << ", " << seed << ")";
                if(use_abs)
                    os << ")";
                os << ");" << std::endl;
                seed++;
            }
            else if(ins->name() == "@param")
            {
                std::string name = any_cast<builtin::param>(ins->get_operator()).parameter;
                os << "p.add_parameter(" << enclose_name(name) << ",";
                print_cpp_shape(os, ins->get_shape());
                os << ");" << std::endl;
            }
            else
            {
                os << "p.add_instruction(" << op;
                for(auto input : ins->inputs())
                {
                    os << ", " << cpp_var_name(ins_names.at(input));
                }
                os << ");" << std::endl;
            }
        },
        names);

    return names;
}

void module::print_cpp(std::ostream& os) const { this->print_cpp(os, {}); }

void module::annotate(std::ostream& os, std::function<void(instruction_ref)> a) const
{
    this->print([&](auto ins, auto ins_names) {
        instruction::print(os, ins, ins_names);
        a(ins);
        os << std::endl;
    });
}

std::vector<module_ref> module::get_sub_modules() const
{
    std::vector<module_ref> vec_modules;
    for(auto ins : iterator_for(*this))
    {
        const auto& mod_args = ins->module_inputs();
        vec_modules.insert(vec_modules.end(), mod_args.begin(), mod_args.end());
        for(const auto& smod : mod_args)
        {
            auto sub_mods = smod->get_sub_modules();
            vec_modules.insert(vec_modules.end(), sub_mods.begin(), sub_mods.end());
        }
    }

    return vec_modules;
}

module& module::sort()
{
    fix([&](auto self, auto ins) {
        this->move_instruction(ins, this->begin());
        for(auto child : ins->inputs())
        {
            if(!contains(this->impl->instructions, child))
            {
                continue;
            }
            self(child);
        }
    })(std::prev(this->end()));
    assert(this->validate() == this->end());
    return *this;
}

void module::calc_implicit_deps(const module& smod,
                                const module& pmod,
                                instruction_ref ins,
                                ins_dep_map& deps) const
{
    const auto& ins_inputs = ins->inputs();
    for(auto ii : iterator_for(smod))
    {
        const auto& ii_inputs = ii->inputs();
        for(auto iii : ii_inputs)
        {
            if(pmod.has_instruction(iii))
            {
                if(not contains(ins_inputs, iii))
                    deps[ins].insert(iii);
            }
        }

        const auto& mod_args = ii->module_inputs();
        if(not mod_args.empty())
        {
            for(const auto* ssmod : mod_args)
            {
                calc_implicit_deps(*ssmod, pmod, ins, deps);
            }
        }
    }
}

ins_dep_map module::calc_implicit_deps() const
{
    ins_dep_map mod_implicit_deps;
    for(auto ins : iterator_for(*this))
    {
        const auto& mod_args = ins->module_inputs();
        if(mod_args.empty())
        {
            continue;
        }

        for(const auto* mod : mod_args)
        {
            calc_implicit_deps(*mod, *this, ins, mod_implicit_deps);
        }
    }

    return mod_implicit_deps;
}

bool operator==(const module& x, const module& y) { return to_string(x) == to_string(y); }

std::ostream& operator<<(std::ostream& os, const module& m)
{
    m.print([&](auto ins, auto ins_names) {
        instruction::print(os, ins, ins_names);
        os << std::endl;
    });

    return os;
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
