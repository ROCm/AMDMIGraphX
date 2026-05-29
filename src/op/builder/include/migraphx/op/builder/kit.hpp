#ifndef MIGRAPHX_GUARD_BUILDER_KIT_HPP
#define MIGRAPHX_GUARD_BUILDER_KIT_HPP

#include <migraphx/config.hpp>
#include <migraphx/op/builder/op_builder.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/common.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {
namespace builder {

struct register_kit_action
{
    template <class T>
    static void apply()
    {
        T{}.apply();
    }
};

template <class T>
struct kit : auto_register<register_kit_action, T>
{
    void apply() const {}

    std::string derived_prefix() const { return static_cast<const T&>(*this).prefix(); }

    op_builder_if from_op(const std::string& op_name) const
    {
        return op_builder_if{[=](module& m,
                                 instruction_ref ins,
                                 const std::vector<instruction_ref>& args,
                                 const std::vector<module_ref>& module_args,
                                 const value& options) -> std::vector<instruction_ref> {
                                 auto opd = make_op(op_name, options);
                                 return {m.insert_instruction(ins, opd, args, module_args)};
                             },
                             [=] { return make_op(op_name).to_value(); }};
    }

    op_builder_if from_builder(const std::string& op_builder) const
    {
        return get_op_builder_if(op_builder);
    }

    op_builder_if with_common(op_builder_if obi, common_options coptions = {}) const
    {
        return op_builder_if{[=](module& m,
                                 instruction_ref ins,
                                 const std::vector<instruction_ref>& args,
                                 const std::vector<module_ref>& module_args,
                                 const value& options) {
                                 auto cargs = insert_common_args(m, ins, args, coptions);
                                 return obi.bld_func(m, ins, cargs, module_args, options);
                             },
                             [=] { return obi.to_val_func(); }};
    }

    void ops(const std::initializer_list<std::string>& op_names) const
    {
        for(const auto& name : op_names)
        {
            register_builder(derived_prefix() + name, from_op(name));
        }
    }

    void common_ops(const std::initializer_list<std::string>& op_names,
                    common_options coptions = {}) const
    {
        for(const auto& name : op_names)
        {
            register_builder(derived_prefix() + name, with_common(from_op(name), coptions));
        }
    }

    void builders(const std::initializer_list<std::string>& builder_names) const
    {
        for(const auto& name : builder_names)
        {
            register_builder(derived_prefix() + name, from_builder(name));
        }
    }
};

} // namespace builder
} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
#endif // MIGRAPHX_GUARD_BUILDER_KIT_HPP
