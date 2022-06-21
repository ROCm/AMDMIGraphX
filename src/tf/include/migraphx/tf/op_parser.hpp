#ifndef MIGRAPHX_GUARD_AMDMIGRAPHX_TF_REGISTER_OP_PARSER_HPP
#define MIGRAPHX_GUARD_AMDMIGRAPHX_TF_REGISTER_OP_PARSER_HPP

#include <migraphx/config.hpp>
#include <migraphx/auto_register.hpp>
#include <migraphx/tf/tf_parser.hpp>
#include <cstring>
#include <vector>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace tf {

struct op_desc
{
    std::string tf_name = "";
    std::string op_name = "";
};

void register_op_parser(const std::string& name, tf_parser::op_func f);
tf_parser::op_func get_op_parser(const std::string& name);
std::vector<std::string> get_op_parsers();

inline std::vector<instruction_ref> implicit_multi_op(std::vector<instruction_ref> inss)
{
    return inss;
}

inline std::vector<instruction_ref> implicit_multi_op(instruction_ref ins) { return {ins}; }

template <class T>
void register_op_parser()
{
    T parser;
    for(auto&& opd : parser.operators())
        register_op_parser(opd.tf_name,
                           [opd, parser](auto&&... xs) { return parser.base_parse(opd, xs...); });
}

struct register_op_parser_action
{
    template <class T>
    static void apply()
    {
        register_op_parser<T>();
    }
};

template <class Derived>
struct op_parser : auto_register<register_op_parser_action, Derived>
{
    bool transpose() const { return false; }
    std::vector<instruction_ref> base_parse(const op_desc& opd,
                                            const tf_parser& parser,
                                            tf_parser::node_info info,
                                            const std::vector<instruction_ref>& args) const
    {
        std::vector<instruction_ref> result;
        auto& self = static_cast<const Derived&>(*this);
        if(self.transpose())
        {
            result = implicit_multi_op(self.parse(opd, parser, info, parser.to_nchw(args)));
            std::transform(result.begin(), result.end(), result.begin(), [&](auto ins) {
                return parser.to_nhwc(ins);
            });
        }
        else
        {
            result = implicit_multi_op(self.parse(opd, parser, info, args));
        }
        return result;
    }
};

} // namespace tf
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
