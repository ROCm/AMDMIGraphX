#ifndef MIGRAPHX_GUARD_AMDMIGRAPHX_ONNX_PARSER_HPP
#define MIGRAPHX_GUARD_AMDMIGRAPHX_ONNX_PARSER_HPP

#include <migraphx/config.hpp>
#include <migraphx/program.hpp>
#include <google/protobuf/text_format.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <onnx.pb.h>
#include <unordered_map>
#include <functional>
#include <utility>
#include <vector>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {

namespace onnx = onnx_for_migraphx;

struct onnx_parser
{
    std::string filename;
    std::string path    = ".";
    using attribute_map = std::unordered_map<std::string, onnx::AttributeProto>;
    struct node_info
    {
        attribute_map attributes{};
        std::size_t num_outputs = 1;
        std::string name        = "";
        module* mod             = nullptr;
        instruction_ref make_contiguous(instruction_ref ins) const;
        instruction_ref add_bias(const std::vector<instruction_ref>& args,
                                 instruction_ref curr_ins,
                                 uint64_t axis) const;
        instruction_ref add_broadcastable_binary_op(const std::string& op_name,
                                                    instruction_ref arg0,
                                                    instruction_ref arg1) const;
        instruction_ref add_instruction(const operation& op,
                                        const std::vector<instruction_ref>& args) const;

        instruction_ref add_instruction(const operation& op,
                                        const std::vector<instruction_ref>& args,
                                        const std::vector<module_ref>& mods) const;

        template <class... Ts>
        instruction_ref add_instruction(const operation& op, Ts... xs) const
        {
            return add_instruction(op, {xs...});
        }
        instruction_ref add_literal(literal l) const;
        template <class... Ts>
        instruction_ref add_literal(Ts&&... xs) const
        {
            return add_literal(literal{std::forward<Ts>(xs)...});
        }
    };
    using node_map = std::unordered_map<std::string, onnx::NodeProto>;
    using op_func  = std::function<std::vector<instruction_ref>(
        onnx_parser&, const node_info&, std::vector<instruction_ref>)>;
    node_map nodes;
    std::unordered_map<std::string, instruction_ref> instructions;
    program prog                  = program();
    std::size_t default_dim_value = 1;
    std::unordered_map<std::string, std::vector<std::size_t>> map_input_dims;
    bool skip_unknown_operators = false;
    int64_t max_loop_iterations = 10;
    int64_t opset_version       = 13;

    std::unordered_map<std::string, op_func> ops;

    onnx_parser();
    operation load(const std::string& name, const node_info& info) const;

    void parse_undefined(module* mod, const std::string& name);

    static int64_t get_opset_version(const onnx::ModelProto& model);

    void parse_from(std::istream& is, std::string name = "");
    void parse_from(const void* data, std::size_t size);
    void parse_graph(module* mod, const onnx::GraphProto& graph);
    literal parse_value(const onnx::AttributeProto& attr) const;
    literal parse_tensor(const onnx::TensorProto& t) const;
    shape parse_type(const onnx::TypeProto& t, const std::vector<std::size_t>& input_dims) const;
};

shape::type_t get_type(int dtype);

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
