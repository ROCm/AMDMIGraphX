#ifndef MIGRAPHX_GUARD_AMDMIGRAPHX_TF_PARSER_HPP
#define MIGRAPHX_GUARD_AMDMIGRAPHX_TF_PARSER_HPP

#include <migraphx/config.hpp>
#include <migraphx/program.hpp>
#include <google/protobuf/text_format.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <graph.pb.h>
#include <unordered_map>
#include <functional>
#include <utility>
#include <vector>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace tf {

// namespace tf = tf_for_migraphx;

struct tf_parser
{
    std::string filename;
    std::string path    = ".";
    using attribute_map = std::unordered_map<std::string, tensorflow::AttrValue>;
    struct node_info
    {
        attribute_map attributes{};
        std::string name = "";
        module* mm       = nullptr;

        instruction_ref make_contiguous(instruction_ref ins) const;

        instruction_ref add_broadcastable_binary_op(const std::string& op_name,
                                                    instruction_ref arg0,
                                                    instruction_ref arg1) const;
        instruction_ref add_instruction(const operation& op,
                                        const std::vector<instruction_ref>& args) const;

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

    using node_map = std::map<std::string, tensorflow::NodeDef>;
    using op_func  = std::function<std::vector<instruction_ref>(
        const tf_parser&, const node_info&, std::vector<instruction_ref>)>;
    node_map nodes;
    std::vector<tensorflow::NodeDef> input_nodes;
    std::vector<std::string> output_node_names;
    std::unordered_map<std::string, instruction_ref> instructions;
    program prog                  = program();
    module* mm                    = prog.get_main_module();
    bool is_nhwc                  = true;
    unsigned int batch_size       = 1;
    std::size_t default_dim_value = 1;
    std::unordered_map<std::string, std::vector<std::size_t>> map_input_dims;

    std::unordered_map<std::string, op_func> ops;

    tf_parser();
    operation load(const std::string& name, const node_info& info) const;
    bool should_transpose(instruction_ref ins) const;
    instruction_ref to_nhwc(instruction_ref ins) const;
    instruction_ref to_nchw(instruction_ref ins) const;
    instruction_ref to_kcxy(instruction_ref ins) const;
    std::vector<instruction_ref> to_nchw(const std::vector<instruction_ref>& args) const;
    std::vector<instruction_ref> to_nhwc(const std::vector<instruction_ref>& args) const;
    int64_t parse_axis(int64_t dim, size_t num_dims) const;
    // tf stores certain attributes such as strides, dilations, as a 4D input.
    // The first and last dims are equal to 1, and the relevant data is in dims 2 and 3.
    // This helper function reorders the data to store for the respective operator member variables.
    template <class T>
    void reorder_data(std::vector<T>& prev_data) const
    {
        std::vector<T> new_data(prev_data.size());
        for(size_t i = 0; i < new_data.size(); i++)
        {
            auto new_idx         = parse_axis(i, new_data.size());
            new_data.at(new_idx) = prev_data.at(i);
        }
        prev_data = new_data;
    }

    void parse_undefined(module* mm, const std::string& name);
    void parse_from(std::istream& is);
    void parse_from(const void* data, std::size_t size);
    void parse_graph(const tensorflow::GraphDef& graph);
    void parse_node(const std::string& name);
    literal parse_tensor(const tensorflow::TensorProto& t) const;
    shape::type_t parse_type(tensorflow::DataType t) const;
};

std::vector<int64_t> get_axes_from_mask(size_t num_axes, uint32_t mask);

} // namespace tf
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
