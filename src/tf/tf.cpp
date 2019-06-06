#include <google/protobuf/text_format.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <graph.pb.h>
#include <iostream>
#include <fstream>
#include <unordered_map>
#include <unordered_set>
#include <functional>
#include <array>
#include <utility>
#include <vector>

#include <migraphx/fallthrough.hpp>
#include <migraphx/program.hpp>
#include <migraphx/operators.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/config.hpp>
#include <migraphx/tf.hpp>
#include <migraphx/pad_calc.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct tf_parser
{
    using attribute_map = std::unordered_map<std::string, tensorflow::AttrValue>;
    using node_map      = std::map<std::string, tensorflow::NodeDef>;
    // using input_node_map = std::unordered_map<std::string, std::unordered_set<std::string>>;
    using op_func = std::function<instruction_ref(attribute_map, std::vector<instruction_ref>)>;

    node_map nodes;
    std::vector<tensorflow::NodeDef> input_nodes;
    std::unordered_map<std::string, instruction_ref> instructions;
    program prog = program();
    bool is_nhwc = true;

    std::unordered_map<std::string, op_func> ops;

    std::vector<size_t> parse_axes(const attribute_map& attributes, const std::string& s) const
    {
        auto attrs = attributes.at(s).list().i();
        std::vector<size_t> axes;
        copy(attrs.begin(), attrs.end(), std::back_inserter(axes));
        if(is_nhwc)
        {
            std::transform(axes.begin(), axes.end(), axes.begin(), [&](size_t axis) {
                return parse_axis(axis);
            });
        }
        return axes;
    }

    template <class T>
    std::vector<T> parse_axes(std::vector<T> axes) const
    {
        if(is_nhwc)
        {
            std::vector<T> new_axes;
            std::transform(axes.begin(),
                           axes.end(),
                           std::back_inserter(new_axes),
                           [&](size_t axis) { return parse_axis(axis); });
            return new_axes;
        }
        return axes;
    }

    // tf stores certain attributes such as strides, dilations, as a 4D input.
    // The first and last dims are equal to 1, and the relevant data is in dims 2 and 3.
    // This helper function reorders the data to store for the respective operator member variables.
    template <class T>
    void reorder_data(std::vector<T>& prev_data) const
    {
        std::vector<T> new_data(prev_data.size());
        for(size_t i = 0; i < new_data.size(); i++)
        {
            auto new_idx         = parse_axis(i);
            new_data.at(new_idx) = prev_data.at(i);
        }
        prev_data = new_data;
    }

    template <class T>
    T parse_axis(const T& dim) const
    {
        T new_dim = dim;
        if(is_nhwc)
        {
            switch(dim)
            {
            case 0: new_dim = 0; break;
            case 1: new_dim = 2; break;
            case 2: new_dim = 3; break;
            case 3: new_dim = 1; break;
            default: break;
            }
        }
        return new_dim;
    }

    std::vector<int64_t> get_axes(size_t num_axes) const
    {
        std::vector<int64_t> axes(num_axes);
        std::iota(axes.begin(), axes.end(), 0);
        return axes;
    }

    tf_parser()
    {
        add_generic_op("Identity", op::identity{});
        add_generic_op("Relu", op::relu{});
        add_generic_op("Relu6", op::clip{6.0, 0.0});

        add_binary_op("Add", op::add{});
        add_binary_op("Mul", op::mul{});

        add_mem_op("AvgPool", &tf_parser::parse_pooling);
        add_mem_op("BiasAdd", &tf_parser::parse_biasadd);
        add_mem_op("ConcatV2", &tf_parser::parse_concat);
        add_mem_op("Const", &tf_parser::parse_constant);
        add_mem_op("Conv2D", &tf_parser::parse_conv);
        add_mem_op("DepthwiseConv2dNative", &tf_parser::parse_depthwiseconv);
        add_mem_op("FusedBatchNorm", &tf_parser::parse_batchnorm);
        add_mem_op("MatMul", &tf_parser::parse_matmul);
        add_mem_op("MaxPool", &tf_parser::parse_pooling);
        add_mem_op("Mean", &tf_parser::parse_mean);
        add_mem_op("Pack", &tf_parser::parse_pack);
        add_mem_op("Pad", &tf_parser::parse_pad);
        add_mem_op("Reshape", &tf_parser::parse_reshape);
        add_mem_op("Softmax", &tf_parser::parse_softmax);
        add_mem_op("Squeeze", &tf_parser::parse_squeeze);
        add_mem_op("StridedSlice", &tf_parser::parse_stridedslice);
    }

    template <class F>
    void add_op(std::string name, F f)
    {
        ops.emplace(name, f);
    }

    // Multi output op
    template <class F>
    void add_multi_op(std::string name, F f)
    {
        ops.emplace(name, f);
    }

    template <class F>
    void add_mem_op(std::string name, F f)
    {
        add_op(name, [=](auto&&... xs) {
            return std::mem_fn(f)(*this, name, std::forward<decltype(xs)>(xs)...);
        });
    }

    template <class T>
    void add_binary_op(std::string name, T x)
    {
        add_op(name, [this, x](const attribute_map& attributes, std::vector<instruction_ref> args) {
            if(args.size() != 2)
                MIGRAPHX_THROW("binary operators should have 2 operands");
            auto l0 = args[1];
            if(contains(attributes, "data_format"))
            {
                if(is_nhwc)
                {
                    l0 = prog.add_instruction(op::transpose{{0, 3, 1, 2}}, args[1]);
                }
            }
            return add_broadcastable_binary_op(args[0], l0, x);
        });
    }

    template <class T>
    instruction_ref add_broadcastable_binary_op(instruction_ref arg0, instruction_ref arg1, T x)
    {
        if(arg0->get_shape().lens() != arg1->get_shape().lens())
        {
            // Example:
            // s0 = (3,2,4,5) and s1 = (2,1,1)
            //
            // In this case we need to broadcast (:,1,1) portion of
            // s1 plus broadcast the 1st dimension of s1
            // giving output_lens = (3,2,4,5)
            //
            // Another example:
            // s0 = (3,2,1,5) and s1 = (2,7,5)
            // In this case we need to broadcast the (:,:,1:,:) axis
            // of s0 plus the 1st dimension of s1 giving
            // output_lens = (3,2,7,5)
            //
            // Get lengths for both arguments
            const std::vector<size_t>* s0 = &arg0->get_shape().lens();
            const std::vector<size_t>* s1 = &arg1->get_shape().lens();

            // Make sure s0 is the smaller size
            if(s0->size() > s1->size())
                std::swap(s0, s1);

            std::vector<size_t> output_lens(*s1);
            auto offset = s1->size() - s0->size();
            std::transform(s0->begin(),
                           s0->end(),
                           s1->begin() + offset,
                           output_lens.begin() + offset,
                           [](auto a, auto b) { return std::max(a, b); });

            auto l0 = prog.add_instruction(op::multibroadcast{output_lens}, arg0);
            auto l1 = prog.add_instruction(op::multibroadcast{output_lens}, arg1);
            return prog.add_instruction(x, l0, l1);
        }
        else
        {
            return prog.add_instruction(x, {arg0, arg1});
        }
    }

    template <class T>
    void add_generic_op(std::string name, T x)
    {
        add_op(name, [this, x](const attribute_map&, std::vector<instruction_ref> args) {
            return prog.add_instruction(x, args);
        });
    }

    instruction_ref
    parse_batchnorm(const std::string&, attribute_map attributes, std::vector<instruction_ref> args)
    {
        float epsilon                                     = 1e-5f;
        float momentum                                    = 0.9f;
        op::batch_norm_inference::bn_infer_mode_t bn_mode = op::batch_norm_inference::spatial;
        if(contains(attributes, "epsilon"))
        {
            epsilon = attributes.at("epsilon").f();
        }
        op::batch_norm_inference op{epsilon, momentum, bn_mode};
        return prog.add_instruction(op, std::move(args));
    }

    instruction_ref
    parse_biasadd(const std::string&, const attribute_map&, std::vector<instruction_ref> args)
    {
        uint64_t axis = 1; // assume output of previous layer is in NCHW (broadcast on channel)
        auto l0 = prog.add_instruction(op::broadcast{axis, args[0]->get_shape().lens()}, args[1]);
        return prog.add_instruction(op::add{}, args[0], l0);
    }

    instruction_ref
    parse_concat(const std::string&, attribute_map attributes, std::vector<instruction_ref> args)
    {
        // get index for axis within args
        size_t axis_idx = attributes.at("N").i();
        size_t axis     = parse_axis(args[axis_idx]->eval().at<int64_t>());
        op::concat op{axis};
        // return only first N arguments (assuming last index is the axis value)
        return prog.add_instruction(
            op, std::vector<instruction_ref>(args.begin(), args.begin() + args.size() - 1));
    }

    instruction_ref parse_constant(const std::string&,
                                   attribute_map attributes,
                                   const std::vector<instruction_ref>&)
    {
        literal v       = parse_tensor(attributes.at("value").tensor());
        auto l0         = prog.add_literal(v);
        size_t num_axes = l0->get_shape().lens().size();
        if(num_axes >= 4)
        {
            std::vector<int64_t> transpose_axes = get_axes(num_axes);
            reorder_data(transpose_axes);
            l0 = prog.add_instruction(op::transpose{transpose_axes}, l0);
        }
        return l0;
    }

    instruction_ref
    parse_conv(const std::string&, attribute_map attributes, std::vector<instruction_ref> args)
    {
        op::convolution op;
        if(contains(attributes, "strides"))
        {
            std::vector<size_t> stride;
            copy(attributes.at("strides").list().i(), std::back_inserter(stride));
            reorder_data(stride);
            if(stride.size() != 4)
            {
                MIGRAPHX_THROW("strides should have 4 values");
            }
            op.stride[0] = stride[2];
            op.stride[1] = stride[3];
        }
        if(contains(attributes, "dilations"))
        {
            std::vector<size_t> dilation;
            copy(attributes.at("dilations").list().i(), std::back_inserter(dilation));
            reorder_data(dilation);
            if(dilation.size() != 4)
            {
                MIGRAPHX_THROW("dilation should have 4 values");
            }
            op.dilation[0] = dilation[2];
            op.dilation[1] = dilation[3];
        }
        auto weights = args[1];
        // check if weights are from a constant

        if(weights->name() != "@param")
        {
            if(is_nhwc)
            {
                weights = prog.add_instruction(op::transpose{{1, 3, 0, 2}}, args[1]);
            }
            else
            {
                weights = prog.add_instruction(op::transpose{{3, 2, 0, 1}}, args[1]);
            }
        }

        if(contains(attributes, "padding"))
        {
            const std::string& pad_mode = attributes.at("padding").s();
            if(pad_mode.find("SAME") != std::string::npos)
            {
                op.padding_mode = op::padding_mode_t::same;
                std::vector<size_t> weight_dims = weights->get_shape().lens();
                size_t weight_h                 = weight_dims[2];
                size_t weight_w                 = weight_dims[3];
                op.padding[0]   = calculate_padding(weight_h, op.dilation[0]);
                op.padding[1]   = calculate_padding(weight_w, op.dilation[1]);
            }
            else if(pad_mode.find("VALID") != std::string::npos)
            {
                op.padding_mode = op::padding_mode_t::valid;
            }
            else if(pad_mode.find("EXPLICIT") != std::string::npos)
            {
                std::vector<size_t> padding;
                copy(attributes.at("explicit_paddings").list().i(), std::back_inserter(padding));
                if(padding.size() != 4)
                {
                    MIGRAPHX_THROW("padding should have 4 values");
                }
                if(padding[0] != padding[2] || padding[1] != padding[3])
                {
                    MIGRAPHX_THROW("migraphx does not support asymetric padding");
                }
                op.padding[0] = padding[0];
                op.padding[1] = padding[1];
            }
        }

        return prog.add_instruction(op, {args[0], weights});
    }

    instruction_ref parse_depthwiseconv(const std::string&,
                                        attribute_map attributes,
                                        std::vector<instruction_ref> args)
    {
        op::convolution op;
        size_t num_channels = args[0]->get_shape().lens()[1];
        op.group            = num_channels;

        if(contains(attributes, "strides"))
        {
            std::vector<size_t> stride;
            copy(attributes.at("strides").list().i(), std::back_inserter(stride));
            reorder_data(stride);
            if(stride.size() != 4)
            {
                MIGRAPHX_THROW("strides should have 4 values");
            }
            op.stride[0] = stride[2];
            op.stride[1] = stride[3];
        }
        if(contains(attributes, "dilations"))
        {
            std::vector<size_t> dilation;
            copy(attributes.at("dilations").list().i(), std::back_inserter(dilation));
            reorder_data(dilation);
            if(dilation.size() != 4)
            {
                MIGRAPHX_THROW("dilation should have 4 values");
            }
            op.dilation[0] = dilation[2];
            op.dilation[1] = dilation[3];
        }

        auto weights = args[1];
        // check if weights are from a constant
        if(weights->name() != "@param")
        {
            if(is_nhwc)
            {
                weights = prog.add_instruction(op::transpose{{1, 3, 0, 2}}, args[1]);
            }
            else
            {
                weights = prog.add_instruction(op::transpose{{3, 2, 0, 1}}, args[1]);
            }
        }

        if(contains(attributes, "padding"))
        {
            const std::string& pad_mode     = attributes.at("padding").s();
            std::vector<size_t> weight_dims = weights->get_shape().lens();
            size_t weight_h                 = weight_dims[2];
            size_t weight_w                 = weight_dims[3];
            if(pad_mode.find("SAME") != std::string::npos)
            {
                op.padding_mode = op::padding_mode_t::same;
                op.padding[0]   = calculate_padding(weight_h, op.dilation[0]);
                op.padding[1]   = calculate_padding(weight_w, op.dilation[1]);
            }
            else if(pad_mode.find("VALID") != std::string::npos)
            {
                op.padding_mode = op::padding_mode_t::valid;
            }
        }

        std::vector<int64_t> new_weights_shape;
        copy(weights->get_shape().lens(), std::back_inserter(new_weights_shape));

        // weight format is (out_channels, in_channels, h, w), but in depthwise_conv,
        // out_channels is equal to the multiplier. Adjust by inserting a reshape and
        // setting in_channels to 1
        int64_t multiplier   = new_weights_shape[0];
        int64_t out_channels = num_channels * multiplier;
        new_weights_shape[0] = out_channels;
        new_weights_shape[1] = 1;
        // Make sure weights are contiguous before doing reshape
        auto cweights    = prog.add_instruction(op::contiguous{}, weights);
        auto new_weights = prog.add_instruction(op::reshape{new_weights_shape}, cweights);

        return prog.add_instruction(op, {args[0], new_weights});
    }

    instruction_ref
    parse_matmul(const std::string&, attribute_map attributes, std::vector<instruction_ref> args)
    {
        bool transa = false;
        bool transb = false;

        if(contains(attributes, "transpose_a"))
        {
            transa = attributes.at("transpose_a").b();
        }
        if(contains(attributes, "transpose_b"))
        {
            transb = attributes.at("transpose_a").b();
        }

        std::vector<int64_t> perm(args[0]->get_shape().lens().size());
        std::iota(perm.begin(), perm.end(), int64_t{0});
        // swap the last two elements
        std::iter_swap(perm.end() - 1, perm.end() - 2);

        auto l1 = (transa) ? prog.add_instruction(op::transpose{perm}, args[0]) : args[0];
        auto l2 = (transb) ? prog.add_instruction(op::transpose{perm}, args[1]) : args[1];

        return prog.add_instruction(op::dot{}, l1, l2);
    }

    instruction_ref
    parse_mean(const std::string&, attribute_map attributes, std::vector<instruction_ref> args)
    {
        auto axes      = parse_axes(args[1]->eval().get<int32_t>().to_vector());
        bool keep_dims = attributes.at("keep_dims").b();
        std::vector<int32_t> hw_axes{2, 3};
        // check if conditions for GlobalAvgPool are met
        auto lens = args[0]->get_shape().lens();
        if(axes == hw_axes and lens.size() == 4)
        {
            op::pooling op{"average"};
            op.lengths[0] = lens[2];
            op.lengths[1] = lens[3];
            auto l0       = prog.add_instruction(op, args.front());
            if(keep_dims)
                return l0;
            return prog.add_instruction(
                op::squeeze{std::vector<int64_t>(hw_axes.begin(), hw_axes.end())}, l0);
        }
        MIGRAPHX_THROW("MIGraphX does not support mean outside of GlobalAvgPool transformation");
    }

    instruction_ref parse_pack(const std::string&,
                               const attribute_map& attributes,
                               std::vector<instruction_ref> args)
    {
        // reinterpret as unsqueeze with concat
        std::vector<instruction_ref> unsqueezed_args;
        int64_t axis = 0;
        if(contains(attributes, "axis"))
            axis = attributes.at("axis").i();
        size_t input_size = args.front()->get_shape().lens().size();
        if(axis > input_size)
        {
            MIGRAPHX_THROW("TF_PARSER: axis value of " + to_string(axis) +
                           " must be smaller than input size " + to_string(input_size));
        }
        // check if input arg needs axis to be converted to NCHW
        if(input_size >= 4)
            axis = parse_axis(axis);

        std::transform(
            args.begin(),
            args.end(),
            std::back_inserter(unsqueezed_args),
            [&](instruction_ref arg) { return prog.add_instruction(op::unsqueeze{{axis}}, arg); });
        return prog.add_instruction(op::concat{static_cast<size_t>(axis)}, unsqueezed_args);
    }

    instruction_ref
    parse_pad(const std::string&, const attribute_map&, std::vector<instruction_ref> args)
    {
        size_t ndims = args.front()->get_shape().lens().size();

        // in tf, the paddings are arranged as a 2d shape (ndims, 2),
        // the last dim contains the left padding and right padding respectively
        std::vector<std::pair<int32_t, int32_t>> pad_per_dim(ndims);
        auto tf_padding = args[1]->eval().get<int32_t>().to_vector();
        for(size_t i = 0; i < 2 * ndims; i += 2)
        {
            pad_per_dim[i / 2].first  = tf_padding[i];
            pad_per_dim[i / 2].second = tf_padding[i + 1];
        }
        reorder_data(pad_per_dim);

        op::pad op;
        std::vector<int64_t> pads(ndims * 2);
        for(size_t i = 0; i < ndims; i++)
        {
            pads[i]         = pad_per_dim[i].first;
            pads[i + ndims] = pad_per_dim[i].second;
        }
        op.pads = pads;
        return prog.add_instruction(op, args.front());
    }

    instruction_ref parse_pooling(const std::string& name,
                                  attribute_map attributes,
                                  std::vector<instruction_ref> args)
    {
        op::pooling op{starts_with(name, "Max") ? "max" : "average"};

        if(contains(attributes, "strides"))
        {
            std::vector<size_t> stride;
            copy(attributes.at("strides").list().i(), std::back_inserter(stride));
            reorder_data(stride);
            if(stride.size() != 4)
            {
                MIGRAPHX_THROW("strides should have 4 values");
            }
            op.stride[0] = stride[2];
            op.stride[1] = stride[3];
        }
        if(contains(attributes, "ksize"))
        {
            std::vector<size_t> ksize;
            copy(attributes.at("ksize").list().i(), std::back_inserter(ksize));
            reorder_data(ksize);
            if(ksize.size() != 4)
            {
                MIGRAPHX_THROW("ksize should have 4 values");
            }
            op.lengths[0] = ksize[2];
            op.lengths[1] = ksize[3];
        }
        if(contains(attributes, "padding"))
        {
            const std::string& pad_mode = attributes.at("padding").s();
            if(pad_mode.find("SAME") != std::string::npos)
            {
                op.padding_mode = op::padding_mode_t::same;
                op.padding[0]   = calculate_padding(op.lengths[0], 1);
                op.padding[1]   = calculate_padding(op.lengths[1], 1);
            }
            else if(pad_mode.find("VALID") != std::string::npos)
            {
                op.padding_mode = op::padding_mode_t::valid;
            }
        }
        return prog.add_instruction(op, args[0]);
    }

    instruction_ref
    parse_reshape(const std::string&, const attribute_map&, std::vector<instruction_ref> args)
    {
        op::reshape op;
        if(args.size() != 2)
            MIGRAPHX_THROW("reshape needs 2 arguments (input, new_shape)");
        auto s = args[1]->eval();
        s.visit([&](auto v) { copy(v, std::back_inserter(op.dims)); });
        return prog.add_instruction(op, args[0]);
    }

    void parse_from(std::istream& is)
    {
        tensorflow::GraphDef graph;
        if(graph.ParseFromIstream(&is))
        {
            this->parse_graph(graph);
        }
        else
        {
            throw std::runtime_error("Failed reading tf file");
        }
    }

    instruction_ref
    parse_softmax(const std::string&, const attribute_map&, std::vector<instruction_ref> args)
    {
        auto dims = args.front()->get_shape().lens();
        auto r =
            prog.add_instruction(op::reshape{{long(dims[0]), long(dims[1]), 1, 1}}, args.front());
        auto s = prog.add_instruction(op::softmax{}, r);
        return prog.add_instruction(op::reshape{{long(dims[0]), long(dims[1])}}, s);
    }

    instruction_ref parse_squeeze(const std::string&,
                                  const attribute_map& attributes,
                                  std::vector<instruction_ref> args)
    {
        op::squeeze op;
        auto axes = parse_axes(attributes, "squeeze_dims");
        copy(axes, std::back_inserter(op.axes));
        auto args0_dims = args[0]->get_shape().lens();
        if(op.axes.empty()) // no squeeze_dims provided, remove any dim that equals 1
        {
            for(size_t i = 0; i < args0_dims.size(); i++)
            {
                if(args0_dims.at(i) == 1)
                {
                    op.axes.push_back(i);
                }
            }
        }
        return prog.add_instruction(op, args[0]);
    }

    instruction_ref parse_stridedslice(const std::string&,
                                       const attribute_map& attributes,
                                       std::vector<instruction_ref> args)
    {
        op::slice op;
        auto starts     = args[1]->eval().get<int32_t>().to_vector();
        auto ends       = args[2]->eval().get<int32_t>().to_vector();
        size_t num_axes = args[0]->get_shape().lens().size();
        if(num_axes >= 4)
        {
            reorder_data(starts);
            reorder_data(ends);
        }

        op.starts = std::vector<int64_t>(starts.begin(), starts.end());
        op.ends   = std::vector<int64_t>(ends.begin(), ends.end());
        op.axes   = std::vector<int64_t>(num_axes);
        std::iota(op.axes.begin(), op.axes.end(), 0);
        uint32_t shrink_axis_mask = 0;
        uint32_t bitwise_compare  = 1;
        std::vector<int64_t> squeeze_axes;

        if(contains(attributes, "shrink_axis_mask"))
            shrink_axis_mask = static_cast<uint32_t>(attributes.at("shrink_axis_mask").i());

        for(size_t i = 0; i < num_axes; i++)
        {
            // the LSB corresponds to axis 0 when determining which axes to squeeze
            if(((shrink_axis_mask >> i) & bitwise_compare) == 1)
                squeeze_axes.push_back(i);
        }
        if(num_axes >= 4)
        {
            squeeze_axes = parse_axes(squeeze_axes);
        }

        auto l0 = prog.add_instruction(op, args[0]);
        return prog.add_instruction(op::squeeze{squeeze_axes}, l0);
    }

    void parse_graph(const tensorflow::GraphDef& graph)
    {
        nodes = get_nodes(graph, input_nodes);
        for(auto&& input : input_nodes)
        {
            const std::string& name   = input.name();
            attribute_map input_attrs = get_attributes(input);
            shape::type_t shape_type  = parse_type(input_attrs.at("dtype").type());
            std::vector<size_t> dims  = parse_dims(input_attrs.at("shape").shape());
            if(is_nhwc and dims.size() >= 4)
            {
                reorder_data(dims);
            }
            shape s            = shape{shape_type, dims};
            instructions[name] = prog.add_parameter(name, s);
        }
        for(auto&& p : nodes)
        {
            this->parse_node(p.first);
        }
    }

    void parse_node(const std::string& name)
    {
        if(instructions.count(name) == 0)
        {
            auto&& node = nodes.at(name);
            std::vector<instruction_ref> args;

            for(auto&& input : node.input())
            {
                if(nodes.count(input) > 0)
                {
                    auto&& iname = get_name(nodes.at(input));
                    assert(name != iname);
                    this->parse_node(iname);
                    args.push_back(instructions.at(iname));
                }
                else
                {
                    args.push_back(instructions.at(input));
                }
            }
            if(ops.count(node.op()) == 0)
            {
                instructions[name] = prog.add_instruction(op::unknown{node.op()}, args);
            }
            else
            {
                instructions[name] = ops[node.op()](get_attributes(node), args);
            }
        }
    }

    static attribute_map get_attributes(const tensorflow::NodeDef& node)
    {
        attribute_map result;
        for(auto&& attr : node.attr())
        {
            result[attr.first] = attr.second;
        }
        return result;
    }

    static std::string get_name(const tensorflow::NodeDef& node) { return node.name(); }

    static node_map get_nodes(const tensorflow::GraphDef& graph,
                              std::vector<tensorflow::NodeDef>& input_nodes)
    {
        node_map result;
        for(auto&& node : graph.node())
        {
            auto node_name = get_name(node);
            // assume each node in graph has an associated name
            if(node_name.empty())
                MIGRAPHX_THROW("tf node with no name found");
            result[node_name] = node;
            if(node.op() == "Placeholder")
            {
                input_nodes.push_back(node);
            }
        }
        return result;
    }

    static shape::type_t parse_type(const tensorflow::DataType t)
    {
        shape::type_t shape_type{};
        switch(t)
        {
        case tensorflow::DataType::DT_INVALID:
            break; // throw std::runtime_error("Unsupported type UNDEFINED");
        case tensorflow::DataType::DT_FLOAT: shape_type = shape::float_type; break;
        case tensorflow::DataType::DT_DOUBLE: shape_type = shape::double_type; break;
        case tensorflow::DataType::DT_INT32: shape_type = shape::int32_type; break;
        case tensorflow::DataType::DT_UINT8:
            break; // throw std::runtime_error("Unsupported type UINT8");
        case tensorflow::DataType::DT_INT16: shape_type = shape::int16_type; break;
        case tensorflow::DataType::DT_INT8: shape_type = shape::int8_type; break;
        case tensorflow::DataType::DT_STRING:
            break; // throw std::runtime_error("Unsupported type STRING");
        case tensorflow::DataType::DT_COMPLEX64:
            break; // throw std::runtime_error("Unsupported type COMPLEX64");
        case tensorflow::DataType::DT_INT64: shape_type = shape::int64_type; break;
        case tensorflow::DataType::DT_BOOL:
            break; // throw std::runtime_error("Unsupported type BOOL");
        case tensorflow::DataType::DT_QINT8:
            break; // throw std::runtime_error("Unsupported type QINT8");
        case tensorflow::DataType::DT_QUINT8:
            break; // throw std::runtime_error("Unsupported type QUINT8");
        case tensorflow::DataType::DT_QINT32:
            break; // throw std::runtime_error("Unsupported type QINT32");
        case tensorflow::DataType::DT_BFLOAT16:
            break; // throw std::runtime_error("Unsupported type BFLOAT16");
        case tensorflow::DataType::DT_QINT16:
            break; // throw std::runtime_error("Unsupported type QINT16");
        case tensorflow::DataType::DT_QUINT16:
            break; // throw std::runtime_error("Unsupported type QUINT16");
        case tensorflow::DataType::DT_UINT16: shape_type = shape::uint16_type; break;
        case tensorflow::DataType::DT_COMPLEX128:
            break; // throw std::runtime_error("Unsupported type COMPLEX128");
        case tensorflow::DataType::DT_HALF: shape_type = shape::half_type; break;
        case tensorflow::DataType::DT_RESOURCE:
            break; // throw std::runtime_error("Unsupported type RESOURCE");
        case tensorflow::DataType::DT_VARIANT:
            break; // throw std::runtime_error("Unsupported type VARIANT");
        case tensorflow::DataType::DT_UINT32: shape_type = shape::uint32_type; break;
        case tensorflow::DataType::DT_UINT64:
            shape_type = shape::uint64_type;
            break;

        // tf pb should not use these types
        case tensorflow::DataType::DT_FLOAT_REF: break;
        case tensorflow::DataType::DT_DOUBLE_REF: break;
        case tensorflow::DataType::DT_INT32_REF: break;
        case tensorflow::DataType::DT_UINT8_REF: break;
        case tensorflow::DataType::DT_INT16_REF: break;
        case tensorflow::DataType::DT_INT8_REF: break;
        case tensorflow::DataType::DT_STRING_REF: break;
        case tensorflow::DataType::DT_COMPLEX64_REF: break;
        case tensorflow::DataType::DT_INT64_REF: break;
        case tensorflow::DataType::DT_BOOL_REF: break;
        case tensorflow::DataType::DT_QINT8_REF: break;
        case tensorflow::DataType::DT_QUINT8_REF: break;
        case tensorflow::DataType::DT_QINT32_REF: break;
        case tensorflow::DataType::DT_BFLOAT16_REF: break;
        case tensorflow::DataType::DT_QINT16_REF: break;
        case tensorflow::DataType::DT_QUINT16_REF: break;
        case tensorflow::DataType::DT_UINT16_REF: break;
        case tensorflow::DataType::DT_COMPLEX128_REF: break;
        case tensorflow::DataType::DT_HALF_REF: break;
        case tensorflow::DataType::DT_RESOURCE_REF: break;
        case tensorflow::DataType::DT_VARIANT_REF: break;
        case tensorflow::DataType::DT_UINT32_REF: break;
        case tensorflow::DataType::DT_UINT64_REF: break;
        case tensorflow::DataType::DataType_INT_MAX_SENTINEL_DO_NOT_USE_: break;
        case tensorflow::DataType::DataType_INT_MIN_SENTINEL_DO_NOT_USE_: break;
        }
        return shape_type;
    }

    static literal parse_tensor(const tensorflow::TensorProto& t)
    {
        std::vector<size_t> dims = parse_dims(t.tensor_shape());
        size_t shape_size = std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<size_t>());
        if(!t.tensor_content().empty()) // has raw data
        {
            const std::string& s = t.tensor_content();
            switch(t.dtype())
            {
            case tensorflow::DataType::DT_INVALID: throw std::runtime_error("");
            case tensorflow::DataType::DT_FLOAT:
                return literal{{shape::float_type, dims}, s.data()};
            case tensorflow::DataType::DT_UINT8: throw std::runtime_error("");
            case tensorflow::DataType::DT_INT8: return literal{{shape::int8_type, dims}, s.data()};
            case tensorflow::DataType::DT_UINT16:
                return literal{{shape::uint16_type, dims}, s.data()};
            case tensorflow::DataType::DT_INT16:
                return literal{{shape::int16_type, dims}, s.data()};
            case tensorflow::DataType::DT_INT32:
                return literal{{shape::int32_type, dims}, s.data()};
            case tensorflow::DataType::DT_INT64:
                return literal{{shape::int64_type, dims}, s.data()};
            case tensorflow::DataType::DT_STRING: throw std::runtime_error("");
            case tensorflow::DataType::DT_BOOL: return literal{{shape::int8_type, dims}, s.data()};
            case tensorflow::DataType::DT_HALF: return literal{{shape::half_type, dims}, s.data()};
            case tensorflow::DataType::DT_DOUBLE:
                return literal{{shape::double_type, dims}, s.data()};
            case tensorflow::DataType::DT_UINT32: throw std::runtime_error("");
            case tensorflow::DataType::DT_UINT64: throw std::runtime_error("");
            case tensorflow::DataType::DT_COMPLEX64: throw std::runtime_error("");
            case tensorflow::DataType::DT_COMPLEX128: throw std::runtime_error("");
            case tensorflow::DataType::DT_QINT8: throw std::runtime_error("");
            case tensorflow::DataType::DT_QUINT8: throw std::runtime_error("");
            case tensorflow::DataType::DT_QINT32: throw std::runtime_error("");
            case tensorflow::DataType::DT_BFLOAT16: throw std::runtime_error("");
            case tensorflow::DataType::DT_QINT16: throw std::runtime_error("");
            case tensorflow::DataType::DT_QUINT16: throw std::runtime_error("");
            case tensorflow::DataType::DT_RESOURCE: throw std::runtime_error("");
            case tensorflow::DataType::DT_VARIANT: throw std::runtime_error("");
            case tensorflow::DataType::DT_FLOAT_REF: throw std::runtime_error("");
            case tensorflow::DataType::DT_DOUBLE_REF: throw std::runtime_error("");
            case tensorflow::DataType::DT_INT32_REF: throw std::runtime_error("");
            case tensorflow::DataType::DT_UINT8_REF: throw std::runtime_error("");
            case tensorflow::DataType::DT_INT16_REF: throw std::runtime_error("");
            case tensorflow::DataType::DT_INT8_REF: throw std::runtime_error("");
            case tensorflow::DataType::DT_STRING_REF: throw std::runtime_error("");
            case tensorflow::DataType::DT_COMPLEX64_REF: throw std::runtime_error("");
            case tensorflow::DataType::DT_INT64_REF: throw std::runtime_error("");
            case tensorflow::DataType::DT_BOOL_REF: throw std::runtime_error("");
            case tensorflow::DataType::DT_QINT8_REF: throw std::runtime_error("");
            case tensorflow::DataType::DT_QUINT8_REF: throw std::runtime_error("");
            case tensorflow::DataType::DT_QINT32_REF: throw std::runtime_error("");
            case tensorflow::DataType::DT_BFLOAT16_REF: throw std::runtime_error("");
            case tensorflow::DataType::DT_QINT16_REF: throw std::runtime_error("");
            case tensorflow::DataType::DT_QUINT16_REF: throw std::runtime_error("");
            case tensorflow::DataType::DT_UINT16_REF: throw std::runtime_error("");
            case tensorflow::DataType::DT_COMPLEX128_REF: throw std::runtime_error("");
            case tensorflow::DataType::DT_HALF_REF: throw std::runtime_error("");
            case tensorflow::DataType::DT_RESOURCE_REF: throw std::runtime_error("");
            case tensorflow::DataType::DT_VARIANT_REF: throw std::runtime_error("");
            case tensorflow::DataType::DT_UINT32_REF: throw std::runtime_error("");
            case tensorflow::DataType::DT_UINT64_REF: throw std::runtime_error("");
            case tensorflow::DataType::DataType_INT_MAX_SENTINEL_DO_NOT_USE_:
                throw std::runtime_error("");
            case tensorflow::DataType::DataType_INT_MIN_SENTINEL_DO_NOT_USE_:
                throw std::runtime_error("");
            }
            MIGRAPHX_THROW("Invalid tensor type");
        }
        switch(t.dtype())
        {
        case tensorflow::DataType::DT_INVALID: throw std::runtime_error("");
        case tensorflow::DataType::DT_FLOAT:
            return create_literal(
                shape::float_type, dims, get_data_vals(t.float_val(), shape_size));
        case tensorflow::DataType::DT_UINT8: throw std::runtime_error("");
        case tensorflow::DataType::DT_INT8:
            return create_literal(shape::int8_type, dims, get_data_vals(t.int_val(), shape_size));
        case tensorflow::DataType::DT_UINT16:
            return create_literal(shape::uint16_type, dims, get_data_vals(t.int_val(), shape_size));
        case tensorflow::DataType::DT_INT16:
            return create_literal(shape::int16_type, dims, get_data_vals(t.int_val(), shape_size));
        case tensorflow::DataType::DT_INT32:
            return create_literal(shape::int32_type, dims, get_data_vals(t.int_val(), shape_size));
        case tensorflow::DataType::DT_INT64:
            return create_literal(
                shape::int64_type, dims, get_data_vals(t.int64_val(), shape_size));
        case tensorflow::DataType::DT_STRING: throw std::runtime_error("");
        case tensorflow::DataType::DT_BOOL:
            return create_literal(shape::int32_type, dims, get_data_vals(t.bool_val(), shape_size));
        case tensorflow::DataType::DT_HALF:
        {
            std::vector<int> data_int32 = get_data_vals(t.half_val(), shape_size);
            std::vector<uint16_t> data_uint16(data_int32.begin(), data_int32.end());
            std::vector<half> data_half;
            std::transform(data_uint16.begin(),
                           data_uint16.end(),
                           std::back_inserter(data_half),
                           [](uint16_t raw_val) { return *reinterpret_cast<half*>(&raw_val); });
            return create_literal(shape::half_type, dims, data_half);
        }
        case tensorflow::DataType::DT_DOUBLE:
            return literal{{shape::double_type, dims}, get_data_vals(t.double_val(), shape_size)};
        case tensorflow::DataType::DT_UINT32: throw std::runtime_error("");
        case tensorflow::DataType::DT_UINT64: throw std::runtime_error("");
        case tensorflow::DataType::DT_COMPLEX64: throw std::runtime_error("");
        case tensorflow::DataType::DT_COMPLEX128: throw std::runtime_error("");
        case tensorflow::DataType::DT_QINT8: throw std::runtime_error("");
        case tensorflow::DataType::DT_QUINT8: throw std::runtime_error("");
        case tensorflow::DataType::DT_QINT32: throw std::runtime_error("");
        case tensorflow::DataType::DT_BFLOAT16: throw std::runtime_error("");
        case tensorflow::DataType::DT_QINT16: throw std::runtime_error("");
        case tensorflow::DataType::DT_QUINT16: throw std::runtime_error("");
        case tensorflow::DataType::DT_RESOURCE: throw std::runtime_error("");
        case tensorflow::DataType::DT_VARIANT: throw std::runtime_error("");
        case tensorflow::DataType::DT_FLOAT_REF: throw std::runtime_error("");
        case tensorflow::DataType::DT_DOUBLE_REF: throw std::runtime_error("");
        case tensorflow::DataType::DT_INT32_REF: throw std::runtime_error("");
        case tensorflow::DataType::DT_UINT8_REF: throw std::runtime_error("");
        case tensorflow::DataType::DT_INT16_REF: throw std::runtime_error("");
        case tensorflow::DataType::DT_INT8_REF: throw std::runtime_error("");
        case tensorflow::DataType::DT_STRING_REF: throw std::runtime_error("");
        case tensorflow::DataType::DT_COMPLEX64_REF: throw std::runtime_error("");
        case tensorflow::DataType::DT_INT64_REF: throw std::runtime_error("");
        case tensorflow::DataType::DT_BOOL_REF: throw std::runtime_error("");
        case tensorflow::DataType::DT_QINT8_REF: throw std::runtime_error("");
        case tensorflow::DataType::DT_QUINT8_REF: throw std::runtime_error("");
        case tensorflow::DataType::DT_QINT32_REF: throw std::runtime_error("");
        case tensorflow::DataType::DT_BFLOAT16_REF: throw std::runtime_error("");
        case tensorflow::DataType::DT_QINT16_REF: throw std::runtime_error("");
        case tensorflow::DataType::DT_QUINT16_REF: throw std::runtime_error("");
        case tensorflow::DataType::DT_UINT16_REF: throw std::runtime_error("");
        case tensorflow::DataType::DT_COMPLEX128_REF: throw std::runtime_error("");
        case tensorflow::DataType::DT_HALF_REF: throw std::runtime_error("");
        case tensorflow::DataType::DT_RESOURCE_REF: throw std::runtime_error("");
        case tensorflow::DataType::DT_VARIANT_REF: throw std::runtime_error("");
        case tensorflow::DataType::DT_UINT32_REF: throw std::runtime_error("");
        case tensorflow::DataType::DT_UINT64_REF: throw std::runtime_error("");
        case tensorflow::DataType::DataType_INT_MAX_SENTINEL_DO_NOT_USE_:
            throw std::runtime_error("");
        case tensorflow::DataType::DataType_INT_MIN_SENTINEL_DO_NOT_USE_:
            throw std::runtime_error("");
        }
        MIGRAPHX_THROW("Invalid tensor type");
    }

    template <class T>
    static std::vector<T> get_data_vals(const google::protobuf::RepeatedField<T>& data,
                                        const size_t& shape_size)
    {
        std::vector<T> data_vals(shape_size);
        // check if shape has enough data values given existing fields
        if(data.size() == 1)
        {
            std::fill(data_vals.begin(), data_vals.end(), data[0]);
        }
        else
            copy(data.begin(), data.end(), std::back_inserter(data_vals));
        return data_vals;
    }

    static std::vector<size_t> parse_dims(const tensorflow::TensorShapeProto& s)
    {
        std::vector<size_t> dims;
        auto input_dims = s.dim();
        std::transform(input_dims.begin(),
                       input_dims.end(),
                       std::back_inserter(dims),
                       [](const tensorflow::TensorShapeProto_Dim& dim) { return dim.size(); });
        return dims;
    }

    template <class T>
    static literal
    create_literal(shape::type_t shape_type, const std::vector<size_t>& dims, std::vector<T> data)
    {
        // assume if explicit value is mentioned in protobuf and dim size <= 1, treat as scalar
        if(dims.empty() or (dims.size() == 1 and dims.front() == 1))
            return literal{{shape_type}, data};
        return literal{{shape_type, dims}, data};
    }
};

program parse_tf(const std::string& name, bool is_nhwc)
{
    std::fstream input(name.c_str(), std::ios::in | std::ios::binary);
    tf_parser parser;
    parser.is_nhwc = is_nhwc;

#ifndef NDEBUG
    // Log the program when it can't be parsed
    try
    {
        parser.parse_from(input);
    }
    catch(...)
    {
        std::cerr << parser.prog << std::endl;
        throw;
    }
#else
    parser.parse_from(input);
#endif
    return std::move(parser.prog);
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
