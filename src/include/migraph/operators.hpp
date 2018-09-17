#ifndef MIGRAPH_GUARD_OPERATORS_HPP
#define MIGRAPH_GUARD_OPERATORS_HPP

#include <array>
#include <migraph/operation.hpp>
#include <migraph/check_shapes.hpp>
#include <migraph/stringutils.hpp>
#include <migraph/streamutils.hpp>
#include <cmath>
#include <utility>

namespace migraph {

struct not_computable
{
    argument compute(context&, const shape&, const std::vector<argument>&) const
    {
        MIGRAPH_THROW("not computable");
    }
};

struct batch_norm_inference
{
    float epsilon  = 1.0e-6f;
    float momentum = 0.9f;

    std::string name() const { return "batch_norm_inference"; }

    enum bn_infer_mode_t
    {
        per_activation,
        spatial,
    };

    bn_infer_mode_t bn_mode = spatial;

    bool is_test = false;

    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this}.has(5);
        return inputs.front();
    }
};

struct convolution
{
    std::array<std::size_t, 2> padding  = {{0, 0}};
    std::array<std::size_t, 2> stride   = {{1, 1}};
    std::array<std::size_t, 2> dilation = {{1, 1}};
    enum padding_mode_t
    {
        default_, // NOLINT
        same,
        valid
    };
    padding_mode_t padding_mode = default_;
    std::string name() const { return "convolution"; }
    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this}.has(2).same_type().same_ndims().only_dims(4);

        const shape& input   = inputs.at(0);
        const shape& weights = inputs.at(1);
        auto t               = input.type();
        if(padding_mode == default_)
        {
            return {t,
                    {
                        input.lens()[0],
                        weights.lens()[0],
                        std::size_t(std::max<std::ptrdiff_t>(
                            1,
                            (input.lens()[2] - (1 + dilation[0] * (weights.lens()[2] - 1)) +
                             2 * padding[0]) /
                                    stride[0] +
                                1)),
                        std::size_t(std::max<std::ptrdiff_t>(
                            1,
                            (input.lens()[3] - (1 + dilation[1] * (weights.lens()[3] - 1)) +
                             2 * padding[1]) /
                                    stride[1] +
                                1)),
                    }};
        }
        else if(padding_mode == same)
        {
            return {t,
                    {input.lens()[0],
                     weights.lens()[0],
                     static_cast<std::size_t>(
                         std::ceil(static_cast<double>(input.lens()[2]) / stride[0])),
                     static_cast<std::size_t>(
                         std::ceil(static_cast<double>(input.lens()[3]) / stride[1]))}};
        }
        else if(padding_mode == valid)
        {
            return {
                t,
                {input.lens()[0],
                 weights.lens()[0],
                 static_cast<std::size_t>(std::ceil(
                     static_cast<double>(input.lens()[2] - weights.lens()[2] + 1) / stride[0])),
                 static_cast<std::size_t>(std::ceil(
                     static_cast<double>(input.lens()[3] - weights.lens()[3] + 1) / stride[1]))}};
        }
        else
        {
            MIGRAPH_THROW("Invalid padding mode");
        }
    }

    friend std::ostream& operator<<(std::ostream& os, const convolution& op)
    {
        os << op.name() << "[";
        os << "padding={" << stream_range(op.padding) << "}, ";
        os << "stride={" << stream_range(op.stride) << "}, ";
        os << "dilation={" << stream_range(op.dilation) << "}";
        os << "]";
        return os;
    }
};

struct im2col
{
    std::array<std::size_t, 2> padding  = {{0, 0}};
    std::array<std::size_t, 2> stride   = {{1, 1}};
    std::array<std::size_t, 2> dilation = {{1, 1}};
    enum padding_mode_t
    {
        default_, // NOLINT
        same,
        valid
    };

    std::string name() const { return "im2col"; }

    shape compute_shape(std::vector<shape> inputs) const
    {
        auto input          = inputs[0];
        auto weights        = inputs[1];
        auto batch_size     = input.lens()[0];
        auto input_channels = weights.lens()[1];
        auto kernel_height  = weights.lens()[2];
        auto kernel_width   = weights.lens()[3];
        check_shapes{inputs, *this}.has(2);
        if(batch_size != 1)
            MIGRAPH_THROW("im2col only support batch_size 1");
        auto output_height = std::size_t(std::max<std::ptrdiff_t>(
            1,
            (input.lens()[2] - (1 + dilation[0] * (kernel_height - 1)) + 2 * padding[0]) /
                    stride[0] +
                1));
        auto output_width  = std::size_t(std::max<std::ptrdiff_t>(
            1,
            (input.lens()[3] - (1 + dilation[1] * (kernel_width - 1)) + 2 * padding[1]) /
                    stride[1] +
                1));
        auto channels_col  = kernel_height * kernel_width * input_channels;
        return {input.type(), {output_height * output_width, channels_col}};
    }
};

struct pooling
{
    std::string mode                   = "average";
    std::array<std::size_t, 2> padding = {{0, 0}};
    std::array<std::size_t, 2> stride  = {{1, 1}};
    std::array<std::size_t, 2> lengths = {{1, 1}};
    std::string name() const { return "pooling"; }

    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this}.has(1).only_dims(4);

        const shape& input = inputs.at(0);
        auto t             = input.type();

        assert(lengths[0] <= (input.lens()[2] + 2 * padding[0]));
        assert(lengths[1] <= (input.lens()[3] + 2 * padding[1]));

        return {t,
                {
                    input.lens()[0],
                    input.lens()[1],
                    std::size_t(std::max<std::ptrdiff_t>(
                        1,
                        std::ptrdiff_t(std::floor((input.lens()[2] + 2 * padding[0] - lengths[0]) /
                                                  static_cast<float>(stride[0]))) +
                            1)),
                    std::size_t(std::max<std::ptrdiff_t>(
                        1,
                        std::ptrdiff_t(std::floor((input.lens()[3] + 2 * padding[1] - lengths[1]) /
                                                  static_cast<float>(stride[1]))) +
                            1)),
                }};
    }

    friend std::ostream& operator<<(std::ostream& os, const pooling& op)
    {
        os << op.name() << "[";
        os << "padding={" << stream_range(op.padding) << "}, ";
        os << "stride={" << stream_range(op.stride) << "}, ";
        os << "lengths={" << stream_range(op.lengths) << "}";
        os << "]";
        return os;
    }
};

struct activation
{
    std::string mode;
    std::string name() const { return "activation"; }
    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this}.has(1);
        return inputs.front();
    }
    friend std::ostream& operator<<(std::ostream& os, const activation& op)
    {
        os << op.name() << ":" << op.mode;
        return os;
    }
};

struct transpose
{
    std::vector<int64_t> dims;
    std::string name() const { return "transpose"; }
    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this}.has(1);
        auto input         = inputs.at(0);
        auto input_lens    = input.lens();
        auto input_strides = input.strides();
        auto t             = input.type();
        if(dims.size() != input_lens.size())
        {
            MIGRAPH_THROW("Permutation has wrong number of axes");
        }
        std::vector<int64_t> axes(dims.size());
        std::iota(axes.begin(), axes.end(), 0);
        if(!std::is_permutation(axes.begin(), axes.end(), dims.begin()))
        {
            MIGRAPH_THROW("Invalid permutation");
        }
        std::vector<size_t> output_lens(input_lens.size());
        std::vector<size_t> output_strides(input_lens.size());
        for(int i = 0; i < output_lens.size(); i++)
        {
            output_lens[i]    = input_lens[dims[i]];
            output_strides[i] = input_strides[dims[i]];
        }
        return {t, output_lens, output_strides};
    }
    argument compute(context&, shape output_shape, std::vector<argument> args) const
    {
        return {std::move(output_shape), std::move(args.front().data)};
    }
    friend std::ostream& operator<<(std::ostream& os, const transpose& op)
    {
        os << op.name() << "[";
        os << "dims={" << stream_range(op.dims) << "}";
        os << "]";
        return os;
    }
};

struct contiguous
{
    std::string name() const { return "contiguous"; }
    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this}.has(1);
        auto lens = inputs.at(0).lens();
        auto t    = inputs.at(0).type();
        if(lens.size() < 2)
        {
            MIGRAPH_THROW("Number of dimensions should exceed 1");
        }
        return {t, lens};
    }
};

struct reshape
{
    std::vector<int64_t> dims;
    std::string name() const { return "reshape"; }
    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this}.has(1);
        auto&& idims = inputs.front().lens();
        std::vector<std::size_t> rdims(dims.begin(), dims.end());
        auto n_neg_dims = std::count(dims.begin(), dims.end(), -1);
        if(n_neg_dims > 1)
            MIGRAPH_THROW("Dimensions for reshape can only have one -1 dim");
        for(std::size_t i = 0; i < dims.size(); i++)
        {
            if(dims[i] == 0)
                rdims[i] = idims[i];
        }
        if(n_neg_dims > 0)
        {
            size_t missing_dim =
                -inputs.front().elements() /
                std::accumulate(rdims.begin(), rdims.end(), 1, std::multiplies<int64_t>());
            for(std::size_t i = 0; i < rdims.size(); i++)
            {
                if(dims[i] == -1)
                    rdims[i] = missing_dim;
            }
        }
        if(dims.back() == -1)
        {
            rdims.pop_back();
            std::copy(idims.begin() + rdims.size(), idims.end(), std::back_inserter(rdims));
        }
        shape s{inputs.front().type(), rdims};
        if(s.elements() != inputs.front().elements())
            MIGRAPH_THROW("Wrong number of elements for reshape");
        return s;
    }
    argument compute(context&, shape output_shape, std::vector<argument> args) const
    {
        return {std::move(output_shape), std::move(args.front().data)};
    }
    friend std::ostream& operator<<(std::ostream& os, const reshape& op)
    {
        os << op.name() << "[";
        os << "dims={" << stream_range(op.dims) << "}";
        os << "]";
        return os;
    }
};

struct gemm
{
    float alpha = 1.0;
    float beta  = 0.0;
    std::string name() const { return "gemm"; }
    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this}.has(2).same_type();
        const shape& a = inputs.at(0);
        const shape& b = inputs.at(1);
        auto t         = a.type();

        if(a.lens()[1] != b.lens()[0])
            MIGRAPH_THROW("Inner dimensions do not match: {" + to_string_range(a.lens()) + "} x {" +
                          to_string_range(b.lens()) + "}");
        return {t, {a.lens()[0], b.lens()[1]}};
    }

    friend std::ostream& operator<<(std::ostream& os, const gemm& op)
    {
        os << op.name() << "[";
        os << "]";
        return os;
    }
};

struct unary
{
    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs}.has(1);
        return inputs.at(0);
    }
};

struct identity : unary
{
    std::string name() const { return "identity"; }
};

struct abs : unary
{
    std::string name() const { return "abs"; }
};

struct exp : unary
{
    std::string name() const { return "exp"; }
};

struct sin : unary
{
    std::string name() const { return "sin"; }
};

struct cos : unary
{
    std::string name() const { return "cos"; }
};

struct tan : unary
{
    std::string name() const { return "tan"; }
};

struct asin : unary
{
    std::string name() const { return "asin"; }
};

struct acos : unary
{
    std::string name() const { return "acos"; }
};

struct atan : unary
{
    std::string name() const { return "atan"; }
};

struct tanh : unary
{
    std::string name() const { return "tanh"; }
};

struct sigmoid : unary
{
    std::string name() const { return "sigmoid"; }
};

struct neg : unary
{
    std::string name() const { return "neg"; }
};

struct softmax
{
    std::string name() const { return "softmax"; }
    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs}.has(1).only_dims(4);
        return inputs.at(0);
    }
};

struct flatten
{
    uint64_t axis = 0;
    std::string name() const { return "flatten"; }
    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs}.has(1);
        auto&& lens = inputs.front().lens();

        if(axis > lens.size())
        {
            MIGRAPH_THROW("axis for flatten must be less than tensor rank");
        }
        auto x =
            std::accumulate(lens.begin(), lens.begin() + axis, std::size_t{1}, std::multiplies<>{});
        auto y =
            std::accumulate(lens.begin() + axis, lens.end(), std::size_t{1}, std::multiplies<>{});
        return {inputs.at(0).type(), {x, y}};
    }
    argument compute(context&, shape output_shape, std::vector<argument> args) const
    {
        return {std::move(output_shape), std::move(args.front().data)};
    }
    friend std::ostream& operator<<(std::ostream& os, const flatten& op)
    {
        os << op.name() << "[";
        os << "axis=" << op.axis;
        os << "]";
        return os;
    }
};
struct broadcast
{
    uint64_t axis = 0;
    std::string name() const { return "broadcast"; }
    shape compute_shape(std::vector<shape> inputs) const
    {
        auto t      = inputs.at(0).type();
        auto result = inputs.at(0);
        auto input  = inputs.at(1);

        std::vector<size_t> bcast_strides(result.lens().size(), 0);

        if(std::all_of(
               result.lens().cbegin(), result.lens().cend(), [&](auto x) { return x == 1; }))
        {
            if(axis != 0)
                MIGRAPH_THROW("when broadcasting tensor of size 1, axis should be 0");
            return {t, result.lens(), std::move(bcast_strides)};
        }
        else
        {
            assert(result.lens().size() - axis >= input.lens().size());
            if(!std::equal(input.lens().begin(), input.lens().end(), result.lens().begin() + axis))
                MIGRAPH_THROW("when broadcasting success sizes must match");
            std::copy(input.strides().begin(), input.strides().end(), bcast_strides.begin() + axis);
            return {t, result.lens(), std::move(bcast_strides)};
        }
    }
    argument compute(context&, shape output_shape, std::vector<argument> args) const
    {
        return {std::move(output_shape), std::move(args.at(1).data)};
    }
    friend std::ostream& operator<<(std::ostream& os, const broadcast& op)
    {
        os << op.name() << "[";
        os << "axis=" << op.axis;
        os << "]";
        return os;
    }
};

struct binary
{
    uint64_t broadcast = 0;
    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs}.has(2).same_type().same_dims();
        return inputs.at(0);
    }
};

struct add : binary
{
    std::string name() const { return "add"; }
};

struct sub : binary
{
    std::string name() const { return "sub"; }
};

struct mul : binary
{
    std::string name() const { return "mul"; }
};

struct div : binary
{
    std::string name() const { return "div"; }
};

struct load
{
    shape s;
    std::size_t offset = 0;
    std::string name() const { return "load"; }
    shape compute_shape(const std::vector<shape>& inputs) const
    {
        check_shapes{inputs}.has(1);
        return s;
    }
    argument compute(context&, const shape&, const std::vector<argument>& args) const
    {
        return {s, args[0].data() + offset};
    }
};

struct outline
{
    shape s;
    std::string name() const { return "outline"; }
    shape compute_shape(const std::vector<shape>& inputs) const
    {
        check_shapes{inputs, *this}.has(0);
        return s;
    }
    argument compute(context&, const shape&, const std::vector<argument>&) const
    {
        return {s, nullptr};
    }
};

} // namespace migraph

#endif
