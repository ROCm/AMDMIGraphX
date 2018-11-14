#ifndef MIGRAPH_GUARD_OPERATORS_HPP
#define MIGRAPH_GUARD_OPERATORS_HPP

#include <array>
#include <migraphx/operation.hpp>
#include <migraphx/check_shapes.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/streamutils.hpp>
#include <migraphx/config.hpp>
#include <cmath>
#include <utility>

namespace migraphx {
inline namespace MIGRAPH_INLINE_NS {
namespace op {

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

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(
            f(self.epsilon, "epsilon"), f(self.momentum, "momentum"), f(self.bn_mode, "bn_mode"));
    }

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

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.padding, "padding"),
                    f(self.stride, "stride"),
                    f(self.dilation, "dilation"),
                    f(self.padding_mode, "padding_mode"));
    }

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
    padding_mode_t padding_mode = default_;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.padding, "padding"),
                    f(self.stride, "stride"),
                    f(self.dilation, "dilation"),
                    f(self.padding_mode, "padding_mode"));
    }

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

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.mode, "mode"),
                    f(self.padding, "padding"),
                    f(self.stride, "stride"),
                    f(self.lengths, "lengths"));
    }

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
};

struct leaky_relu
{
    std::string name() const { return "leaky_relu"; }
    float alpha;
    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this}.has(1);
        return inputs.front();
    }
    friend std::ostream& operator<<(std::ostream& os, const leaky_relu& op)
    {
        os << op.name() << ":" << op.alpha;
        return os;
    }
};

struct transpose
{
    std::vector<int64_t> dims;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.dims, "dims"));
    }

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
    int output_alias(const std::vector<shape>&) const { return 0; }
};

struct contiguous
{
    std::string name() const { return "contiguous"; }
    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this}.has(1);
        auto lens = inputs.at(0).lens();
        auto t    = inputs.at(0).type();
        return {t, lens};
    }
};

struct concat
{
    std::size_t axis = 0;
    std::string name() const { return "concat"; }
    std::vector<std::size_t> compute_offsets(const shape& output_shape,
                                             const std::vector<argument> args) const
    {
        std::vector<std::size_t> offsets;
        std::vector<std::size_t> offset(args[0].get_shape().lens().size(), 0);
        offset[axis] = 0;
        for(const auto& arg : args)
        {
            offsets.push_back(output_shape.index(offset));
            offset[axis] += arg.get_shape().lens()[axis];
        }
        return offsets;
    }
    shape compute_shape(std::vector<shape> inputs) const
    {
        if(inputs.empty())
        {
            MIGRAPH_THROW("Number of input tensors should exceed 0");
        }

        const auto& first_shape_lens = inputs.front().lens();
        const auto& type             = inputs.front().type();
        for(std::size_t l = 0; l < first_shape_lens.size(); l++)
        {
            if(l != axis)
            {
                if(!std::all_of(inputs.begin(), inputs.end(), [&](auto s) {
                       return s.lens()[l] == first_shape_lens[l];
                   }))
                {
                    MIGRAPH_THROW("Non-axis dimensions should match");
                }
            }
        }
        std::size_t new_dim_axis = 0;
        for(const auto& input : inputs)
        {
            const auto& lens = input.lens();
            new_dim_axis += lens[axis];
        }
        std::vector<std::size_t> new_lens;
        std::copy(first_shape_lens.begin(), first_shape_lens.end(), std::back_inserter(new_lens));
        new_lens[axis] = new_dim_axis;
        return {type, new_lens};
    }
    int output_alias(const std::vector<shape>&) const { return 0; }
};

struct slice
{
    std::vector<int64_t> axes;
    std::vector<int64_t> starts;
    std::vector<int64_t> ends;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.axes, "axes"), f(self.starts, "starts"), f(self.ends, "ends"));
    }

    std::string name() const { return "slice"; }

    auto fix_index(const std::vector<std::size_t>& lens, std::size_t axis, int64_t index) const
    {
        int64_t r = std::min(index, static_cast<int64_t>(lens[axis]));
        if(r < 0)
            r += lens[axis];
        return std::size_t(r);
    }

    auto compute_offset(const shape& s) const
    {
        const std::vector<std::size_t>& lens    = s.lens();
        const std::vector<std::size_t>& strides = s.strides();
        auto offset                             = 0;
        if(!axes.empty())
        {
            for(std::size_t i = 0; i < axes.size(); i++)
            {
                auto axis = axes[i];
                offset += fix_index(lens, axis, starts[i]) * strides[axis];
            }
        }
        else
        {
            for(std::size_t axis = 0; axis < lens.size(); axis++)
            {
                offset += fix_index(lens, axis, starts[axis]) * strides[axis];
            }
        }
        return offset;
    }

    shape compute_shape(std::vector<shape> inputs) const
    {
        auto input_shape        = inputs[0];
        auto t                  = input_shape.type();
        const auto& old_lens    = input_shape.lens();
        const auto& old_strides = input_shape.strides();
        // std::vector<int64_t> t_axes(old_lens.size());
        // if(axes.size() == 0)
        // {
        //     std::iota(t_axes.begin(), t_axes.end(), 0);
        // }
        // else
        // {
        //     std::copy(axes.begin(), axes.end(), t_axes.begin());
        // }
        if(starts.size() != axes.size() || axes.size() != ends.size())
        {
            MIGRAPH_THROW("inconsistent sizes");
        }
        std::vector<std::size_t> new_lens = old_lens;
        for(std::size_t i = 0; i < axes.size(); i++)
        {
            auto axis = axes[i];
            new_lens[axis] =
                fix_index(old_lens, axis, ends[i]) - fix_index(old_lens, axis, starts[i]);
        }
        return shape{t, new_lens, old_strides};
    }
    argument compute(context&, shape output_shape, std::vector<argument> args) const
    {
        auto input  = args[0];
        auto offset = compute_offset(input.get_shape()) * output_shape.type_size();
        return {std::move(output_shape), [=] { return input.data() + offset; }};
    }
    int output_alias(const std::vector<shape>&) const { return 0; }
};

struct squeeze
{
    std::vector<int64_t> axes;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.axes, "axes"));
    }

    std::string name() const { return "squeeze"; }
    shape compute_shape(std::vector<shape> inputs) const
    {
        auto input_shape = inputs[0];
        auto type        = input_shape.type();
        auto old_lens    = input_shape.lens();
        if(std::any_of(
               axes.begin(), axes.end(), [&](auto axis) { return input_shape.lens()[axis] != 1; }))
        {
            MIGRAPH_THROW("squeeze axis dimension should be equal to 1");
        }
        std::vector<std::size_t> new_lens;
        if(axes.empty())
        {
            std::copy_if(old_lens.begin(),
                         old_lens.end(),
                         std::back_inserter(new_lens),
                         [](auto len) { return len != 1; });
        }
        else
        {
            for(std::size_t i = 0; i < old_lens.size(); i++)
            {
                if(std::find(axes.begin(), axes.end(), i) == axes.end())
                {
                    new_lens.push_back(old_lens[i]);
                }
            }
        }
        return shape{type, new_lens};
    }
    argument compute(context&, shape output_shape, std::vector<argument> args) const
    {
        return {std::move(output_shape), std::move(args.front().data)};
    }
    int output_alias(const std::vector<shape>&) const { return 0; }
};

struct unsqueeze
{
    std::vector<int64_t> axes;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.axes, "axes"));
    }

    std::string name() const { return "unsqueeze"; }
    shape compute_shape(std::vector<shape> inputs) const
    {
        auto input_shape     = inputs[0];
        auto type            = input_shape.type();
        auto old_lens        = input_shape.lens();
        std::size_t new_size = old_lens.size() + axes.size();
        std::vector<std::size_t> new_lens(new_size);
        std::size_t p = 0;
        for(std::size_t i = 0; i < new_size; i++)
        {
            if(std::find(axes.begin(), axes.end(), i) != axes.end())
            {
                new_lens[i] = 1;
            }
            else
            {
                new_lens[i] = old_lens[p++];
            }
        }
        return shape{type, new_lens};
    }
    argument compute(context&, shape output_shape, std::vector<argument> args) const
    {
        return {std::move(output_shape), std::move(args.front().data)};
    }
    int output_alias(const std::vector<shape>&) const { return 0; }
};

struct reshape
{
    std::vector<int64_t> dims;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.dims, "dims"));
    }

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
    int output_alias(const std::vector<shape>&) const { return 0; }
};

struct dot
{
    float alpha = 1.0;
    float beta  = 0.0;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.alpha, "alpha"), f(self.beta, "beta"));
    }

    std::string name() const { return "dot"; }
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
};

struct unary
{
    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs}.has(1);
        return inputs.at(0);
    }
};

struct identity
{
    std::string name() const { return "identity"; }
    shape compute_shape(std::vector<shape> inputs) const { return inputs.at(0); }
    argument compute(context&, shape output_shape, std::vector<argument> args) const
    {
        return {std::move(output_shape), std::move(args.at(0).data)};
    }
    int output_alias(const std::vector<shape>&) const { return 0; }
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

struct relu : unary
{
    std::string name() const { return "relu"; }
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

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.axis, "axis"));
    }

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
    int output_alias(const std::vector<shape>&) const { return 0; }
};
struct broadcast
{
    uint64_t axis = 0;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.axis, "axis"));
    }

    shape broadcast_shape;
    std::string name() const { return "broadcast"; }
    shape compute_shape(std::vector<shape> inputs) const
    {
        auto t     = inputs.at(0).type();
        auto input = inputs.at(0);

        std::vector<size_t> bcast_strides(broadcast_shape.lens().size(), 0);

        if(std::all_of(broadcast_shape.lens().cbegin(), broadcast_shape.lens().cend(), [&](auto x) {
               return x == 1;
           }))
        {
            if(axis != 0)
                MIGRAPH_THROW("when broadcasting tensor of size 1, axis should be 0");
            return {t, broadcast_shape.lens(), std::move(bcast_strides)};
        }
        else
        {
            assert(broadcast_shape.lens().size() - axis >= input.lens().size());
            if(!std::equal(
                   input.lens().begin(), input.lens().end(), broadcast_shape.lens().begin() + axis))
                MIGRAPH_THROW("when broadcasting success sizes must match");
            std::copy(input.strides().begin(), input.strides().end(), bcast_strides.begin() + axis);
            return {t, broadcast_shape.lens(), std::move(bcast_strides)};
        }
    }
    argument compute(context&, shape output_shape, std::vector<argument> args) const
    {
        return {std::move(output_shape), std::move(args.at(0).data)};
    }
    int output_alias(const std::vector<shape>&) const { return 0; }
};

struct multibroadcast
{
    std::vector<std::size_t> output_lens;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.output_lens, "output_lens"));
    }

    std::string name() const { return "multibroadcast"; }

    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this}.has(1);
        auto t     = inputs.at(0).type();
        auto input = inputs.at(0);

        if(input.lens().empty())
            MIGRAPH_THROW("inputs dimensions should be > 0");

        if(input.lens().size() > output_lens.size())
            MIGRAPH_THROW("inputs dimensions should <= output size");

        std::vector<size_t> bcast_strides(output_lens.size(), 0);
        auto offset = output_lens.size() - input.lens().size();
        for(int i = input.lens().size() - 1; i >= 0; i--)
        {
            if(output_lens[i + offset] == input.lens()[i])
            {
                bcast_strides[i + offset] = input.strides()[i];
            }
        }
        return {t, output_lens, bcast_strides};
    }
    argument compute(context&, shape output_shape, std::vector<argument> args) const
    {
        return {std::move(output_shape), std::move(args.at(0).data)};
    }
    int output_alias(const std::vector<shape>&) const { return 0; }
};

struct scalar
{
    shape scalar_bcast;

    std::string name() const { return "scalar"; }

    shape compute_shape(std::vector<shape> inputs) const
    {
        assert(check_shapes{inputs}.has(1).only_dims(1).size() == 1);
        auto t     = inputs.at(0).type();
        auto input = inputs.at(0);
        std::vector<std::size_t> strides(scalar_bcast.lens().size(), 0);
        return {t, scalar_bcast.lens(), strides};
    }

    argument compute(context&, shape output_shape, std::vector<argument> args) const
    {
        return {std::move(output_shape), std::move(args.at(0).data)};
    }
    int output_alias(const std::vector<shape>&) const { return 0; }
};

struct binary
{
    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs}.has(2).same_type().same_dims();
        auto t    = inputs.at(0).type();
        auto lens = inputs.at(0).lens();
        return {t, lens};
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

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.s, "shape"), f(self.offset, "offset"));
    }

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
    int output_alias(const std::vector<shape>&) const { return 0; }
};

struct outline
{
    shape s;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.s, "shape"));
    }

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

} // namespace op
} // namespace MIGRAPH_INLINE_NS
} // namespace migraphx

#endif
