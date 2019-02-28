#ifndef MIGRAPHX_GUARD_OPERATORS_HPP
#define MIGRAPHX_GUARD_OPERATORS_HPP

#include <array>
#include <migraphx/operation.hpp>
#include <migraphx/check_shapes.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/streamutils.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/shape_for_each.hpp>
#include <migraphx/config.hpp>
#include <cmath>
#include <utility>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

enum padding_mode_t
{
    default_, // NOLINT
    same,
    valid
};

struct not_computable
{
    argument compute(const shape&, const std::vector<argument>&) const
    {
        MIGRAPHX_THROW("not computable");
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

struct lrn
{
    float alpha = 0.0001;
    float beta  = 0.75;
    float bias  = 1.0;
    int size    = 1;
    std::string name() const { return "lrn"; }

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.alpha, "alpha"),
                    f(self.beta, "beta"),
                    f(self.bias, "bias"),
                    f(self.size, "size"));
    }

    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this}.has(1);
        return inputs.front();
    }
};

struct convolution
{
    std::array<std::size_t, 2> padding  = {{0, 0}};
    std::array<std::size_t, 2> stride   = {{1, 1}};
    std::array<std::size_t, 2> dilation = {{1, 1}};

    padding_mode_t padding_mode = default_;
    int group                   = 1;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.padding, "padding"),
                    f(self.stride, "stride"),
                    f(self.dilation, "dilation"),
                    f(self.padding_mode, "padding_mode"),
                    f(self.group, "group"));
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
            MIGRAPHX_THROW("Invalid padding mode");
        }
    }
};

struct im2col
{
    std::array<std::size_t, 2> padding  = {{0, 0}};
    std::array<std::size_t, 2> stride   = {{1, 1}};
    std::array<std::size_t, 2> dilation = {{1, 1}};

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
            MIGRAPHX_THROW("im2col only support batch_size 1");
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
    padding_mode_t padding_mode        = default_;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.mode, "mode"),
                    f(self.padding, "padding"),
                    f(self.padding, "padding_mode"),
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

        if(padding_mode == default_)
        {
            return {
                t,
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
        else if(padding_mode == same)
        {
            return {t,
                    {input.lens()[0],
                     input.lens()[1],
                     static_cast<std::size_t>(
                         std::ceil(static_cast<double>(input.lens()[2]) / stride[0])),
                     static_cast<std::size_t>(
                         std::ceil(static_cast<double>(input.lens()[3]) / stride[1]))}};
        }
        else if(padding_mode == valid)
        {
            return {t,
                    {
                        input.lens()[0],
                        input.lens()[1],
                        std::size_t(std::max<std::ptrdiff_t>(
                            1,
                            std::ptrdiff_t(std::floor((input.lens()[2] - lengths[0]) /
                                                      static_cast<float>(stride[0]))) +
                                1)),
                        std::size_t(std::max<std::ptrdiff_t>(
                            1,
                            std::ptrdiff_t(std::floor((input.lens()[3] - lengths[1]) /
                                                      static_cast<float>(stride[1]))) +
                                1)),
                    }};
        }
        else
        {
            MIGRAPHX_THROW("Invalid padding mode");
        }
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

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.alpha, "alpha"));
    }
};

struct elu
{
    std::string name() const { return "elu"; }
    float alpha;
    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this}.has(1);
        return inputs.front();
    }

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.alpha, "alpha"));
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
            MIGRAPHX_THROW("Permutation has wrong number of axes");
        }
        std::vector<int64_t> axes(dims.size());
        std::iota(axes.begin(), axes.end(), 0);
        if(!std::is_permutation(axes.begin(), axes.end(), dims.begin()))
        {
            MIGRAPHX_THROW("Invalid permutation");
        }
        std::vector<size_t> output_lens(input_lens.size());
        std::vector<size_t> output_strides(input_lens.size());
        for(std::size_t i = 0; i < output_lens.size(); i++)
        {
            output_lens[i]    = input_lens[dims[i]];
            output_strides[i] = input_strides[dims[i]];
        }
        return {t, output_lens, output_strides};
    }
    argument compute(shape output_shape, std::vector<argument> args) const
    {
        return {std::move(output_shape), std::move(args.front().data)};
    }
    int output_alias(const std::vector<shape>&) const { return 0; }
};

/// The contiguous operator takes a non-standard input tensor and returns
/// the same tensor but in standard form. For example, if input tensor A which has lens = (4,5)
/// is first transposed, i.e. lens = (5,4), this tensor's data layout remained the same
/// during the transpose operation; only it's shape lengths and strides were changed.
/// This leaves the tensor in a non-standard form. The contiguous operator copies the
/// underlying data such that resulting tensor is returned to a standard form.
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
    argument compute(const shape& output_shape, std::vector<argument> args) const
    {
        assert(output_shape.standard());
        argument result{output_shape};
        visit_all(result, args[0])([&](auto output, auto input) {
            shape_for_each(output.get_shape(), [&](const auto& idx) {
                output(idx.begin(), idx.end()) = input(idx.begin(), idx.end());
            });
        });
        return result;
    }
};

struct concat
{
    std::size_t axis = 0;
    std::string name() const { return "concat"; }
    std::vector<std::size_t> compute_offsets(const shape& output_shape,
                                             const std::vector<argument>& args) const
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
            MIGRAPHX_THROW("Number of input tensors should exceed 0");
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
                    MIGRAPHX_THROW("Non-axis dimensions should match");
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
    argument compute(const shape& output_shape, std::vector<argument> args) const
    {
        argument result{output_shape};
        std::vector<std::size_t> coffsets = compute_offsets(output_shape, args);
        for(std::size_t l = 0; l < args.size(); l++)
        {
            auto argl             = args[l];
            std::size_t nelements = argl.get_shape().elements();
            visit_all(result, argl)([&](auto output, auto input) {
                auto slice_shape =
                    shape{output_shape.type(), input.get_shape().lens(), output_shape.strides()};
                auto slice = make_view(slice_shape, output.data() + coffsets[l]);
                // cppcheck-suppress useStlAlgorithm
                for(std::size_t i = 0; i < nelements; i++)
                {
                    slice[i] = input[i];
                }
            });
        }
        return result;
    }
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
        if(starts.size() != axes.size() || axes.size() != ends.size())
        {
            MIGRAPHX_THROW("inconsistent sizes");
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
    argument compute(shape output_shape, std::vector<argument> args) const
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
            MIGRAPHX_THROW("squeeze axis dimension should be equal to 1");
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
    argument compute(shape output_shape, std::vector<argument> args) const
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
    argument compute(shape output_shape, std::vector<argument> args) const
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
            MIGRAPHX_THROW("Dimensions for reshape can only have one -1 dim");
        for(std::size_t i = 0; i < dims.size(); i++)
        {
            if(dims[i] == 0)
                rdims[i] = idims[i];

            // since rdims using size_t type, -1 is the max value
            // is size_t that cause later compuation incorrect
            if(dims[i] == -1)
                rdims[i] = 1;
        }
        if(n_neg_dims > 0)
        {
            size_t missing_dim =
                inputs.front().elements() /
                std::accumulate(rdims.begin(), rdims.end(), 1, std::multiplies<int64_t>());
            for(std::size_t i = 0; i < rdims.size(); i++)
            {
                if(dims[i] == -1)
                    rdims[i] = missing_dim;
            }
        }

        shape s{inputs.front().type(), rdims};
        if(s.elements() != inputs.front().elements())
            MIGRAPHX_THROW("Wrong number of elements for reshape");
        return s;
    }
    argument compute(shape output_shape, std::vector<argument> args) const
    {
        return {std::move(output_shape), std::move(args.front().data)};
    }
    int output_alias(const std::vector<shape>&) const { return 0; }
};

struct pad
{
    std::vector<int64_t> pads;
    float value = 0.0f;
    enum pad_op_mode_t
    {
        constant_pad,
        reflect_pad,
        edge_pad
    };
    pad_op_mode_t mode = constant_pad;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.mode, "mode"), f(self.pads, "pads"), f(self.value, "value"));
    }

    std::string name() const { return "pad"; }
    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this}.has(1);
        auto&& idims = inputs.front().lens();
        std::vector<std::size_t> rdims(idims.begin(), idims.end());
        std::size_t num_dims = rdims.size();

        for(std::size_t i = 0; i < num_dims; i++)
        {
            rdims[i] += pads[i] + pads[i + num_dims];
        }

        shape s{inputs.front().type(), rdims};
        return s;
    }
};

struct as_shape
{
    shape s;
    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.s, "shape"));
    }

    std::string name() const { return "as_shape"; }
    shape compute_shape(const std::vector<shape>& inputs) const
    {
        check_shapes{inputs, *this}.has(1).standard();
        assert(inputs.front().elements() == s.elements());
        return s;
    }
    argument compute(shape output_shape, std::vector<argument> args) const
    {
        return {std::move(output_shape), std::move(args.front().data)};
    }
    int output_alias(const std::vector<shape>&) const { return 0; }
};

struct gather
{
    int axis = 0;
    std::string name() const { return "gather"; }

    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this}.has(2);
        auto lens = inputs[0].lens();
        int n_dim = static_cast<int>(lens.size());
        if(axis >= n_dim || axis < -n_dim)
        {
            MIGRAPHX_THROW("Gather: axis is out of range.");
        }

        // negative axis means counting dimensions from back
        int axis_index = (axis < 0) ? (n_dim + axis) : axis;

        auto type = inputs[0].type();
        lens.erase(lens.begin() + axis_index);
        if(!inputs[1].scalar())
        {
            auto ind_lens = inputs[1].lens();
            lens.insert(lens.begin() + axis_index, ind_lens.begin(), ind_lens.end());
        }

        // for scalar output
        if(lens.empty())
        {
            return {type};
        }

        return {type, lens};
    }

    argument compute(const shape& output_shape, std::vector<argument> args) const
    {
        argument result{output_shape};
        // negative axis means counting dimensions from back
        int axis_index =
            (axis < 0) ? static_cast<int>(args[0].get_shape().lens().size() + axis) : axis;

        // max dimension in axis
        visit_all(result, args[0])([&](auto output, auto data) {
            args[1].visit([&](auto indices) {
                if(output_shape.scalar())
                {
                    output[0] = data[indices.front()];
                }
                else
                {
                    auto out_lens        = data.get_shape().lens();
                    out_lens[axis_index] = indices.get_shape().elements();
                    migraphx::shape out_comp_shape{data.get_shape().type(), out_lens};
                    shape_for_each(out_comp_shape, [&](const auto& out_idx) {
                        auto data_idx        = out_idx;
                        data_idx[axis_index] = indices[data_idx[axis_index]];
                        output[out_comp_shape.index(out_idx.begin(), out_idx.end())] =
                            data(data_idx.begin(), data_idx.end());
                    });
                }
            });
        });

        return result;
    }
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

        // according to the specification of the numpy.matmul()
        // inputs with the shape dims more than 2 are acceptable
        // as long as dim values are the same in the two inputs
        if(!std::equal(a.lens().rbegin() + 2, a.lens().rend(), b.lens().rbegin() + 2))
        {
            MIGRAPHX_THROW("DOT: dim values mismatch");
        }

        std::size_t dim_0 = a.lens().size() - 2;
        std::size_t dim_1 = a.lens().size() - 1;
        if(a.lens()[dim_1] != b.lens()[dim_0])
            MIGRAPHX_THROW("Inner dimensions do not match: {" + to_string_range(a.lens()) +
                           "} x {" + to_string_range(b.lens()) + "}");
        auto out_lens   = a.lens();
        out_lens[dim_1] = b.lens()[dim_1];
        return {t, out_lens};
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
    argument compute(shape output_shape, std::vector<argument> args) const
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

struct log : unary
{
    std::string name() const { return "log"; }
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

struct sinh : unary
{
    std::string name() const { return "sinh"; }
};

struct cosh : unary
{
    std::string name() const { return "cosh"; }
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

struct logsoftmax
{
    int axis = 1;
    std::string name() const { return "logsoftmax"; }
    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs}.has(1);
        if(axis < 0 || axis > inputs[0].lens().size())
        {
            MIGRAPHX_THROW("LogSoftMax: input axis value " + std::to_string(axis) +
                           " is out of range");
        }
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
            MIGRAPHX_THROW("axis for flatten must be less than tensor rank");
        }
        auto x =
            std::accumulate(lens.begin(), lens.begin() + axis, std::size_t{1}, std::multiplies<>{});
        auto y =
            std::accumulate(lens.begin() + axis, lens.end(), std::size_t{1}, std::multiplies<>{});
        return {inputs.at(0).type(), {x, y}};
    }
    argument compute(shape output_shape, std::vector<argument> args) const
    {
        return {std::move(output_shape), std::move(args.front().data)};
    }
    int output_alias(const std::vector<shape>&) const { return 0; }
};

/// The broadcast operator performs the numpy-style broadcasting of an axis of a given tensor. This
/// is achieved primarily by setting the stride of the broadcasted axis to zero. Linear indicies are
/// computed from multi-indicies by computing the inner product on the multi-index with the strides.
/// For example, if we have a tensor A(2,3) it has lengths of (2,3) and strides of (3,1). If we want
/// to compute the linear offset that corresponds to the element on the 2nd row (i = 1) and 3rd
/// column (j = 2), we compute the following inner product (1,2) dot (3, 1) = 1*3 + 2*1 = 5. It is
/// obvious from there that we can negate the effects of a given axis by setting the stride of that
/// axis to zero.
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
                MIGRAPHX_THROW("when broadcasting tensor of size 1, axis should be 0");
            return {t, broadcast_shape.lens(), std::move(bcast_strides)};
        }
        else
        {
            assert(broadcast_shape.lens().size() - axis >= input.lens().size());
            if(!std::equal(
                   input.lens().begin(), input.lens().end(), broadcast_shape.lens().begin() + axis))
                MIGRAPHX_THROW("when broadcasting success sizes must match");
            std::copy(input.strides().begin(), input.strides().end(), bcast_strides.begin() + axis);
            return {t, broadcast_shape.lens(), std::move(bcast_strides)};
        }
    }
    argument compute(shape output_shape, std::vector<argument> args) const
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
            MIGRAPHX_THROW("inputs dimensions should be > 0");

        if(input.lens().size() > output_lens.size())
            MIGRAPHX_THROW("inputs dimensions should <= output size");

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
    argument compute(shape output_shape, std::vector<argument> args) const
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
        auto t = inputs.at(0).type();
        std::vector<std::size_t> strides(scalar_bcast.lens().size(), 0);
        return {t, scalar_bcast.lens(), strides};
    }

    argument compute(shape output_shape, std::vector<argument> args) const
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

struct max : binary
{
    std::string name() const { return "max"; }
};

struct min : binary
{
    std::string name() const { return "min"; }
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
    argument compute(const shape&, const std::vector<argument>& args) const
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
    argument compute(const shape&, const std::vector<argument>&) const { return {s, nullptr}; }
};

// indicate rnn computation direction
enum class rnn_direction
{
    forward,
    reverse,
    bidirectional,
};

struct rnn
{
    std::size_t hidden_size = 1;
    std::vector<operation> actv_funcs{tanh{}, tanh{}};
    rnn_direction direction = rnn_direction::forward;
    float clip              = 0.0f;

    std::string name() const { return "rnn"; }
    shape compute_shape(std::vector<shape> inputs) const
    {
        auto in_dims     = inputs[0].lens();
        auto hidden_dims = inputs[2].lens();
        if(hidden_size != hidden_dims[2])
        {
            MIGRAPHX_THROW("RNN: hidden size mismatch in attribute and input");
        }

        std::size_t num_directions = 1;
        if(direction == rnn_direction::bidirectional)
        {
            num_directions = 2;
        }

        if(num_directions != hidden_dims[0])
        {
            MIGRAPHX_THROW("RNN: num_direction mismatch in attribute and input");
        }

        std::vector<std::size_t> out_dims(in_dims);
        out_dims.insert(out_dims.begin() + 1, num_directions);
        out_dims.back() = hidden_size;

        return {inputs[0].type(), out_dims};
    }
};

struct rnn_last_output
{
    std::string name() const { return "rnn_last_output"; }
    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this}.has(1);
        auto dims = inputs[0].lens();

        // remove the first dimension, remaing are output shape
        dims.erase(dims.begin());
        return {inputs[0].type(), dims};
    }
};

struct gru
{
    std::size_t hidden_size = 1;
    std::vector<operation> actv_funcs{sigmoid{}, tanh{}};
    rnn_direction direction = rnn_direction::forward;
    float clip              = 0.0f;
    int linear_before_reset = 0;

    std::string name() const { return "gru"; }
    shape compute_shape(std::vector<shape> inputs) const
    {
        auto in_dims     = inputs[0].lens();
        auto hidden_dims = inputs[2].lens();
        if(hidden_size != hidden_dims[2])
        {
            MIGRAPHX_THROW("GRU: hidden size mismatch in attribute and input");
        }

        std::size_t num_directions = 1;
        if(direction == rnn_direction::bidirectional)
        {
            num_directions = 2;
        }

        if(num_directions != hidden_dims[0])
        {
            MIGRAPHX_THROW("GRU: num_direction does not match the direction attribute");
        }

        std::vector<std::size_t> out_dims(in_dims);
        out_dims.insert(out_dims.begin() + 1, num_directions);
        out_dims.back() = hidden_size;

        return {inputs[0].type(), out_dims};
    }
};

struct lstm
{
    std::size_t hidden_size = 1;
    std::vector<operation> actv_funcs{sigmoid{}, tanh{}, tanh{}};
    rnn_direction direction = rnn_direction::forward;
    float clip              = 0.0f;
    int input_forget        = 0;

    std::string name() const { return "lstm"; }
    shape compute_shape(std::vector<shape> inputs) const
    {
        auto in_dims     = inputs[0].lens();
        auto hidden_dims = inputs[2].lens();
        if(hidden_size != hidden_dims[2])
        {
            MIGRAPHX_THROW("LSTM: hidden size mismatch in attribute and input");
        }

        std::size_t num_directions = 1;
        if(direction == rnn_direction::bidirectional)
        {
            num_directions = 2;
        }

        if(num_directions != hidden_dims[0])
        {
            MIGRAPHX_THROW("LSTM: num_direction does not match the direction attribute");
        }

        std::vector<std::size_t> out_dims(in_dims);
        out_dims.insert(out_dims.begin() + 1, num_directions);
        out_dims.back() = hidden_size;

        return {inputs[0].type(), out_dims};
    }
};

struct lstm_last_cell_output
{
    std::string name() const { return "lstm_last_cell_output"; }
    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this}.has(1);
        auto dims = inputs[0].lens();

        // remove the first dimension, remaing are output shape
        dims.erase(dims.begin());
        return {inputs[0].type(), dims};
    }
};

struct undefined
{
    std::string name() const { return "undefined"; }
    shape compute_shape(const std::vector<shape>& inputs) const
    {
        check_shapes{inputs, *this}.has(0);
        return {};
    }

    argument compute(const shape&, const std::vector<argument>&) const { return {{}, nullptr}; }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
