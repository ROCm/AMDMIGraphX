#ifndef RTG_GUARD_OPERATORS_HPP
#define RTG_GUARD_OPERATORS_HPP

#include <rtg/operation.hpp>
#include <rtg/stringutils.hpp>
#include <rtg/streamutils.hpp>
#include <cmath>

namespace rtg {

struct not_computable
{
    argument compute(std::vector<argument>) const { RTG_THROW("not computable"); }
};

struct convolution
{
    std::array<std::size_t, 2> padding  = {{0, 0}};
    std::array<std::size_t, 2> stride   = {{1, 1}};
    std::array<std::size_t, 2> dilation = {{1, 1}};
    std::string name() const
    {
        return "convolution";
    }
    shape compute_shape(std::vector<shape> inputs) const
    {
        if(inputs.size() != 2)
            RTG_THROW("Wrong number of arguments");
        const shape& input   = inputs.at(0);
        const shape& weights = inputs.at(1);
        if(input.type() != weights.type())
            RTG_THROW("Type doesn't match");
        if(input.lens().size() != weights.lens().size())
            RTG_THROW("Dimensions don't match");
        if(input.lens().size() != 4)
            RTG_THROW("Only 4d convolution supported");

        auto t = input.type();
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

    argument compute(std::vector<argument>) const { RTG_THROW("not computable"); }

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

struct pooling
{
    std::string mode;
    std::array<std::size_t, 2> padding = {{0, 0}};
    std::array<std::size_t, 2> stride  = {{1, 1}};
    std::array<std::size_t, 2> lengths = {{1, 1}};
    std::string name() const
    {
        return "pooling";
    }
    shape compute_shape(std::vector<shape> inputs) const
    {
        if(inputs.empty())
            RTG_THROW("Wrong number of arguments");
        const shape& input = inputs.at(0);
        if(input.lens().size() != 4)
            RTG_THROW("Only 4d pooling supported");

        auto t = input.type();
        return {t,
                {
                    input.lens()[0],
                    input.lens()[1],
                    std::size_t(std::max<std::ptrdiff_t>(
                        1,
                        std::ceil((input.lens()[3] + 2 * padding[0] - lengths[0]) /
                                  static_cast<float>(stride[0])) +
                            1)),
                    std::size_t(std::max<std::ptrdiff_t>(
                        1,
                        std::ceil((input.lens()[4] + 2 * padding[1] - lengths[1]) /
                                  static_cast<float>(stride[1])) +
                            1)),
                }};
    }

    argument compute(std::vector<argument>) const { RTG_THROW("not computable"); }

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
        if(inputs.empty())
            RTG_THROW("Wrong number of arguments");
        return inputs.front();
    }

    argument compute(std::vector<argument>) const { RTG_THROW("not computable"); }
    friend std::ostream& operator<<(std::ostream& os, const activation& op)
    {
        os << op.name() << ":" << op.mode;
        return os;
    }
};

struct reshape
{
    std::vector<int64_t> dims;
    std::string name() const { return "reshape"; }
    shape compute_shape(std::vector<shape> inputs) const
    {
        if(inputs.empty())
            RTG_THROW("Wrong number of arguments");
        auto&& idims = inputs.front().lens();
        std::vector<std::size_t> rdims(dims.begin(), dims.end());
        for(std::size_t i = 0; i < dims.size(); i++)
        {
            if(dims[i] == 0)
                rdims[i] = idims[i];
        }
        if(dims.back() == -1)
        {
            rdims.pop_back();
            std::copy(idims.begin() + rdims.size(), idims.end(), std::back_inserter(rdims));
        }
        return {inputs.front().type(), rdims};
    }

    argument compute(std::vector<argument>) const { RTG_THROW("not computable"); }

    friend std::ostream& operator<<(std::ostream& os, const reshape& op)
    {
        os << op.name() << "[";
        os << "dims={" << stream_range(op.dims) << "}, ";
        os << "]";
        return os;
    }
};

} // namespace rtg

#endif
