#ifndef RTG_GUARD_OPERATORS_HPP
#define RTG_GUARD_OPERATORS_HPP

#include <rtg/operand.hpp>
#include <rtg/stringutils.hpp>
#include <cmath>

namespace rtg {

struct not_computable
{
    argument compute(std::vector<argument>) const
    {
        throw std::runtime_error("not computable");
    }
};

struct convolution
{
    std::array<std::size_t, 2> padding = {0, 0};
    std::array<std::size_t, 2> stride = {1, 1};
    std::array<std::size_t, 2> dilation = {1, 1};
    std::string name() const
    {
        return "convolution[padding={" + to_string(padding) + 
            "}, stride={" + to_string(stride) +
            "}, dilation={" + to_string(dilation) +
            "}]";
    }
    shape compute_shape(std::vector<shape> inputs) const
    {
        if(inputs.size() != 2) throw std::runtime_error("Wrong number of arguments");
        const shape& input = inputs.at(0);
        const shape& weights = inputs.at(1);
        if(input.type() != weights.type()) throw std::runtime_error("Type doesn't match");
        if(input.lens().size() != weights.lens().size()) throw std::runtime_error("Dimensions don't match");
        if(input.lens().size() != 4) throw std::runtime_error("Only 4d convolution supported"); 

        auto t = input.type();
        return {t, {
            input.lens()[0],
            weights.lens()[0],
            std::size_t(std::max<std::ptrdiff_t>(
                1, (input.lens()[2] - (1 + dilation[0] * (weights.lens()[2] - 1)) + 2 * padding[0]) / stride[0] + 1)),
            std::size_t(std::max<std::ptrdiff_t>(
                1, (input.lens()[3] - (1 + dilation[1] * (weights.lens()[3] - 1)) + 2 * padding[1]) / stride[1] + 1)),
        }};
    }

    argument compute(std::vector<argument>) const
    {
        throw std::runtime_error("not computable");
    }
};

struct pooling
{
    std::string mode;
    std::array<std::size_t, 2> padding = {0, 0};
    std::array<std::size_t, 2> stride = {1, 1};
    std::array<std::size_t, 2> lengths = {1, 1};
    std::string name() const
    {
        return "pooling:" + mode + "[padding={" + to_string(padding) + 
            "}, stride={" + to_string(stride) +
            "}, lengths={" + to_string(lengths) +
            "}]";
    }
    shape compute_shape(std::vector<shape> inputs) const
    {
        if(inputs.empty()) throw std::runtime_error("Wrong number of arguments");
        const shape& input = inputs.at(0);    
        if(input.lens().size() != 4) throw std::runtime_error("Only 4d pooling supported"); 

        auto t = input.type();
        return {t, {
            input.lens()[0],
            input.lens()[1],
            std::size_t(std::max<std::ptrdiff_t>(
                1, std::ceil((input.lens()[3] + 2 * padding[0] - lengths[0]) / static_cast<float>(stride[0])) + 1)),
            std::size_t(std::max<std::ptrdiff_t>(
                1, std::ceil((input.lens()[4] + 2 * padding[1] - lengths[1]) / static_cast<float>(stride[1])) + 1)),
        }};
    }

    argument compute(std::vector<argument>) const
    {
        throw std::runtime_error("not computable");
    }
};


struct activation
{
    std::string mode;
    std::string name() const
    {
        return "activation:" + mode;
    }
    shape compute_shape(std::vector<shape> inputs) const
    {
        if(inputs.empty()) throw std::runtime_error("Wrong number of arguments");
        return inputs.front();
    }

    argument compute(std::vector<argument>) const
    {
        throw std::runtime_error("not computable");
    }
};


} // namespace rtg

#endif
