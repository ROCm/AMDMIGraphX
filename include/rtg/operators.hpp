#ifndef RTG_GUARD_OPERATORS_HPP
#define RTG_GUARD_OPERATORS_HPP

#include <rtg/operand.hpp>
#include <rtg/stringutils.hpp>

namespace rtg {

struct not_computable
{
    argument compute(std::vector<argument>) const
    {
        throw "not computable";
    }
};

struct convolution : not_computable
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
        if(inputs.size() != 2) throw "Wrong number of arguments";
        const shape& input = inputs.at(0);
        const shape& weights = inputs.at(1);
        if(input.type() != weights.type()) throw "Type doesn't match";
        if(input.size() != weights.size()) throw "Dimensions don't match";
        if(input.size() != 4) throw "Only 4d convolution supported";        

        auto t = input.type();
        return {t, {
            input[0],
            weights[0],
            std::max<std::ptrdiff_t>(
                1, (input[2] - (1 + dilation[0] * (weights[2] - 1)) + 2 * padding[0]) / stride[0] + 1),
            std::max<std::ptrdiff_t>(
                1, (input[3] - (1 + dilation[1] * (weights[3] - 1)) + 2 * padding[1]) / stride[1] + 1),
        }};
    }
};

struct pooling : not_computable
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
        if(!inputs.empty()) throw "Wrong number of arguments";
        const shape& input = inputs.at(0);    
        if(input.size() != 4) throw "Only 4d pooling supported";        

        auto t = input.type();
        return {t, {
            input[0],
            input[1],
            std::max<std::ptrdiff_t>(
                1, std::ceil((input[3] + 2 * padding[0] - lengths[0]) / static_cast<float>(stride[0])) + 1),
            std::max<std::ptrdiff_t>(
                1, std::ceil((input[4] + 2 * padding[1] - lengths[1]) / static_cast<float>(stride[1])) + 1),
        }};
    }
};


struct activation : not_computable
{
    std::string mode;
    std::string name() const
    {
        return "activation:" + mode;
    }
    shape compute_shape(std::vector<shape> inputs) const
    {
        if(!inputs.empty()) throw "Wrong number of arguments";
        return inputs.front();
    }
};


} // namespace rtg

#endif
