
#include <rtg/cpu/cpu_target.hpp>
#include <rtg/instruction.hpp>
#include <rtg/dfor.hpp>
#include <rtg/operators.hpp>

namespace rtg {
namespace cpu {

struct cpu_convolution
{
    convolution op;

    std::string name() const { return "cpu::convolution"; }
    shape compute_shape(std::vector<shape> inputs) const { return op.compute_shape(inputs); }
    argument compute(shape output_shape, std::vector<argument> args) const
    {
        argument result{output_shape};
        visit_all(result, args[0], args[1])([&](auto output, auto input, auto weights) {
            auto in_n = input.get_shape().lens()[0];
            auto in_c = input.get_shape().lens()[1];
            auto in_h = input.get_shape().lens()[2];
            auto in_w = input.get_shape().lens()[3];

            auto wei_c = weights.get_shape().lens()[1];
            auto wei_h = weights.get_shape().lens()[2];
            auto wei_w = weights.get_shape().lens()[3];

            dfor(output_shape.lens()[0],
                 output_shape.lens()[1],
                 output_shape.lens()[2],
                 output_shape.lens()[3])(
                [&](std::size_t o, std::size_t w, std::size_t i, std::size_t j) {
                    const int start_x = i * op.stride[0] - op.padding[0];
                    const int start_y = j * op.stride[1] - op.padding[1];

                    double acc = 0;
                    dfor(wei_c, wei_h, wei_w)([&](std::size_t k, std::size_t x, std::size_t y) {
                        const int in_x = start_x + x;
                        const int in_y = start_y + y;
                        if(in_x >= 0 && in_x < in_h && in_y >= 0 && in_y < in_w)
                        {
                            acc += input(o, k, in_x, in_y) * weights(w, k, x, y);
                        }
                    });
                    output(o, w, i, j) = acc;
                });
        });
        return result;
    }
};

struct relu
{
    std::string name() const { return "cpu::relu"; }
    shape compute_shape(std::vector<shape> inputs) const { return inputs.front(); }

    argument compute(shape output_shape, std::vector<argument> args) const
    {
        argument result{output_shape};
        result.visit([&](auto output) {
            args[0].visit([&](auto input) {
                std::transform(input.begin(), input.end(), output.begin(), [](auto x) {
                    return x > 0 ? x : 0;
                });
            });
        });
        return result;
    }
};

struct cpu_apply
{
    program* prog;

    void apply()
    {
        for(auto it = prog->begin(); it != prog->end(); it++)
        {
            if(it->op.name() == "convolution")
            {
                apply_convolution(it);
            }
            else if(it->op.name() == "activation")
            {
                apply_activation(it);
            }
        }
    }

    void apply_convolution(instruction_ref ins)
    {
        auto&& op = any_cast<convolution>(ins->op);
        prog->replace_instruction(ins, cpu_convolution{op}, ins->arguments);
    }

    void apply_activation(instruction_ref ins)
    {
        auto&& op = any_cast<activation>(ins->op);
        if(op.mode == "relu")
            prog->replace_instruction(ins, relu{}, ins->arguments);
    }
};

std::string cpu_target::name() const { return "cpu"; }

void cpu_target::apply(program& p) const { cpu_apply{&p}.apply(); }

} // namespace cpu

} // namespace rtg
