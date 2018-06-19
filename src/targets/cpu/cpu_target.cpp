
#include <rtg/cpu/cpu_target.hpp>
#include <rtg/instruction.hpp>
#include <rtg/dfor.hpp>
#include <rtg/operators.hpp>

namespace rtg {
namespace cpu {

template <typename T>
T zero(const T&)
{
    return T(0);
}

struct cpu_convolution
{
    convolution op;

    std::string name() const { return "cpu::convolution"; }
    shape compute_shape(std::vector<shape> inputs) const { return op.compute_shape(inputs); }
    argument compute(shape output_shape, std::vector<argument> args) const
    {
        argument result{output_shape};
        visit_all(result, args[0], args[1])([&](auto output, auto input, auto weights) {
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

struct cpu_transpose
{
    transpose op;
   
    std::string name() const { return "cpu::transpose"; } 
    shape compute_shape(std::vector<shape> inputs) const { return op.compute_shape(inputs); }
    argument compute(shape output_shape, std::vector<argument> args) const
    {
        return {output_shape, std::move(args.front().data)};
    }
};

struct cpu_contiguous
{
    contiguous op;
    std::string name() const { return "cpu::contiguous"; }
    shape compute_shape(std::vector<shape> inputs) const
    {
        return op.compute_shape(inputs);
    }
    argument compute(shape output_shape, std::vector<argument> args) const
    {
        argument result{output_shape};
        visit_all(result, args[0])([&](auto output, auto input) {
            auto input_shape = args[0].get_shape();
            auto ndim = output_shape.lens().size();
            using value_type = typename decltype(input)::value_type;
            value_type* ptr = static_cast<value_type*>(output.data());
            if (ndim == 2) {
                dfor(input_shape.lens()[0],
                     input_shape.lens()[1])(
                    [&](std::size_t i0, std::size_t i1) {
                        *ptr++ = input(i0,i1);
                    });
            }
            else if (ndim == 3) {
                dfor(input_shape.lens()[0],
                     input_shape.lens()[1],
                     input_shape.lens()[2])(
                    [&](std::size_t i0, std::size_t i1, std::size_t i2) {
                        *ptr++ = input(i0,i1,i2);
                    });
            }
            else if (ndim == 4) {
                dfor(input_shape.lens()[0],
                     input_shape.lens()[1],
                     input_shape.lens()[2],
                     input_shape.lens()[3])(
                    [&](std::size_t i0, std::size_t i1, std::size_t i2, std::size_t i3) {
                        *ptr++ = input(i0,i1,i2,i3);
                    });
            }
            else if (ndim == 5) {
                dfor(input_shape.lens()[0],
                     input_shape.lens()[1],
                     input_shape.lens()[2],
                     input_shape.lens()[3],
                     input_shape.lens()[4])(
                    [&](std::size_t i0, 
                        std::size_t i1, 
                        std::size_t i2, 
                        std::size_t i3, 
                        std::size_t i4) {
                        *ptr++ = input(i0,i1,i2,i3,i4);
                    });
            }
            else if (ndim == 6) {
                dfor(input_shape.lens()[0],
                     input_shape.lens()[1],
                     input_shape.lens()[2],
                     input_shape.lens()[3],
                     input_shape.lens()[4],
                     input_shape.lens()[5])(
                    [&](std::size_t i0, 
                        std::size_t i1, 
                        std::size_t i2, 
                        std::size_t i3, 
                        std::size_t i4, 
                        std::size_t i5) {
                        *ptr++ = input(i0,i1,i2,i3,i4,i5);
                    });
            }
        });
        return result;
    }
};

struct cpu_reshape
{
    reshape op;
    std::string name() const { return "cpu::reshape"; }
    shape compute_shape(std::vector<shape> inputs) const { return op.compute_shape(inputs); }

    argument compute(shape output_shape, std::vector<argument> args) const
    {
        return {output_shape, std::move(args.front().data)};
    }
};

struct cpu_gemm
{
    gemm op;
    std::string name() const { return "cpu::gemm"; }
    shape compute_shape(std::vector<shape> inputs) const { return op.compute_shape(inputs); }

    argument compute(shape output_shape, std::vector<argument> args) const
    {
        argument result{output_shape};
        visit_all(result, args[0], args[1])([&](auto cmat, auto amat, auto bmat) {
            auto m = amat.get_shape().lens()[0];
            auto n = bmat.get_shape().lens()[1];
            auto k = bmat.get_shape().lens()[0];

            auto a = amat.data();
            auto b = bmat.data();
            auto c = cmat.data();
            for(int ii = 0; ii < m; ii++)
            {
                for(int jj = 0; jj < n; jj++)
                {
                    c[ii * n + jj] = 0;
                }
            }
            for(int ii = 0; ii < m; ii++)
            {
                for(int kk = 0; kk < k; kk++)
                {
                    auto aik  = a[ii * k + kk];
                    auto* bkj = &b[kk * n];
                    auto* cij = &c[ii * n];
                    for(int jj = 0; jj < n; jj++, cij++, bkj++)
                    {
                        *cij += aik * (*bkj);
                    }
                }
            }
        });
        return result;
    }
};

struct identity_op
{
    std::string name() const { return "cpu::identity"; }
    auto fcn() const
    {
        return [](auto x) { return x; };
    }
};

struct abs_op
{
    std::string name() const { return "cpu::abs"; }
    auto fcn() const
    {
        return [](auto x) { return std::abs(x); };
    }
};

struct exp_op
{
    std::string name() const { return "cpu::exp"; }
    auto fcn() const
    {
        return [](auto x) { return std::exp(x); };
    }
};

struct sin_op
{
    std::string name() const { return "cpu::sin"; }
    auto fcn() const
    {
        return [](auto x) { return std::sin(x); };
    }
};

struct cos_op
{
    std::string name() const { return "cpu::cos"; }
    auto fcn() const
    {
        return [](auto x) { return std::cos(x); };
    }
};

struct tan_op
{
    std::string name() const { return "cpu::tan"; }
    auto fcn() const
    {
        return [](auto x) { return std::tan(x); };
    }
};

struct asin_op
{
    std::string name() const { return "cpu::asin"; }
    auto fcn() const
    {
        return [](auto x) { return std::asin(x); };
    }
};

struct acos_op
{
    std::string name() const { return "cpu::acos"; }
    auto fcn() const
    {
        return [](auto x) { return std::acos(x); };
    }
};

struct atan_op
{
    std::string name() const { return "cpu::atan"; }
    auto fcn() const
    {
        return [](auto x) { return std::atan(x); };
    }
};

struct tanh_op
{
    std::string name() const { return "cpu::tanh"; }
    auto fcn() const
    {
        return [](auto x) { return std::tanh(x); };
    }
};

struct sigmoid_op
{
    std::string name() const { return "cpu::sigmoid"; }
    auto fcn() const
    {
        return [](auto x) { return 1.f / (1.f + std::exp(-x)); };
    }
};

struct neg_op
{
    std::string name() const { return "cpu::neg"; }
    auto fcn() const
    {
        return [](auto x) { return -x; };
    }
};

struct relu_op
{
    std::string name() const { return "cpu::relu"; }
    auto fcn() const
    {
        return [](auto x) { return x > 0 ? x : 0; };
    }
};

template <typename Op>
struct cpu_unary
{
    Op op;
    std::string name() const { return op.name(); }
    shape compute_shape(std::vector<shape> inputs) const { return inputs.front(); }
    argument compute(shape output_shape, std::vector<argument> args) const
    {
        argument result{output_shape};
        result.visit([&](auto output) {
            args[0].visit([&](auto input) {
                std::transform(input.begin(), input.end(), output.begin(), op.fcn());
            });
        });
        return result;
    }
};

struct softmax2d
{
    std::string name() const { return "cpu::softmax2d"; }
    shape compute_shape(std::vector<shape> inputs) const { return inputs.front(); }
    argument compute(shape output_shape, std::vector<argument> args) const
    {
        argument result{output_shape};
        visit_all(result, args[0])([&](auto output, auto input) {
            using value_type = typename decltype(input)::value_type;
            auto nb          = input.get_shape().lens()[0];
            auto nc          = input.get_shape().lens()[1];
            auto nh          = input.get_shape().lens()[2];
            auto nw          = input.get_shape().lens()[3];
            dfor(nb, nh, nw)([&](std::size_t b, std::size_t i, std::size_t j) {
                value_type cmax = std::numeric_limits<value_type>::lowest();
                for(int c = 0; c < nc; c++)
                {
                    cmax = std::max(cmax, input(b, c, i, j));
                }
                for(int c = 0; c < nc; c++)
                {
                    output(b, c, i, j) = std::exp(input(b, c, i, j) - cmax);
                }
                value_type sum = value_type(0);
                for(int c = 0; c < nc; c++)
                {
                    sum += output(b, c, i, j);
                }
                for(int c = 0; c < nc; c++)
                {
                    output(b, c, i, j) = output(b, c, i, j) / sum;
                }
            });
        });
        return result;
    }
};

struct add_op
{
    std::string name() const { return "add"; }
    auto fcn() const
    {
        return [](auto x, auto y) { return x + y; };
    }
};

struct sub_op
{
    std::string name() const { return "sub"; }
    auto fcn() const
    {
        return [](auto x, auto y) { return x - y; };
    }
};

struct mul_op
{
    std::string name() const { return "mul"; }
    auto fcn() const
    {
        return [](auto x, auto y) { return x * y; };
    }
};

struct div_op
{
    std::string name() const { return "div"; }
    auto fcn() const
    {
        return [](auto x, auto y) { return x / y; };
    }
};

template <typename Op>
struct cpu_binary
{
    Op op;
    std::string name() const { return op.name(); }
    shape compute_shape(std::vector<shape> inputs) const { return inputs.front(); }
    argument compute(shape output_shape, std::vector<argument> args) const
    {
        argument result{output_shape};
        visit_all(result, args[0], args[1])([&](auto output, auto input1, auto input2) {
            std::transform(input1.begin(), input1.end(), input2.begin(), output.begin(), op.fcn());
        });
        return result;
    }
};

struct cpu_apply
{
    program* prog;
    std::unordered_map<std::string, std::function<void(instruction_ref)>> apply_map{};

    template<class T>
    auto simple_op()
    {
        return [this](instruction_ref ins)
        {
            apply_simple_op<T>(ins);
        };
    }

    template<class T, class Op>
    auto extend_op()
    {
        return [this](instruction_ref ins)
        {
            apply_extend_op<T, Op>(ins);
        };
    }

    void init()
    {
        apply_map["convolution"] = extend_op<cpu_convolution, convolution>();
        apply_map["gemm"] = extend_op<cpu_gemm, gemm>();
        apply_map["reshape"] = extend_op<cpu_reshape, reshape>();
        apply_map["contiguous"] = extend_op<cpu_contiguous, contiguous>();
        apply_map["transpose"] = extend_op<cpu_transpose, transpose>();
        
        apply_map["identity"] = simple_op<cpu_unary<identity_op>>();
        apply_map["tanh"] = simple_op<cpu_unary<tanh_op>>();
        apply_map["sigmoid"] = simple_op<cpu_unary<sigmoid_op>>();
        apply_map["exp"] = simple_op<cpu_unary<exp_op>>();
        apply_map["neg"] = simple_op<cpu_unary<neg_op>>();
        apply_map["sin"] = simple_op<cpu_unary<sin_op>>();
        apply_map["cos"] = simple_op<cpu_unary<cos_op>>();
        apply_map["tan"] = simple_op<cpu_unary<tan_op>>();

        apply_map["softmax"] = simple_op<softmax2d>();
    }

    void apply()
    {
        init();
        for(auto it = prog->begin(); it != prog->end(); it++)
        {
            if(it->op.name() == "activation")
            {
                apply_activation(it);
            } 
            else if(apply_map.count(it->op.name()) > 0)
            {
                apply_map.at(it->op.name())(it);
            }
        }
    }

    template<class T>
    void apply_simple_op(instruction_ref ins)
    {
        prog->replace_instruction(ins, T{}, ins->arguments);
    }

    template<class T, class Op>
    void apply_extend_op(instruction_ref ins)
    {
        auto&& op = any_cast<Op>(ins->op);
        prog->replace_instruction(ins, T{op}, ins->arguments);
    }

    void apply_activation(instruction_ref ins)
    {
        auto&& op = any_cast<activation>(ins->op);
        if(op.mode == "relu")
            prog->replace_instruction(ins, cpu_unary<relu_op>{}, ins->arguments);
    }
};

std::string cpu_target::name() const { return "cpu"; }

void cpu_target::apply(program& p) const { cpu_apply{&p}.apply(); }

} // namespace cpu

} // namespace rtg
