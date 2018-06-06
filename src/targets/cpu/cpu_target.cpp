
#include <rtg/cpu/cpu_target.hpp>
#include <rtg/instruction.hpp>
#include <rtg/dfor.hpp>
#include <rtg/operators.hpp>

namespace rtg {
namespace cpu {

template <typename T>
T zero(const T& x) { return T(0); }

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

            dfor(in_n, in_c, in_h, in_w)(
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

struct cpu_gemm
{
    gemm op;
    std::string name() const { return "cpu::gemm"; }
    shape compute_shape(std::vector<shape> inputs) const
    {
        return op.compute_shape(inputs);
    }

    argument compute(shape output_shape, std::vector<argument> args) const 
    {
        argument C{output_shape};
        visit_all(C, args[0], args[1])([&](auto C, auto A, auto B) {
            auto M = A.get_shape().lens()[0];
            auto N = B.get_shape().lens()[1];
            auto K = B.get_shape().lens()[0];

            auto a = A.data();
            auto b = B.data();
            auto c = C.data();
            for (int ii = 0; ii < M; ii++) {
              for (int jj = 0; jj < N; jj++) {
                c[ii*N+jj] = 0;
              }
            }
            for (int ii = 0; ii < M; ii++) {
              for (int kk = 0; kk < K; kk++) {
                auto aik = a[ii*K+kk];
                auto* bkj = &b[kk*N];
                auto* cij = &c[ii*N];
                for (int jj = 0; jj < N; jj++, cij++, bkj++) {
                  *cij += aik*(*bkj);
                }
              }
            }
        });
        return C;
    }
};

struct identity_op
{
    std::string name() const {return "cpu::identity"; }
    auto fcn() const { return [](auto x) { return x; }; }
};

struct abs_op 
{
    std::string name() const {return "cpu::abs"; }
    auto fcn() const { return [](auto x) { return std::abs(x); }; }
};

struct exp_op 
{
    std::string name() const {return "cpu::exp"; }
    auto fcn() const { return [](auto x) { return std::exp(x); }; }
};

struct sin_op 
{
    std::string name() const {return "cpu::sin"; }
    auto fcn() const { return [](auto x) { return std::sin(x); }; }
};

struct cos_op 
{
    std::string name() const {return "cpu::cos"; }
    auto fcn() const { return [](auto x) { return std::cos(x); }; }
};

struct tan_op 
{
    std::string name() const {return "cpu::tan"; }
    auto fcn() const { return [](auto x) { return std::tan(x); }; }
};

struct asin_op 
{
    std::string name() const {return "cpu::asin"; }
    auto fcn() const { return [](auto x) { return std::asin(x); }; }
};

struct acos_op 
{
    std::string name() const {return "cpu::acos"; }
    auto fcn() const { return [](auto x) { return std::acos(x); }; }
};

struct atan_op 
{
    std::string name() const {return "cpu::atan"; }
    auto fcn() const { return [](auto x) { return std::atan(x); }; }
};

struct tanh_op
{
    std::string name() const {return "cpu::tanh"; }
    auto fcn() const { return [](auto x) { return std::tanh(x); }; }
};

struct sigmoid_op
{
    std::string name() const {return "cpu::sigmoid"; }
    auto fcn() const { return [](auto x) { return 1.f/(1.f + std::exp(-x)); }; }
};

struct neg_op
{
    std::string name() const {return "cpu::neg"; }
    auto fcn() const { return [](auto x) { return -x; }; }
};

struct relu_op
{
    std::string name() const {return "cpu::relu"; }
    auto fcn() const { return [](auto x) { return x > 0 ? x : 0; }; }
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

struct softmax
{
  std::string name() const { return "cpu::softmax"; }
  shape compute_shape(std::vector<shape> inputs) const { return inputs.front(); }
  argument compute(shape output_shape, std::vector<argument> args) const
  {
      argument result{output_shape};
      result.visit([&](auto output) {
          args[0].visit([&](auto input) {
              std::transform(input.begin(), input.end(), output.begin(), 
                  [](auto x) { return std::exp(x); });
              float t = std::accumulate(output.begin(), output.end(), zero(input.front()));
              std::transform(output.begin(), output.end(), output.begin(), 
                  [t](auto x) { return x/t; });
          });
      });
      return result;
  }
};

struct add_op
{
    std::string name() const { return "add"; }
    auto fcn() const { return [](auto x, auto y) {return x + y;};}
};

struct sub_op
{
    std::string name() const { return "sub"; }
    auto fcn() const { return [](auto x, auto y) {return x - y;};}
};

struct mul_op
{
    std::string name() const { return "mul"; }
    auto fcn() const { return [](auto x, auto y) {return x * y;};}
};

struct div_op
{
    std::string name() const { return "div"; }
    auto fcn() const { return [](auto x, auto y) {return x / y;};}
};

template <typename Op>
struct cpu_binary
{
  Op op;
  std::string name() const { op.name(); }
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
            else if(it->op.name() == "identity")
            {
                apply_identity(it);
            }
            else if(it->op.name() == "softmax")
            {
                apply_softmax(it);
            }
            else if(it->op.name() == "tanh")
            {
                apply_tanh(it);
            }
            else if(it->op.name() == "sigmoid")
            {
                apply_sigmoid(it);
            }
            else if(it->op.name() == "exp")
            {
                apply_exp(it);
            }
            else if(it->op.name() == "neg")
            {
                apply_neg(it);
            }
            else if(it->op.name() == "sin")
            {
                apply_sin(it);
            }
            else if(it->op.name() == "cos")
            {
                apply_cos(it);
            }
            else if(it->op.name() == "tan")
            {
                apply_tan(it);
            }
            else if(it->op.name() == "gemm")
            {
                apply_gemm(it);
            }
        }
    }

    void apply_convolution(instruction_ref ins)
    {
        auto&& op = any_cast<convolution>(ins->op);
        prog->replace_instruction(ins, cpu_convolution{op}, ins->arguments);
    }

    void apply_gemm(instruction_ref ins)
    {
        auto&& op = any_cast<gemm>(ins->op);
        prog->replace_instruction(ins, cpu_gemm{op}, ins->arguments);
    }

    void apply_activation(instruction_ref ins)
    {
        auto&& op = any_cast<activation>(ins->op);
        if(op.mode == "relu")
            prog->replace_instruction(ins, cpu_unary<relu_op>{}, ins->arguments);
    }

    void apply_identity(instruction_ref ins)
    {
        auto&& op = any_cast<identity>(ins->op);
        prog->replace_instruction(ins, cpu_unary<identity_op>{}, ins->arguments);
    }

    void apply_softmax(instruction_ref ins)
    {
        auto&& op = any_cast<softmax>(ins->op);
        prog->replace_instruction(ins, softmax{}, ins->arguments);
    }

    void apply_tanh(instruction_ref ins)
    {
        auto&& op = any_cast<tanh>(ins->op);
        prog->replace_instruction(ins, cpu_unary<tanh_op>{}, ins->arguments);
    }

    void apply_sigmoid(instruction_ref ins)
    {
        auto&& op = any_cast<sigmoid>(ins->op);
        prog->replace_instruction(ins, cpu_unary<sigmoid_op>{}, ins->arguments);
    }

    void apply_exp(instruction_ref ins)
    {
        auto&& op = any_cast<exp>(ins->op);
        prog->replace_instruction(ins, cpu_unary<exp_op>{}, ins->arguments);
    }

    void apply_neg(instruction_ref ins)
    {
        auto&& op = any_cast<neg>(ins->op);
        prog->replace_instruction(ins, cpu_unary<neg_op>{}, ins->arguments);
    }

    void apply_sin(instruction_ref ins)
    {
        auto&& op = any_cast<sin>(ins->op);
        prog->replace_instruction(ins, cpu_unary<sin_op>{}, ins->arguments);
    }

    void apply_cos(instruction_ref ins)
    {
        auto&& op = any_cast<cos>(ins->op);
        prog->replace_instruction(ins, cpu_unary<cos_op>{}, ins->arguments);
    }

    void apply_tan(instruction_ref ins)
    {
        auto&& op = any_cast<tan>(ins->op);
        prog->replace_instruction(ins, cpu_unary<tan_op>{}, ins->arguments);
    }
};

std::string cpu_target::name() const { return "cpu"; }

void cpu_target::apply(program& p) const { cpu_apply{&p}.apply(); }

} // namespace cpu

} // namespace rtg
