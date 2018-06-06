#include <cassert>
#include <iostream>
#include <vector>
#include <rtg/literal.hpp>
#include <rtg/operators.hpp>
#include <rtg/cpu/cpu_target.hpp>

using rtg::shape;
using rtg::argument;

void exp_test() {
    rtg::program p;
    rtg::shape s{rtg::shape::float_type, {3}};
    auto l = p.add_literal(rtg::literal{s, {-1,0,1}});
    p.add_instruction(rtg::exp{}, l);
    p.compile(rtg::cpu::cpu_target{});
    auto result = p.eval({});
    std::vector<float> results_vector(3);
    memcpy(results_vector.data(), result.data(), 3*sizeof(float));
    std::vector<float> gold = {0.36787944f,1.f,2.71828183f};
    float tol = 1e-8;
    for (int i = 0; i < results_vector.size(); i++) {
        assert(std::abs(results_vector[i]-gold[i]) < tol);
    }
}

void gemm_test() {
    rtg::program p;
    std::vector<float> A = {-0.00925222,  0.56250403,  0.70107397,  0.75402161, -0.505885  ,
                             1.33628943, -0.11413   , -0.31270559,  1.59336732, -0.19361027,
                            -0.91620867,  0.40108416, -0.06969921,  0.68483471, -0.39906632,
                            -1.66423624,  0.69040076, -1.31490171, -0.11282616, -0.79391814};
    std::vector<float> B = { 6.09568541e-01,  -6.10527007e-01,   3.66646462e-01,
                             1.18951101e-01,   5.58777432e-01,  -3.21296298e-01,
                            -5.95997198e-01,  -5.01425721e-01,  -2.84606807e-01,
                            -5.73673557e-01,  -8.99430260e-01,  -4.25103093e-01,
                             1.53027987e+00,  -3.81407415e-04,  -3.29650255e-01};
    std::vector<float> C = {-1.56327541e+00,  -7.09570140e-01,  -5.37424982e-01,
                            -2.22994831e-01,  -2.15586437e+00,   2.09177941e-03,
                            -1.47279677e+00,   2.02627040e-01,  -6.04527691e-01,
                            -1.29885596e+00,   2.16294914e+00,  -1.48101497e-01};
    rtg::shape a_shape{rtg::shape::float_type, {4,5}};
    auto a = p.add_literal(rtg::literal{a_shape, A});
    rtg::shape b_shape{rtg::shape::float_type, {5,3}};
    auto b = p.add_literal(rtg::literal{b_shape, B});
    p.add_instruction(rtg::gemm{}, a, b);
    p.compile(rtg::cpu::cpu_target{});
    auto result = p.eval({});
    std::vector<float> results_vector(12);
    memcpy(results_vector.data(), result.data(), 12*sizeof(float));
    float tol = 1e-6;
    for (int i = 0; i < results_vector.size(); i++) {
        assert(std::abs(results_vector[i]-C[i]) < tol);
    }
}

int main()
{
    exp_test();
    gemm_test();
}
