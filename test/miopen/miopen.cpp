
#include <rtg/program.hpp>
#include <rtg/operators.hpp>
#include <rtg/generate.hpp>
#include <rtg/cpu/cpu_target.hpp>
#include <rtg/miopen/miopen_target.hpp>
#include <rtg/miopen/miopen.hpp>
#include <rtg/miopen/hip.hpp>
#include <rtg/manage_ptr.hpp>

#include <miopen/miopen.h>

#include "test.hpp"
#include "verify.hpp"

rtg::program create_program()
{
    rtg::program p;
    auto input   = p.add_parameter("x", rtg::shape{rtg::shape::float_type, {4, 3, 3, 3}});
    auto weights = p.add_parameter("w", rtg::shape{rtg::shape::float_type, {4, 3, 3, 3}});
    auto conv    = p.add_instruction(rtg::convolution{}, input, weights);
    p.add_instruction(rtg::activation{"relu"}, conv);
    return p;
}

// TODO: Move to header
rtg::argument get_tensor_argument_gpu(rtg::shape s)
{
    auto v = rtg::generate_tensor_data<float>(s);
    auto p = rtg::share(rtg::miopen::write_to_gpu(v));
    return {s, [p]() mutable { return reinterpret_cast<char*>(p.get()); }};
}

std::vector<float> cpu()
{
    std::vector<float> result;
    auto p = create_program();
    auto x = rtg::generate_argument({rtg::shape::float_type, {4, 3, 3, 3}});
    auto w = rtg::generate_argument({rtg::shape::float_type, {4, 3, 3, 3}});
    p.compile(rtg::cpu::cpu_target{});
    auto r      = p.eval({{"x", x}, {"w", w}});
    auto output = r.get<float>();
    result.assign(output.begin(), output.end());
    return result;
}

std::vector<float> gpu()
{
    std::vector<float> result;
    auto p = create_program();
    auto x = get_tensor_argument_gpu({rtg::shape::float_type, {4, 3, 3, 3}});
    auto w = get_tensor_argument_gpu({rtg::shape::float_type, {4, 3, 3, 3}});
    p.compile(rtg::miopen::miopen_target{});
    auto y      = get_tensor_argument_gpu(p.get_parameter_shape("output"));
    auto handle = rtg::miopen::make_obj<rtg::miopen::miopen_handle>(&miopenCreate);
    auto r      = p.eval(
        {{"x", x}, {"w", w}, {"output", y}, {"handle", {rtg::shape::any_type, handle.get()}}});
    result = rtg::miopen::read_from_gpu<float>(r.data(), r.get_shape().elements());
    return result;
}

void test1()
{
    auto x = cpu();
    auto y = gpu();
    EXPECT(test::verify_range(x, y));
}

int main() { test1(); }
