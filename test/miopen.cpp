
#include <rtg/program.hpp>
#include <rtg/operators.hpp>
#include <rtg/cpu/cpu_target.hpp>
#include <rtg/miopen/miopen_target.hpp>
#include <rtg/manage_ptr.hpp>

#include <miopen/miopen.h>

#include <random>

#include "test.hpp"

using hip_ptr       = RTG_MANAGE_PTR(void, hipFree);
using miopen_handle = RTG_MANAGE_PTR(miopenHandle_t, miopenDestroy);

template <class Result, class F, class... Ts>
Result make_obj(F f, Ts... xs)
{
    typename Result::pointer x = nullptr;
    auto status                = f(&x, xs...);
    Result r{x};
    if(status != miopenStatusSuccess)
        RTG_THROW("MIOpen call failed");
    return r;
}

hip_ptr hip_allocate(std::size_t sz)
{
    void* result;
    // TODO: Check status
    hipMalloc(&result, sz);
    return hip_ptr{result};
}

template <class T>
hip_ptr write(const T& x)
{
    using type  = typename T::value_type;
    auto size   = x.size() * sizeof(type);
    auto result = hip_allocate(size);
    // TODO: Check status
    hipMemcpy(result.get(), x.data(), size, hipMemcpyHostToDevice);
    return result;
}

template <class T>
std::vector<T> read(const hip_ptr& x, std::size_t sz)
{
    std::vector<T> result(sz);
    // TODO: Check status
    hipMemcpy(result.data(), x.get(), sz * sizeof(T), hipMemcpyDeviceToHost);
    return result;
}

rtg::program create_program()
{
    rtg::program p;
    auto input   = p.add_parameter("x", rtg::shape{rtg::shape::float_type, {4, 3, 3, 3}});
    auto weights = p.add_parameter("w", rtg::shape{rtg::shape::float_type, {4, 3, 3, 3}});
    auto conv    = p.add_instruction(rtg::convolution{}, input, weights);
    p.add_instruction(rtg::activation{"relu"}, conv);
    return p;
}

std::vector<float> get_tensor_data(rtg::shape s)
{
    std::vector<float> result(s.elements());
    std::mt19937 engine{0};
    std::uniform_real_distribution<> dist;
    std::generate(result.begin(), result.end(), [&] { return dist(engine); });
    return result;
}

rtg::argument get_tensor_argument_cpu(rtg::shape s)
{
    auto v = get_tensor_data(s);
    return {s, [v]() mutable { return reinterpret_cast<char*>(v.data()); }};
}

rtg::argument get_tensor_argument_gpu(rtg::shape s)
{
    auto v = get_tensor_data(s);
    auto p = rtg::share(write(v));
    return {s, [p]() mutable { return reinterpret_cast<char*>(p.get()); }};
}

std::vector<float> cpu()
{
    std::vector<float> result;
    auto p = create_program();
    auto x = get_tensor_argument_cpu({rtg::shape::float_type, {4, 3, 3, 3}});
    auto w = get_tensor_argument_cpu({rtg::shape::float_type, {4, 3, 3, 3}});
    p.compile(rtg::cpu::cpu_target{});
    auto r = p.eval({{"x", x}, {"w", w}});
    r.visit([&](auto output) { result.assign(output.begin(), output.end()); });
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
    auto handle = make_obj<miopen_handle>(&miopenCreate);
    auto r      = p.eval(
        {{"x", x}, {"w", w}, {"output", y}, {"handle", {rtg::shape::any_type, handle.get()}}});
    r.visit([&](auto output) { result.assign(output.begin(), output.end()); });
    return result;
}

void test1()
{
    auto x = cpu();
    auto y = gpu();
    // TODO: Use expect
    if(x == y)
        std::cout << "FAILED" << std::endl;
}

int main() { test1(); }
