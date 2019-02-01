#ifndef MIGRAPHX_GUARD_RTGLIB_VERIFY_ARGS_HPP
#define MIGRAPHX_GUARD_RTGLIB_VERIFY_ARGS_HPP

#include <migraphx/verify.hpp>
#include <migraphx/argument.hpp>
#include <migraphx/config.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

inline bool verify_args(const std::string& name,
                        const argument& cpu_arg,
                        const argument& gpu_arg,
                        double tolerance = 80)
{
    bool passed = true;
    visit_all(cpu_arg, gpu_arg)([&](auto cpu, auto gpu) {
        double error;
        passed = verify_range(cpu, gpu, tolerance, &error);
        if(not passed)
        {
            // TODO: Check for nans
            std::cout << "FAILED: " << name << std::endl;
            std::cout << "error: " << error << std::endl;
            if(cpu.size() < 32)
                std::cout << "cpu:" << cpu << std::endl;
            if(gpu.size() < 32)
                std::cout << "gpu:" << gpu << std::endl;
            if(range_zero(cpu))
                std::cout << "Cpu data is all zeros" << std::endl;
            if(range_zero(gpu))
                std::cout << "Gpu data is all zeros" << std::endl;

            auto mxdiff = max_diff(cpu, gpu);
            std::cout << "Max diff: " << mxdiff << std::endl;

            auto idx = mismatch_idx(cpu, gpu, float_equal);
            if(idx < range_distance(cpu))
            {
                std::cout << "Mismatch at " << idx << ": " << cpu[idx] << " != " << gpu[idx]
                          << std::endl;
            }

            auto cpu_nan_idx = find_idx(cpu, not_finite);
            if(cpu_nan_idx >= 0)
                std::cout << "Non finite number found in cpu at " << cpu_nan_idx << ": "
                          << cpu[cpu_nan_idx] << std::endl;

            auto gpu_nan_idx = find_idx(gpu, not_finite);
            if(gpu_nan_idx >= 0)
                std::cout << "Non finite number found in gpu at " << gpu_nan_idx << ": "
                          << gpu[gpu_nan_idx] << std::endl;
            std::cout << std::endl;
        }
        else
        {
            if(range_zero(cpu))
                std::cout << "Cpu data is all zeros" << std::endl;
            if(range_zero(gpu))
                std::cout << "Gpu data is all zeros" << std::endl;

            // auto mxdiff = max_diff(cpu, gpu);
            // std::cout << "Max diff: " << mxdiff << std::endl;

            // auto idx = mismatch_idx(cpu, gpu, float_equal);
            // if(idx < range_distance(cpu))
            // {
            //     std::cout << "Mismatch at " << idx << ": " << cpu[idx] << " != " << gpu[idx]
            //               << std::endl;
            // }

            auto cpu_nan_idx = find_idx(cpu, not_finite);
            if(cpu_nan_idx >= 0)
                std::cout << "Non finite number found in cpu at " << cpu_nan_idx << ": "
                          << cpu[cpu_nan_idx] << std::endl;

            auto gpu_nan_idx = find_idx(gpu, not_finite);
            if(gpu_nan_idx >= 0)
                std::cout << "Non finite number found in gpu at " << gpu_nan_idx << ": "
                          << gpu[gpu_nan_idx] << std::endl;
            // std::cout << std::endl;
        }
    });
    return passed;
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
