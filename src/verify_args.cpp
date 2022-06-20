
#include <migraphx/verify_args.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

bool verify_args(const std::string& name,
                 const argument& ref_arg,
                 const argument& target_arg,
                 double tolerance)
{
    bool passed = true;
    visit_all(ref_arg, target_arg)([&](auto ref, auto target) {
        double error;
        passed = verify_range(ref, target, tolerance, &error);
        if(not passed)
        {
            // TODO: Check for nans
            std::cout << "FAILED: " << name << std::endl;
            std::cout << "error: " << error << std::endl;
            if(ref.size() < 32)
                std::cout << "ref:" << ref << std::endl;
            if(target.size() < 32)
                std::cout << "target:" << target << std::endl;
            if(range_zero(ref))
                std::cout << "Ref data is all zeros" << std::endl;
            if(range_zero(target))
                std::cout << "Target data is all zeros" << std::endl;

            auto mxdiff = max_diff(ref, target);
            std::cout << "Max diff: " << mxdiff << std::endl;

            auto idx = mismatch_idx(ref, target, float_equal);
            if(idx < range_distance(ref))
            {
                std::cout << "Mismatch at " << idx << ": " << ref[idx] << " != " << target[idx]
                          << std::endl;
            }

            auto ref_nan_idx = find_idx(ref, not_finite);
            if(ref_nan_idx >= 0)
                std::cout << "Non finite number found in ref at " << ref_nan_idx << ": "
                          << ref[ref_nan_idx] << std::endl;

            auto target_nan_idx = find_idx(target, not_finite);
            if(target_nan_idx >= 0)
                std::cout << "Non finite number found in target at " << target_nan_idx << ": "
                          << target[target_nan_idx] << std::endl;
            std::cout << std::endl;
        }
        else
        {
            if(range_zero(ref))
                std::cout << "Ref data is all zeros" << std::endl;
            if(range_zero(target))
                std::cout << "Target data is all zeros" << std::endl;

            // auto mxdiff = max_diff(ref, target);
            // std::cout << "Max diff: " << mxdiff << std::endl;

            // auto idx = mismatch_idx(ref, target, float_equal);
            // if(idx < range_distance(ref))
            // {
            //     std::cout << "Mismatch at " << idx << ": " << ref[idx] << " != " << target[idx]
            //               << std::endl;
            // }

            auto ref_nan_idx = find_idx(ref, not_finite);
            if(ref_nan_idx >= 0)
                std::cout << "Non finite number found in ref at " << ref_nan_idx << ": "
                          << ref[ref_nan_idx] << std::endl;

            auto target_nan_idx = find_idx(target, not_finite);
            if(target_nan_idx >= 0)
                std::cout << "Non finite number found in target at " << target_nan_idx << ": "
                          << target[target_nan_idx] << std::endl;
            // std::cout << std::endl;
        }
    });
    return passed;
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
