#include "run_verify.hpp"
#include <migraphx/ranges.hpp>
#include <test.hpp>

#ifdef HAVE_GPU
#include <migraphx/gpu/analyze_streams.hpp>
#include <migraphx/gpu/target.hpp>
#endif
#ifdef HAVE_CPU
#include <migraphx/cpu/target.hpp>
#endif

inline void check_gpu_streams(const migraphx::program& p)
{
#ifdef HAVE_GPU
    const auto* mm = p.get_main_module();
    auto races     = migraphx::gpu::analyze_streams(*mm);
    for(auto&& race : races)
    {
        std::cout << "FAILED: " << std::endl;
        std::cout << "Race condition detected for: ";
        mm->debug_print(race.ins);
        std::cout << "Should happen after: ";
        mm->debug_print(race.before);
    }
#else
    (void)p;
#endif
}

void validate_gpu(const migraphx::program& p, const migraphx::parameter_map& m)
{
    check_gpu_streams(p);

    // Ensure the program doesn't modify the context in a dry run
    auto ctx = p.get_context();
    assert(&ctx != &p.get_context());
    EXPECT(is_shared(ctx, p.get_context()));
    p.dry_run(m);
    EXPECT(is_shared(ctx, p.get_context()));
}

int main(int argc, const char* argv[])
{
    run_verify rv;
    rv.add_validation_for("gpu", &validate_gpu);
    rv.disable_test_for("cpu", {"test_if_lp", "test_if_param", "test_if_literal"});
    rv.disable_test_for("gpu",
                        {"batch_quant_dot_2",
                         "batch_quant_dot_3",
                         "batch_quant_dot_5",
                         "quant_dot_3args_1",
                         "quant_dot_3args_2",
                         "quant_dot_3args_3",
                         "quant_dot_3args_4",
                         "quant_dot_3args_5"});
    rv.run(argc, argv);
}
