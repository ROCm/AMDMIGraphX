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
        p.debug_print(race.ins);
        std::cout << "Should happen after: ";
        p.debug_print(race.before);
    }
#else
    (void)p;
#endif
}

void validate_gpu(const migraphx::program& p, const migraphx::parameter_map& m)
{
    check_gpu_streams(p);
    // Program should have an output parameter
    EXPECT(std::any_of(
        m.begin(), m.end(), [](auto& x) { return migraphx::contains(x.first, "output"); }));

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
    rv.run(argc, argv);
}
