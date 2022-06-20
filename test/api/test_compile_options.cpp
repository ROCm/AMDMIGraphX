#include <migraphx/migraphx.h>
#include <migraphx/migraphx.hpp>
#include <migraphx/compile_options.hpp>
#include "test.hpp"

TEST_CASE(compile_options_api_test)
{
    migraphx::api::compile_options options;
    options.set_offload_copy(false);
    options.set_fast_math(false);
    const auto* s_options = reinterpret_cast<const migraphx::MIGRAPHX_INLINE_NS::compile_options*>(
        options.get_handle_ptr());
    CHECK(s_options->fast_math == false);
    CHECK(s_options->offload_copy == false);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
