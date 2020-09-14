#include <migraphx/register_target.hpp>
#include <migraphx/cpu/target.hpp>
#include <migraphx/target.hpp>
#include "test.hpp"

TEST_CASE(make_target)
{
    for(const auto& name : migraphx::get_targets())
    {
        auto t = migraphx::make_target(name);
        CHECK(t.name() == name);
    }
}

TEST_CASE(targets)
{
    auto ts = migraphx::get_targets();
    EXPECT(ts.size() > 0);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
