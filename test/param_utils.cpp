#include <migraphx/param_utils.hpp>
#include <migraphx/ranges.hpp>
#include <random>
#include <test.hpp>

TEST_CASE(test_param_name)
{
    CHECK(migraphx::param_name(0) == "x0");
    CHECK(migraphx::param_name(1) == "x1");
    CHECK(migraphx::param_name(10) == "x:00010");
    CHECK(migraphx::param_name(11) == "x:00011");
    CHECK(migraphx::param_name(100) == "x:00100");
    CHECK(migraphx::param_name(101) == "x:00101");
    CHECK(migraphx::param_name(10011) == "x:10011");
    CHECK(migraphx::param_name(99999) == "x:99999");
    CHECK(test::throws([] { migraphx::param_name(100000); }));
    CHECK(test::throws([] { migraphx::param_name(100001); }));
}

TEST_CASE(test_param_name_sorted)
{
    auto pname = [](std::size_t i) { return migraphx::param_name(i); };
    std::vector<std::string> names;
    migraphx::transform(migraphx::range(8, 25), std::back_inserter(names), pname);
    migraphx::transform(migraphx::range(90, 130), std::back_inserter(names), pname);
    migraphx::transform(migraphx::range(990, 1030), std::back_inserter(names), pname);
    migraphx::transform(migraphx::range(9990, 10030), std::back_inserter(names), pname);
    migraphx::transform(migraphx::range(99990, 100000), std::back_inserter(names), pname);
    CHECK(std::is_sorted(names.begin(), names.end()));

    auto xnames = names;
    // Shuffled
    std::shuffle(xnames.begin(), xnames.end(), std::minstd_rand{});
    std::sort(xnames.begin(), xnames.end());
    EXPECT(xnames == names);
    // Reversed
    std::reverse(xnames.begin(), xnames.end());
    std::sort(xnames.begin(), xnames.end());
    EXPECT(xnames == names);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
