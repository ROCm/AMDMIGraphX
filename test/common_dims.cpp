#include <migraphx/common_dims.hpp>
#include <test.hpp>

TEST_CASE(common1)
{
    auto cd = migraphx::common_dims::compute({2, 32, 2560}, {2, 1280, 8, 8});
    EXPECT(cd.dims == std::vector<std::size_t>{2, 32, 40, 8, 8});
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
