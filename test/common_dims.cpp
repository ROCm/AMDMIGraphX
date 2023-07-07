#include <migraphx/common_dims.hpp>
#include <test.hpp>

using axes_map = std::vector<std::vector<std::size_t>>;

TEST_CASE(common_d1_less)
{
    auto cd = migraphx::common_dims::compute({2, 32, 40, 8}, {2, 1280, 8});
    EXPECT(cd.dims == std::vector<std::size_t>{2, 32, 40, 8});
    EXPECT(cd.axes_map1 == axes_map{{0}, {1}, {2}, {3}});
    EXPECT(cd.axes_map2 == axes_map{{0}, {1, 2}, {3}});
}

TEST_CASE(common1)
{
    auto cd = migraphx::common_dims::compute({2, 32, 2560}, {2, 1280, 8, 8});
    EXPECT(cd.dims == std::vector<std::size_t>{2, 32, 40, 8, 8});
    EXPECT(cd.axes_map1 == axes_map{{0}, {1}, {2, 3, 4}});
    EXPECT(cd.axes_map2 == axes_map{{0}, {1, 2}, {3}, {4}});
}

TEST_CASE(common2)
{
    auto cd = migraphx::common_dims::compute({2, 1280, 8, 8}, {2, 32, 2560});
    EXPECT(cd.dims == std::vector<std::size_t>{2, 32, 40, 8, 8});
    EXPECT(cd.axes_map1 == axes_map{{0}, {1, 2}, {3}, {4}});
    EXPECT(cd.axes_map2 == axes_map{{0}, {1}, {2, 3, 4}});
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
