#include <migraphx/reduce_dims.hpp>
#include <migraphx/permutation.hpp>
#include "test.hpp"

migraphx::shape make_shape(std::vector<std::size_t> lens)
{
    return {migraphx::shape::float_type, lens};
}

migraphx::shape make_shape(std::vector<std::size_t> lens, std::vector<std::size_t> strides)
{
    return {migraphx::shape::float_type, lens, strides};
}

TEST_CASE(same_standard)
{
    auto is                              = make_shape({64, 3, 7, 7});
    auto os                              = make_shape({64 * 3 * 7 * 7});
    std::vector<migraphx::shape> ishapes = {is, is, is};
    std::vector<migraphx::shape> eshapes = {os, os, os};
    auto rshapes                         = migraphx::reduce_dims(ishapes);

    EXPECT(eshapes == rshapes);
}

TEST_CASE(same_broadcast1)
{
    auto is                              = make_shape({64, 3, 7, 7});
    auto os                              = make_shape({64, 3, 7 * 7});
    std::vector<migraphx::shape> ishapes = {is, make_shape({64, 3, 7, 7}, {0, 1, 0, 0}), is};
    std::vector<migraphx::shape> eshapes = {os, make_shape({64, 3, 7 * 7}, {0, 1, 0}), os};
    auto rshapes                         = migraphx::reduce_dims(ishapes);

    EXPECT(eshapes == rshapes);
}

TEST_CASE(same_broadcast2)
{
    auto is                              = make_shape({64, 3, 8, 7, 7});
    auto os                              = make_shape({64, 8 * 3, 7 * 7});
    std::vector<migraphx::shape> ishapes = {is, make_shape({64, 3, 8, 7, 7}, {0, 8, 1, 0, 0}), is};
    std::vector<migraphx::shape> eshapes = {os, make_shape({64, 8 * 3, 7 * 7}, {0, 1, 0}), os};
    auto rshapes                         = migraphx::reduce_dims(ishapes);

    EXPECT(eshapes == rshapes);
}

TEST_CASE(same_transposed)
{
    auto is                              = make_shape({64, 3, 7, 7});
    auto os                              = make_shape({64 * 3, 7, 7});
    std::vector<migraphx::shape> ishapes = {is, migraphx::reorder_shape(is, {0, 1, 3, 2}), is};
    std::vector<migraphx::shape> eshapes = {os, migraphx::reorder_shape(os, {0, 2, 1}), os};
    auto rshapes                         = migraphx::reduce_dims(ishapes);

    EXPECT(eshapes == rshapes);
}

TEST_CASE(different_masked1)
{
    auto is                              = make_shape({64, 3, 7, 7});
    auto os                              = make_shape({64, 3, 7 * 7});
    std::vector<migraphx::shape> ishapes = {is, make_shape({1, 3, 1, 1}), is};
    std::vector<migraphx::shape> eshapes = {os, make_shape({1, 3, 1}), os};
    auto rshapes                         = migraphx::reduce_dims(ishapes);

    EXPECT(eshapes == rshapes);
}

TEST_CASE(different_masked2)
{
    auto is                              = make_shape({64, 3, 7, 7});
    auto os                              = make_shape({64, 3, 7 * 7});
    std::vector<migraphx::shape> ishapes = {
        is, make_shape({1, 3, 1, 1}), make_shape({64, 1, 7, 7})};
    std::vector<migraphx::shape> eshapes = {os, make_shape({1, 3, 1}), make_shape({64, 1, 7 * 7})};
    auto rshapes                         = migraphx::reduce_dims(ishapes);

    EXPECT(eshapes == rshapes);
}

TEST_CASE(different_incompatible)
{
    auto is                              = make_shape({64, 3, 7, 7});
    std::vector<migraphx::shape> ishapes = {is, make_shape({1, 3, 2, 1}), is};
    auto rshapes                         = migraphx::reduce_dims(ishapes);

    EXPECT(ishapes == rshapes);
}

TEST_CASE(different_ranks)
{
    auto is                              = make_shape({64, 3, 7, 7});
    std::vector<migraphx::shape> ishapes = {is, make_shape({1, 3}), is};
    auto rshapes                         = migraphx::reduce_dims(ishapes);

    EXPECT(ishapes == rshapes);
}

TEST_CASE(empty)
{
    auto rshapes = migraphx::reduce_dims({});
    EXPECT(rshapes.empty());
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
