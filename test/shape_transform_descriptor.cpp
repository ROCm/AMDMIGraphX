#include <migraphx/shape_transform_descriptor.hpp>
#include <migraphx/make_op.hpp>
#include <test.hpp>

using migraphx::make_op;
using migraphx::shape_transform_descriptor;
using all_lens   = std::vector<std::vector<std::size_t>>;
using final_lens = std::vector<std::size_t>;
using all_axes   = std::vector<std::vector<std::vector<std::size_t>>>;
using d_axes     = std::vector<std::vector<std::size_t>>;
using ops        = std::vector<migraphx::operation>;

all_lens get_all_lens(const shape_transform_descriptor& d)
{
    all_lens result;
    std::transform(d.dimensions.begin(),
                   d.dimensions.end(),
                   std::back_inserter(result),
                   [](const auto& dimension) {
                       std::vector<std::size_t> sub_lens;
                       std::transform(dimension.subdimensions.begin(),
                                      dimension.subdimensions.end(),
                                      std::back_inserter(sub_lens),
                                      [](const auto& x) { return x.len; });
                       return sub_lens;
                   });
    return result;
}

final_lens get_final_lens(const shape_transform_descriptor& d)
{
    final_lens result;
    std::transform(d.dimensions.begin(),
                   d.dimensions.end(),
                   std::back_inserter(result),
                   [](const auto& x) { return x.len(); });
    return result;
}

all_axes get_all_axes(const shape_transform_descriptor& d)
{
    all_axes result;
    std::transform(d.dimensions.begin(),
                   d.dimensions.end(),
                   std::back_inserter(result),
                   [](const auto& dimension) {
                       std::vector<std::vector<std::size_t>> sub_axis;
                       std::transform(dimension.subdimensions.begin(),
                                      dimension.subdimensions.end(),
                                      std::back_inserter(sub_axis),
                                      [](const auto& x) { return x.axis; });
                       return sub_axis;
                   });
    return result;
}

template <class... Ts>
shape_transform_descriptor make_descriptor(const std::vector<std::size_t>& dims, const Ts&... xs)
{
    auto desc = shape_transform_descriptor{dims};
    CHECK(desc.apply({xs...}));
    return desc;
}

TEST_CASE(record_reshape)
{
    auto desc = make_descriptor({256, 3, 16, 16}, make_op("reshape", {{"dims", {16, 16, 48, 16}}}));
    EXPECT(get_final_lens(desc) == final_lens{16, 16, 48, 16});
    EXPECT(get_all_lens(desc) == all_lens{{16}, {16}, {3, 16}, {16}});
    EXPECT(get_all_axes(desc) ==
           all_axes{d_axes{{0, 0}}, d_axes{{0, 1}}, d_axes{{1}, {2}}, d_axes{{3}}});
}

TEST_CASE(record_transpose)
{
    auto desc =
        make_descriptor({256, 3, 16, 16}, make_op("transpose", {{"permutation", {0, 2, 3, 1}}}));
    EXPECT(get_final_lens(desc) == final_lens{256, 16, 16, 3});
    EXPECT(get_all_lens(desc) == all_lens{{256}, {16}, {16}, {3}});
    EXPECT(get_all_axes(desc) == all_axes{d_axes{{0}}, d_axes{{2}}, d_axes{{3}}, d_axes{{1}}});
}

TEST_CASE(record_multibroadcast)
{
    auto desc =
        make_descriptor({1, 3, 1, 1}, make_op("multibroadcast", {{"out_lens", {256, 3, 16, 16}}}));
    EXPECT(get_final_lens(desc) == final_lens{256, 3, 16, 16});
    EXPECT(get_all_lens(desc) == all_lens{{256}, {3}, {16}, {16}});
    EXPECT(get_all_axes(desc) == all_axes{d_axes{{}}, d_axes{{1}}, d_axes{{}}, d_axes{{}}});
}

TEST_CASE(optimize_multibroadcast_transpose_reshape)
{
    EXPECT(
        migraphx::optimize_shape_transforms({1, 5, 2},
                                            {make_op("multibroadcast", {{"out_lens", {20, 5, 2}}}),
                                             make_op("transpose", {{"permutation", {0, 2, 1}}}),
                                             make_op("reshape", {{"dims", {20, 10}}}),}) ==
        ops{make_op("transpose", {{"permutation", {0, 2, 1}}}),
            make_op("reshape", {{"dims", {1, 10}}}),
            make_op("multibroadcast", {{"out_lens", {20, 10}}}),});
}

TEST_CASE(optimize_resize)
{
    EXPECT(
        migraphx::optimize_shape_transforms({3, 4, 4},
                                            {make_op("reshape", {{"dims", {3, 1, 4, 1, 4}}}),
                                            make_op("multibroadcast", {{"out_lens", {3, 2, 4, 2, 4}}}),
                                             make_op("reshape", {{"dims", {3, 8, 8}}}),}) ==
        ops{make_op("reshape", {{"dims", {3, 1, 4, 1, 4}}}),
                                            make_op("multibroadcast", {{"out_lens", {3, 2, 4, 2, 4}}}),
                                             make_op("reshape", {{"dims", {3, 8, 8}}}),});
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
