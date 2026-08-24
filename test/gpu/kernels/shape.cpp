#include <migraphx/kernels/shape.hpp>
#include <migraphx/kernels/test.hpp>


TEST_CASE(test_shape_assign)
{
    auto s1 = migraphx::make_shape(migraphx::index_ints<100, 32, 8, 8>{});
    auto s2 = s1; // NOLINT
    // cppcheck-suppress knownConditionTrueFalse
    EXPECT(s1 == s2);
    // cppcheck-suppress knownConditionTrueFalse
    EXPECT(not(s1 != s2));
}

TEST_CASE(test_shape_packed_default)
{
    auto s = migraphx::make_shape(migraphx::index_ints<2, 2>{});
    EXPECT(s.standard());
    EXPECT(s.packed());
    EXPECT(not s.transposed());
    EXPECT(not s.broadcasted());
}

TEST_CASE(test_shape_standard)
{
    auto s = migraphx::make_shape(migraphx::index_ints<2, 2, 3>{}, migraphx::index_ints<6, 3, 1>{});
    EXPECT(s.standard());
    EXPECT(s.packed());
    EXPECT(not s.transposed());
    EXPECT(not s.broadcasted());
}

TEST_CASE(test_shape_standard_singleton_dim)
{
    auto s = migraphx::make_shape(migraphx::index_ints<5, 1, 8>{}, migraphx::index_ints<8, 4, 1>{});
    EXPECT(s.standard());
    EXPECT(s.packed());
    EXPECT(not s.transposed());
    EXPECT(not s.broadcasted());
}

TEST_CASE(test_shape_standard_stray_singleton_dim)
{
    // A shape can be transposed (nonzero strides out of order) but still be considered
    // standard if the only out-of-order strides are on axes with a length of 1.
    auto s = migraphx::make_shape(migraphx::index_ints<5, 1, 1, 8>{}, migraphx::index_ints<8, 3, 4, 1>{});
    EXPECT(s.standard());
    EXPECT(s.packed());
    EXPECT(s.transposed());
    EXPECT(not s.broadcasted());
}

TEST_CASE(test_shape_packed)
{
    auto s = migraphx::make_shape(migraphx::index_ints<2, 2>{}, migraphx::index_ints<2, 1>{});
    EXPECT(s.standard());
    EXPECT(s.packed());
    EXPECT(not s.transposed());
    EXPECT(not s.broadcasted());
}

TEST_CASE(test_shape_non_packed_single_dim)
{
    auto s = migraphx::make_shape(migraphx::index_ints<1, 64, 35, 35>{}, migraphx::index_ints<156800, 1225, 35, 1>{});
    EXPECT(s.standard());
    EXPECT(s.packed());
    EXPECT(not s.transposed());
    EXPECT(not s.broadcasted());
}

TEST_CASE(test_shape_transposed1)
{
    auto s = migraphx::make_shape(migraphx::index_ints<2, 2>{}, migraphx::index_ints<1, 2>{});
    EXPECT(not s.standard());
    EXPECT(s.packed());
    EXPECT(s.transposed());
    EXPECT(not s.broadcasted());
}

TEST_CASE(test_shape_transposed2)
{
    auto s = migraphx::make_shape(migraphx::index_ints<1, 1, 1, 1, 2>{}, migraphx::index_ints<2, 2, 2, 2, 1>{});
    EXPECT(s.standard());
    EXPECT(s.packed());
    EXPECT(not s.transposed());
    EXPECT(not s.broadcasted());
}

TEST_CASE(test_shape_overlap)
{
    auto s = migraphx::make_shape(migraphx::index_ints<2, 2, 3>{}, migraphx::index_ints<6, 3, 2>{});
    EXPECT(not s.standard());
    EXPECT(not s.packed());
    EXPECT(not s.transposed());
    EXPECT(not s.broadcasted());
}

TEST_CASE(test_shape_overlap2)
{
    auto s = migraphx::make_shape(migraphx::index_ints<2, 2, 3>{}, migraphx::index_ints<6, 2, 1>{});
    EXPECT(not s.standard());
    EXPECT(not s.packed());
    EXPECT(not s.transposed());
    EXPECT(not s.broadcasted());
}

TEST_CASE(test_shape_overlap3)
{
    auto s = migraphx::make_shape(migraphx::index_ints<2, 2, 3>{}, migraphx::index_ints<4, 2, 1>{});
    EXPECT(not s.standard());
    EXPECT(not s.packed());
    EXPECT(not s.transposed());
    EXPECT(not s.broadcasted());
}

TEST_CASE(test_shape_scalar2)
{
    auto s = migraphx::make_shape(migraphx::index_ints<1>{}, migraphx::index_ints<0>{});
    EXPECT(s.standard());
    EXPECT(s.packed());
    EXPECT(not s.transposed());
    EXPECT(s.broadcasted());
}

TEST_CASE(test_shape_scalar_broadcast)
{
    auto s = migraphx::make_shape(migraphx::index_ints<1, 2, 3, 3>{}, migraphx::index_ints<0, 0, 0, 0>{});
    EXPECT(not s.standard());
    EXPECT(not s.packed());
    EXPECT(not s.transposed());
    EXPECT(s.broadcasted());
}

TEST_CASE(test_shape_broadcasted)
{
    auto s = migraphx::make_shape(migraphx::index_ints<2, 2>{}, migraphx::index_ints<1, 0>{});
    EXPECT(not s.standard());
    EXPECT(not s.packed());
    EXPECT(not s.transposed());
    EXPECT(s.broadcasted());
}

TEST_CASE(test_shape_broadcasted2)
{
    auto s = migraphx::make_shape(migraphx::index_ints<1, 2>{}, migraphx::index_ints<0, 1>{});
    EXPECT(s.standard());
    EXPECT(s.packed());
    EXPECT(not s.transposed());
    EXPECT(s.broadcasted());
}

TEST_CASE(test_shape_broadcasted3)
{
    auto s = migraphx::make_shape(migraphx::index_ints<3, 2>{}, migraphx::index_ints<0, 1>{});
    EXPECT(not s.standard());
    EXPECT(not s.packed());
    EXPECT(not s.transposed());
    EXPECT(s.broadcasted());
}

TEST_CASE(test_shape_broadcasted4)
{
    auto s = migraphx::make_shape(migraphx::index_ints<2, 2, 3>{}, migraphx::index_ints<6, 0, 1>{});
    EXPECT(not s.standard());
    EXPECT(not s.packed());
    EXPECT(not s.transposed());
    EXPECT(s.broadcasted());
}

TEST_CASE(test_shape_broadcasted5)
{
    auto s = migraphx::make_shape(migraphx::index_ints<2, 2, 3>{}, migraphx::index_ints<1, 0, 6>{});
    EXPECT(not s.standard());
    EXPECT(not s.packed());
    EXPECT(s.transposed());
    EXPECT(s.broadcasted());
}

TEST_CASE(test_shape_step_broadcasted)
{
    auto s = migraphx::make_shape(migraphx::index_ints<2, 2>{}, migraphx::index_ints<0, 3>{});
    EXPECT(not s.standard());
    EXPECT(not s.packed());
    EXPECT(not s.transposed());
    EXPECT(s.broadcasted());
}

TEST_CASE(test_shape_default_copy)
{
    // auto s1{};
    // auto s2{};
    // EXPECT(s1 == s2);
    // EXPECT(not(s1 != s2));
}

TEST_CASE(test_shape4)
{
    auto s = migraphx::make_shape(migraphx::index_ints<100, 32, 8, 8>{});
    EXPECT(s.standard());
    EXPECT(s.packed());
    EXPECT(not s.transposed());
    EXPECT(not s.broadcasted());
    EXPECT(s.lens[0] == 100);
    EXPECT(s.lens[1] == 32);
    EXPECT(s.lens[2] == 8);
    EXPECT(s.lens[3] == 8);
    EXPECT(s.strides[0] == s.lens[1] * s.strides[1]);
    EXPECT(s.strides[1] == s.lens[2] * s.strides[2]);
    EXPECT(s.strides[2] == s.lens[3] * s.strides[3]);
    EXPECT(s.strides[3] == 1);
    EXPECT(s.elements() == 100 * 32 * 8 * 8);
    EXPECT(s.index({0, 0, 0, 0}) == 0);
    EXPECT(s.index({0, 0, 0, 1}) == 1);
    EXPECT(s.index({0, 0, 0, 0}) == s.index(0));
    EXPECT(s.index({0, 0, 0, 1}) == s.index(1));
    EXPECT(s.index({0, 0, 1, 0}) == s.index(8));
    EXPECT(s.index({0, 1, 0, 0}) == s.index(8 * 8));
    EXPECT(s.index({1, 0, 0, 0}) == s.index(8 * 8 * 32));
    EXPECT(s.index(0) == 0);
    EXPECT(s.index(1) == 1);
    EXPECT(s.index(8) == 8);
    EXPECT(s.index(8 * 8) == 8 * 8);
    EXPECT(s.index(8 * 8 * 32) == 8 * 8 * 32);
    EXPECT(s.index(s.elements() - 1) == s.elements() - 1);
}

TEST_CASE(test_shape42)
{
    auto s = migraphx::make_shape(migraphx::index_ints<100, 32, 8, 8>{}, migraphx::index_ints<2048, 64, 8, 1>{});
    EXPECT(s.standard());
    EXPECT(s.packed());
    EXPECT(not s.transposed());
    EXPECT(not s.broadcasted());
    EXPECT(s.lens[0] == 100);
    EXPECT(s.lens[1] == 32);
    EXPECT(s.lens[2] == 8);
    EXPECT(s.lens[3] == 8);
    EXPECT(s.strides[0] == s.lens[1] * s.strides[1]);
    EXPECT(s.strides[1] == s.lens[2] * s.strides[2]);
    EXPECT(s.strides[2] == s.lens[3] * s.strides[3]);
    EXPECT(s.strides[3] == 1);
    EXPECT(s.elements() == 100 * 32 * 8 * 8);
    EXPECT(s.index({0, 0, 0, 0}) == 0);
    EXPECT(s.index({0, 0, 0, 1}) == 1);
    EXPECT(s.index({0, 0, 0, 0}) == s.index(0));
    EXPECT(s.index({0, 0, 0, 1}) == s.index(1));
    EXPECT(s.index({0, 0, 1, 0}) == s.index(8));
    EXPECT(s.index({0, 1, 0, 0}) == s.index(8 * 8));
    EXPECT(s.index({1, 0, 0, 0}) == s.index(8 * 8 * 32));
    EXPECT(s.index(0) == 0);
    EXPECT(s.index(1) == 1);
    EXPECT(s.index(8) == 8);
    EXPECT(s.index(8 * 8) == 8 * 8);
    EXPECT(s.index(8 * 8 * 32) == 8 * 8 * 32);
    EXPECT(s.index(s.elements() - 1) == s.elements() - 1);
}

TEST_CASE(test_shape4_transposed)
{
    auto s = migraphx::make_shape(migraphx::index_ints<32, 100, 8, 8>{}, migraphx::index_ints<64, 2048, 8, 1>{});
    EXPECT(s.transposed());
    EXPECT(s.packed());
    EXPECT(not s.standard());
    EXPECT(not s.broadcasted());
    EXPECT(s.lens[0] == 32);
    EXPECT(s.lens[1] == 100);
    EXPECT(s.lens[2] == 8);
    EXPECT(s.lens[3] == 8);
    EXPECT(s.strides[0] == 64);
    EXPECT(s.strides[1] == 2048);
    EXPECT(s.strides[2] == 8);
    EXPECT(s.strides[3] == 1);
    EXPECT(s.elements() == 100 * 32 * 8 * 8);
    EXPECT(s.index({0, 0, 0, 0}) == 0);
    EXPECT(s.index({0, 0, 0, 1}) == 1);
    EXPECT(s.index({0, 0, 0, 0}) == s.index(0));
    EXPECT(s.index({0, 0, 0, 1}) == s.index(1));
    EXPECT(s.index({0, 0, 1, 0}) == s.index(8));
    EXPECT(s.index({0, 1, 0, 0}) == s.index(8 * 8));
    EXPECT(s.index({1, 0, 0, 0}) == s.index(8 * 8 * 100));
    EXPECT(s.index(0) == 0);
    EXPECT(s.index(1) == 1);
    EXPECT(s.index(8) == 8);
    EXPECT(s.index(8 * 8) == 2048);
    EXPECT(s.index(8 * 8 * 100) == 64);
    EXPECT(s.index(s.elements() - 1) == s.elements() - 1);
}

TEST_CASE(test_shape4_nonpacked)
{
    auto lens = migraphx::index_ints<100, 32, 8, 8>{};
    auto offsets = migraphx::index_ints<5, 10, 0, 6>{};
    auto adj_lens = migraphx::transform(lens, offsets, [](auto l, auto o) { return l + o; });
    auto strides = migraphx::make_shape(adj_lens).strides;

    auto s = migraphx::make_shape(lens, strides);
    EXPECT(not s.standard());
    EXPECT(not s.packed());
    EXPECT(not s.transposed());
    EXPECT(not s.broadcasted());
    EXPECT(s.lens[0] == 100);
    EXPECT(s.lens[1] == 32);
    EXPECT(s.lens[2] == 8);
    EXPECT(s.lens[3] == 8);
    EXPECT(s.strides[0] == 4704);
    EXPECT(s.strides[1] == 112);
    EXPECT(s.strides[2] == 14);
    EXPECT(s.strides[3] == 1);
    EXPECT(s.elements() == 100 * 32 * 8 * 8);

    EXPECT(s.index(0) == 0);
    EXPECT(s.index(1) == 1);
    EXPECT(s.index({0, 0, 0, 0}) == 0);
    EXPECT(s.index({0, 0, 0, 1}) == s.index(1));
    EXPECT(s.index({0, 0, 1, 0}) == s.index(8));
    EXPECT(s.index({0, 1, 0, 0}) == s.index(8 * 8));
    EXPECT(s.index({1, 0, 0, 0}) == s.index(8 * 8 * 32));
    EXPECT(s.index(s.elements() - 1) == 469273);
}

template<class Shape, class NewLens>
static constexpr auto with_lens(Shape s, NewLens newlens)
{
    auto perm = migraphx::find_permutation(s);
    return migraphx::make_shape_from_permutation(newlens, perm);
}

TEST_CASE(test_with_lens1)
{
    auto s1 = migraphx::make_shape(migraphx::index_ints<2, 2>{}, migraphx::index_ints<1, 2>{});
    auto s2 = with_lens(s1, migraphx::index_ints<4, 3>{});
    EXPECT(s2.transposed());
    auto s3 = migraphx::make_shape(migraphx::index_ints<4, 3>{}, migraphx::index_ints<1, 4>{});
    EXPECT(s2 == s3);
}

TEST_CASE(test_with_lens2)
{
    auto s1 = migraphx::make_shape(migraphx::index_ints<2, 2>{}, migraphx::index_ints<2, 1>{});
    auto s2 = with_lens(s1, migraphx::index_ints<3, 4>{});
    EXPECT(s2.standard());
    auto s3 = migraphx::make_shape(migraphx::index_ints<3, 4>{});
    EXPECT(s2 == s3);
}

TEST_CASE(test_with_lens_ambigous1)
{
    auto s1 = migraphx::make_shape(migraphx::index_ints<64, 1, 24, 24>{});
    auto s2 = with_lens(s1, migraphx::index_ints<64, 3, 24, 24>{});
    EXPECT(not s2.transposed());
    auto s3 = migraphx::make_shape(migraphx::index_ints<64, 3, 24, 24>{});
    EXPECT(s2 == s3);
}

TEST_CASE(test_with_lens_ambigous2)
{
    auto s1 = migraphx::reorder_shape(migraphx::make_shape(migraphx::index_ints<64, 24, 24, 1>{}), migraphx::index_ints<0, 3, 1, 2>{});
    auto s2 = with_lens(s1, migraphx::index_ints<64, 3, 24, 24>{});
    EXPECT(s2.transposed());
    auto s3 =
        migraphx::reorder_shape(migraphx::make_shape(migraphx::index_ints<64, 24, 24, 3>{}), migraphx::index_ints<0, 3, 1, 2>{});
    EXPECT(s2 == s3);
}

TEST_CASE(test_with_lens_ambigous3)
{
    auto s1 = migraphx::make_shape(migraphx::index_ints<64, 3, 1, 1>{});
    auto s2 = with_lens(s1, migraphx::index_ints<64, 3, 24, 24>{});
    EXPECT(not s2.transposed());
    auto s3 = migraphx::make_shape(migraphx::index_ints<64, 3, 24, 24>{});
    EXPECT(s2 == s3);
}

TEST_CASE(test_with_lens_ambigous4)
{
    auto s1 = migraphx::reorder_shape(migraphx::make_shape(migraphx::index_ints<64, 1, 1, 3>{}), migraphx::index_ints<0, 3, 1, 2>{});
    auto s2 = with_lens(s1, migraphx::index_ints<64, 3, 24, 24>{});
    EXPECT(s2.transposed());
    auto s3 =
        migraphx::reorder_shape(migraphx::make_shape(migraphx::index_ints<64, 24, 24, 3>{}), migraphx::index_ints<0, 3, 1, 2>{});
    EXPECT(s2 == s3);
}

TEST_CASE(test_with_lens_ambigous5)
{
    auto s1 = migraphx::make_shape(migraphx::index_ints<1, 5, 24, 24>{});
    auto s2 = with_lens(s1, migraphx::index_ints<64, 3, 24, 24>{});
    EXPECT(not s2.transposed());
    auto s3 = migraphx::make_shape(migraphx::index_ints<64, 3, 24, 24>{});
    EXPECT(s2 == s3);
}

TEST_CASE(test_with_lens_ambigous6)
{
    auto s1 = migraphx::reorder_shape(migraphx::make_shape(migraphx::index_ints<1, 24, 24, 5>{}), migraphx::index_ints<0, 3, 1, 2>{});
    auto s2 = with_lens(s1, migraphx::index_ints<64, 3, 24, 24>{});
    EXPECT(s2.transposed());
    auto s3 =
        migraphx::reorder_shape(migraphx::make_shape(migraphx::index_ints<64, 24, 24, 3>{}), migraphx::index_ints<0, 3, 1, 2>{});
    EXPECT(s2 == s3);
}

TEST_CASE(test_with_lens_ambigous7)
{
    auto s1 = migraphx::reorder_shape(migraphx::make_shape(migraphx::index_ints<1, 1, 1, 3>{}), migraphx::index_ints<0, 3, 1, 2>{});
    auto s2 = with_lens(s1, migraphx::index_ints<64, 3, 24, 24>{});
    EXPECT(s2.transposed());
    auto s3 =
        migraphx::reorder_shape(migraphx::make_shape(migraphx::index_ints<64, 24, 24, 3>{}), migraphx::index_ints<0, 3, 1, 2>{});
    EXPECT(s2 == s3);
}

TEST_CASE(test_with_lens_ambigous8)
{
    auto s1 = migraphx::make_shape(migraphx::index_ints<1, 1, 24, 24>{});
    auto s2 = with_lens(s1, migraphx::index_ints<64, 3, 24, 24>{});
    EXPECT(not s2.transposed());
    auto s3 = migraphx::make_shape(migraphx::index_ints<64, 3, 24, 24>{});
    EXPECT(s2 == s3);
}

TEST_CASE(test_with_lens_ambigous9)
{
    auto s1 = migraphx::reorder_shape(migraphx::make_shape(migraphx::index_ints<1, 24, 24, 1>{}), migraphx::index_ints<0, 3, 1, 2>{});
    auto s2 = with_lens(s1, migraphx::index_ints<64, 3, 24, 24>{});
    EXPECT(s2.transposed());
    auto s3 =
        migraphx::reorder_shape(migraphx::make_shape(migraphx::index_ints<64, 24, 24, 3>{}), migraphx::index_ints<0, 3, 1, 2>{});
    EXPECT(s2 == s3);
}

TEST_CASE(test_with_lens_ambigous10)
{
    auto s1 = migraphx::make_shape(migraphx::index_ints<3, 2, 4, 1>{});
    auto s2 = with_lens(s1, migraphx::index_ints<3, 2, 4, 1>{});
    EXPECT(not s2.transposed());
    auto s3 = migraphx::make_shape(migraphx::index_ints<3, 2, 4, 1>{});
    EXPECT(s2 == s3);
}

TEST_CASE(test_with_lens_ambigous11)
{
    auto s1 = migraphx::make_shape(migraphx::index_ints<64, 1, 1, 1>{});
    auto s2 = with_lens(s1, migraphx::index_ints<64, 3, 24, 24>{});
    EXPECT(s1.standard());
    EXPECT(s2.standard());
    auto s3 = migraphx::make_shape(migraphx::index_ints<64, 3, 24, 24>{});
    EXPECT(s2 == s3);
}

TEST_CASE(test_with_lens_ambigous12)
{
    auto s1 = migraphx::make_shape(migraphx::index_ints<1, 64, 1, 1>{});
    auto s2 = with_lens(s1, migraphx::index_ints<64, 3, 24, 24>{});
    EXPECT(s1.standard());
    EXPECT(s2.standard());
    auto s3 = migraphx::make_shape(migraphx::index_ints<64, 3, 24, 24>{});
    EXPECT(s2 == s3);
}

TEST_CASE(test_with_lens_ambigous13)
{
    auto s1 = migraphx::reorder_shape(migraphx::make_shape(migraphx::index_ints<1, 1, 1, 3>{}), migraphx::index_ints<0, 3, 1, 2>{});
    auto s2 = with_lens(s1, migraphx::index_ints<64, 3, 24, 24>{});
    EXPECT(s2.transposed());
    auto s3 =
        migraphx::reorder_shape(migraphx::make_shape(migraphx::index_ints<64, 24, 24, 3>{}), migraphx::index_ints<0, 3, 1, 2>{});
    EXPECT(s2 == s3);
}

TEST_CASE(test_multi_index)
{
    auto s = migraphx::make_shape(migraphx::index_ints<2, 4, 6>{});
    EXPECT(s.multi(0) == migraphx::index_ints<0, 0, 0>{});
    EXPECT(s.multi(4) == migraphx::index_ints<0, 0, 4>{});
    EXPECT(s.multi(6) == migraphx::index_ints<0, 1, 0>{});
    EXPECT(s.multi(8) == migraphx::index_ints<0, 1, 2>{});
    EXPECT(s.multi(24) == migraphx::index_ints<1, 0, 0>{});
    EXPECT(s.multi(30) == migraphx::index_ints<1, 1, 0>{});
    EXPECT(s.multi(34) == migraphx::index_ints<1, 1, 4>{});
}

TEST_CASE(test_single_index)
{
    auto s = migraphx::make_shape(migraphx::index_ints<2, 4, 6>{});
    EXPECT(0 == s.single({0, 0, 0}));
    EXPECT(4 == s.single({0, 0, 4}));
    EXPECT(6 == s.single({0, 1, 0}));
    EXPECT(8 == s.single({0, 1, 2}));
    EXPECT(24 == s.single({1, 0, 0}));
    EXPECT(30 == s.single({1, 1, 0}));
    EXPECT(34 == s.single({1, 1, 4}));
}

TEST_CASE(find_permutation_2d_standard)
{
    auto s                = migraphx::make_shape(migraphx::index_ints<2, 3>{});
    auto permutation = migraphx::index_ints<0, 1>{};
    EXPECT(migraphx::find_permutation(s) == permutation);
}

TEST_CASE(find_permutation_2d_transpose)
{
    auto s                = migraphx::make_shape(migraphx::index_ints<2, 3>{}, migraphx::index_ints<1, 2>{});
    auto permutation = migraphx::index_ints<1, 0>{};
    EXPECT(migraphx::find_permutation(s) == permutation);
}

TEST_CASE(find_permutation_3d)
{
    auto s                = migraphx::make_shape(migraphx::index_ints<2, 3, 4>{}, migraphx::index_ints<1, 8, 2>{});
    auto permutation = migraphx::index_ints<1, 2, 0>{};
    EXPECT(migraphx::find_permutation(s) == permutation);
}

TEST_CASE(find_permutation_4d)
{
    // ori_lens = 2, 3, 4, 5
    // ori_strides = 60, 20, 5, 1
    // perm = 3, 2, 0, 1
    // inv_perm = 2, 3, 1, 0
    // out_strides = 5, 1, 20, 60
    auto s                = migraphx::make_shape(migraphx::index_ints<5, 4, 2, 3>{}, migraphx::index_ints<5, 1, 20, 60>{});
    auto permutation = migraphx::index_ints<3, 2, 0, 1>{};
    EXPECT(migraphx::find_permutation(s) == permutation);
}

TEST_CASE(from_2d_permutation)
{
    auto out_lens = migraphx::index_ints<2, 3>{};
    auto permutation = migraphx::index_ints<1, 0>{};
    auto out_shape =
        migraphx::make_shape_from_permutation(out_lens, permutation);
    EXPECT(out_shape.lens == out_lens);
    EXPECT(migraphx::find_permutation(out_shape) == permutation);
}

TEST_CASE(from_3d_permutation)
{
    auto out_lens = migraphx::index_ints<2, 3, 4>{};
    auto permutation = migraphx::index_ints<1, 2, 0>{};
    auto out_shape =
        migraphx::make_shape_from_permutation(out_lens, permutation);
    EXPECT(out_shape.lens == out_lens);
    EXPECT(migraphx::find_permutation(out_shape) == permutation);
}

TEST_CASE(from_4d_permutation)
{
    auto out_lens = migraphx::index_ints<5, 4, 2, 3>{};
    auto permutation = migraphx::index_ints<3, 2, 0, 1>{};
    auto out_shape =
        migraphx::make_shape_from_permutation(out_lens, permutation);
    EXPECT(out_shape.lens == out_lens);
    EXPECT(migraphx::find_permutation(out_shape) == permutation);
}

