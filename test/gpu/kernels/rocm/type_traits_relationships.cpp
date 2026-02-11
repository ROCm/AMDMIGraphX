
#include <rocm/type_traits.hpp>
#include <migraphx/kernels/test.hpp>

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wnon-virtual-dtor"

namespace tt_rel {

struct base
{
};

struct derived : base
{
};

struct derived2 : base
{
};

struct multi_derived : derived, derived2
{
};

struct private_derived : private base
{
};

struct unrelated
{
};

struct virtual_base
{
    virtual void foo();
};

struct virtual_derived : virtual_base
{
};

struct convertible_to_int
{
    operator int();
};

struct convertible_from_int
{
    convertible_from_int(int);
};

struct trivial_class
{
};

struct non_trivial_copy
{
    non_trivial_copy() = default;
    non_trivial_copy(const non_trivial_copy&);
};

struct non_trivial_assign
{
    non_trivial_assign& operator=(const non_trivial_assign&);
};

struct nothrow_default
{
    nothrow_default() noexcept;
};

struct throwing_default
{
    throwing_default() noexcept(false);
};

struct nothrow_copy
{
    nothrow_copy() = default;
    nothrow_copy(const nothrow_copy&) noexcept;
};

struct throwing_copy
{
    throwing_copy() = default;
    throwing_copy(const throwing_copy&) noexcept(false);
};

struct nothrow_assign
{
    nothrow_assign& operator=(const nothrow_assign&) noexcept;
};

struct throwing_assign
{
    throwing_assign& operator=(const throwing_assign&) noexcept(false);
};

enum test_enum
{
    val_a,
    val_b
};

using func_t = void();

} // namespace tt_rel

#pragma clang diagnostic pop

TEST_CASE(is_same)
{
    EXPECT(rocm::is_same<int, int>{});
    EXPECT(rocm::is_same<float, float>{});
    EXPECT(rocm::is_same<void, void>{});
    EXPECT(rocm::is_same<const int, const int>{});
    EXPECT(rocm::is_same<volatile int, volatile int>{});
    EXPECT(rocm::is_same<const volatile int, const volatile int>{});
    EXPECT(rocm::is_same<int*, int*>{});
    EXPECT(rocm::is_same<int&, int&>{});
    EXPECT(rocm::is_same<int&&, int&&>{});
    EXPECT(rocm::is_same<tt_rel::base, tt_rel::base>{});
    EXPECT(rocm::is_same<int[2], int[2]>{});

    EXPECT(not rocm::is_same<int, float>{});
    EXPECT(not rocm::is_same<int, const int>{});
    EXPECT(not rocm::is_same<int, volatile int>{});
    EXPECT(not rocm::is_same<int, int&>{});
    EXPECT(not rocm::is_same<int, int&&>{});
    EXPECT(not rocm::is_same<int, int*>{});
    EXPECT(not rocm::is_same<tt_rel::base, tt_rel::derived>{});
    EXPECT(not rocm::is_same<int, unsigned int>{});
    EXPECT(not rocm::is_same<int, long>{});
    EXPECT(not rocm::is_same<float, double>{});
}

TEST_CASE(is_base_of)
{
    EXPECT(rocm::is_base_of<tt_rel::base, tt_rel::base>{});
    EXPECT(rocm::is_base_of<tt_rel::base, tt_rel::derived>{});
    EXPECT(rocm::is_base_of<tt_rel::base, tt_rel::derived2>{});
    EXPECT(rocm::is_base_of<tt_rel::derived, tt_rel::multi_derived>{});
    EXPECT(rocm::is_base_of<tt_rel::derived2, tt_rel::multi_derived>{});
    EXPECT(rocm::is_base_of<tt_rel::base, tt_rel::private_derived>{});
    EXPECT(rocm::is_base_of<tt_rel::virtual_base, tt_rel::virtual_derived>{});
    EXPECT(rocm::is_base_of<const tt_rel::base, const tt_rel::derived>{});
    EXPECT(rocm::is_base_of<tt_rel::derived, tt_rel::derived>{});

    EXPECT(not rocm::is_base_of<tt_rel::derived, tt_rel::base>{});
    EXPECT(not rocm::is_base_of<tt_rel::base, tt_rel::unrelated>{});
    EXPECT(not rocm::is_base_of<tt_rel::unrelated, tt_rel::base>{});
    EXPECT(not rocm::is_base_of<int, int>{});
    EXPECT(not rocm::is_base_of<int, float>{});
    EXPECT(not rocm::is_base_of<void, void>{});
    EXPECT(not rocm::is_base_of<tt_rel::base, int>{});
}

TEST_CASE(is_convertible)
{
    EXPECT(rocm::is_convertible<int, int>{});
    EXPECT(rocm::is_convertible<int, float>{});
    EXPECT(rocm::is_convertible<float, double>{});
    EXPECT(rocm::is_convertible<int, long>{});
    EXPECT(rocm::is_convertible<int, long long>{});
    EXPECT(rocm::is_convertible<double, int>{});
    EXPECT(rocm::is_convertible<tt_rel::derived*, tt_rel::base*>{});
    EXPECT(rocm::is_convertible<tt_rel::derived&, tt_rel::base&>{});
    EXPECT(rocm::is_convertible<int, const int&>{});
    EXPECT(rocm::is_convertible<int*, void*>{});
    EXPECT(rocm::is_convertible<int, bool>{});
    EXPECT(rocm::is_convertible<float*, void*>{});
    EXPECT(rocm::is_convertible<tt_rel::convertible_to_int, int>{});

    EXPECT(not rocm::is_convertible<void, int>{});
    EXPECT(not rocm::is_convertible<int, void>{});
    EXPECT(not rocm::is_convertible<tt_rel::base*, tt_rel::derived*>{});
    EXPECT(not rocm::is_convertible<int*, float*>{});
    EXPECT(not rocm::is_convertible<tt_rel::func_t, tt_rel::func_t>{});
    EXPECT(not rocm::is_convertible<int[2], int[2]>{});
    EXPECT(not rocm::is_convertible<tt_rel::unrelated, tt_rel::base>{});
    EXPECT(not rocm::is_convertible<void*, int*>{});
    EXPECT(not rocm::is_convertible<const int*, int*>{});
    EXPECT(not rocm::is_convertible<volatile int*, int*>{});
}

TEST_CASE(is_assignable)
{
    EXPECT(rocm::is_assignable<int&, int>{});
    EXPECT(rocm::is_assignable<int&, float>{});
    EXPECT(rocm::is_assignable<int&, double>{});
    EXPECT(rocm::is_assignable<int&, short>{});
    EXPECT(rocm::is_assignable<float&, int>{});
    EXPECT(rocm::is_assignable<tt_rel::nothrow_assign&, const tt_rel::nothrow_assign&>{});
    EXPECT(rocm::is_assignable<tt_rel::throwing_assign&, const tt_rel::throwing_assign&>{});

    EXPECT(not rocm::is_assignable<int, int>{});
    EXPECT(not rocm::is_assignable<const int&, int>{});
    EXPECT(not rocm::is_assignable<int&, void>{});
    EXPECT(not rocm::is_assignable<void, void>{});
    EXPECT(not rocm::is_assignable<int&, int*>{});
    EXPECT(not rocm::is_assignable<int&, int[2]>{});
}

TEST_CASE(is_nothrow_assignable)
{
    EXPECT(rocm::is_nothrow_assignable<int&, int>{});
    EXPECT(rocm::is_nothrow_assignable<int&, float>{});
    EXPECT(rocm::is_nothrow_assignable<int&, double>{});
    EXPECT(rocm::is_nothrow_assignable<float&, int>{});
    EXPECT(rocm::is_nothrow_assignable<tt_rel::nothrow_assign&, const tt_rel::nothrow_assign&>{});

    EXPECT(not rocm::is_nothrow_assignable<tt_rel::throwing_assign&, const tt_rel::throwing_assign&>{});
    EXPECT(not rocm::is_nothrow_assignable<int, int>{});
    EXPECT(not rocm::is_nothrow_assignable<void, void>{});
    EXPECT(not rocm::is_nothrow_assignable<const int&, int>{});
}

TEST_CASE(is_trivially_assignable)
{
    EXPECT(rocm::is_trivially_assignable<int&, int>{});
    EXPECT(rocm::is_trivially_assignable<int&, float>{});
    EXPECT(rocm::is_trivially_assignable<float&, int>{});
    EXPECT(rocm::is_trivially_assignable<tt_rel::trivial_class&, const tt_rel::trivial_class&>{});

    EXPECT(not rocm::is_trivially_assignable<tt_rel::non_trivial_assign&, const tt_rel::non_trivial_assign&>{});
    EXPECT(not rocm::is_trivially_assignable<int, int>{});
    EXPECT(not rocm::is_trivially_assignable<void, void>{});
    EXPECT(not rocm::is_trivially_assignable<const int&, int>{});
}

TEST_CASE(is_constructible)
{
    EXPECT(rocm::is_constructible<int>{});
    EXPECT(rocm::is_constructible<float>{});
    EXPECT(rocm::is_constructible<tt_rel::base>{});
    EXPECT(rocm::is_constructible<tt_rel::trivial_class>{});
    EXPECT(rocm::is_constructible<int, int>{});
    EXPECT(rocm::is_constructible<int, float>{});
    EXPECT(rocm::is_constructible<float, int>{});
    EXPECT(rocm::is_constructible<tt_rel::base, const tt_rel::base&>{});
    EXPECT(rocm::is_constructible<tt_rel::convertible_from_int, int>{});
    EXPECT(rocm::is_constructible<int*, int*>{});

    EXPECT(not rocm::is_constructible<void>{});
    EXPECT(not rocm::is_constructible<tt_rel::func_t>{});
    EXPECT(not rocm::is_constructible<tt_rel::base, tt_rel::unrelated>{});
    EXPECT(not rocm::is_constructible<int, int*>{});
    EXPECT(not rocm::is_constructible<int*, float*>{});
}

TEST_CASE(is_nothrow_constructible)
{
    EXPECT(rocm::is_nothrow_constructible<int>{});
    EXPECT(rocm::is_nothrow_constructible<float>{});
    EXPECT(rocm::is_nothrow_constructible<int*>{});
    EXPECT(rocm::is_nothrow_constructible<int, int>{});
    EXPECT(rocm::is_nothrow_constructible<float, int>{});
    EXPECT(rocm::is_nothrow_constructible<tt_rel::nothrow_default>{});
    EXPECT(rocm::is_nothrow_constructible<tt_rel::nothrow_copy, const tt_rel::nothrow_copy&>{});
    EXPECT(rocm::is_nothrow_constructible<tt_rel::trivial_class>{});

    EXPECT(not rocm::is_nothrow_constructible<tt_rel::throwing_default>{});
    EXPECT(not rocm::is_nothrow_constructible<tt_rel::throwing_copy, const tt_rel::throwing_copy&>{});
    EXPECT(not rocm::is_nothrow_constructible<void>{});
    EXPECT(not rocm::is_nothrow_constructible<tt_rel::func_t>{});
}

TEST_CASE(is_trivially_constructible)
{
    EXPECT(rocm::is_trivially_constructible<int>{});
    EXPECT(rocm::is_trivially_constructible<float>{});
    EXPECT(rocm::is_trivially_constructible<int*>{});
    EXPECT(rocm::is_trivially_constructible<tt_rel::trivial_class>{});
    EXPECT(rocm::is_trivially_constructible<int, int>{});
    EXPECT(rocm::is_trivially_constructible<float, float>{});
    EXPECT(rocm::is_trivially_constructible<tt_rel::trivial_class, const tt_rel::trivial_class&>{});
    EXPECT(rocm::is_trivially_constructible<tt_rel::test_enum>{});

    EXPECT(not rocm::is_trivially_constructible<tt_rel::non_trivial_copy, const tt_rel::non_trivial_copy&>{});
    EXPECT(not rocm::is_trivially_constructible<void>{});
    EXPECT(not rocm::is_trivially_constructible<tt_rel::func_t>{});
    EXPECT(not rocm::is_trivially_constructible<tt_rel::convertible_from_int, int>{});
}
