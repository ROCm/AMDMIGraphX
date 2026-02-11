
#include <rocm/type_traits.hpp>
#include <migraphx/kernels/test.hpp>

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wnon-virtual-dtor"

namespace tt_prop {

struct empty_class
{
};

struct non_empty_class
{
    int x;
};

struct pod_class
{
    int x;
    double y;
};

union test_union
{
    int x;
    double y;
};

struct abstract_base
{
    virtual void foo() = 0;
};

struct abstract_derived : abstract_base
{
    virtual void bar() = 0;
};

struct concrete_override : abstract_base
{
    void foo() override {}
};

struct still_abstract : abstract_base
{
    virtual void bar() = 0;
    void foo() override {}
};

struct polymorphic_class
{
    virtual void foo();
};

struct polymorphic_derived : polymorphic_class
{
};

struct non_polymorphic
{
    void foo();
};

struct final_class final
{
};

struct final_derived final : non_polymorphic
{
};

struct non_aggregate
{
    non_aggregate(int);
};

struct trivial_class
{
};

struct non_trivial
{
    non_trivial();
    ~non_trivial();
};

struct non_trivially_copyable
{
    non_trivially_copyable(const non_trivially_copyable&);
};

struct standard_layout_class
{
    int x;
    int y;
};

struct non_standard_layout
{
    virtual void f();
    int x;
};

enum test_enum
{
    val_a,
    val_b
};

using func_t   = void();
using func_ptr = void (*)();

} // namespace tt_prop

#pragma clang diagnostic pop

TEST_CASE(is_const)
{
    EXPECT(rocm::is_const<const int>{});
    EXPECT(rocm::is_const<const volatile int>{});
    EXPECT(rocm::is_const<int* const>{});
    EXPECT(rocm::is_const<int* const volatile>{});
    EXPECT(rocm::is_const<const tt_prop::empty_class>{});
    EXPECT(rocm::is_const<const int[2]>{});
    EXPECT(rocm::is_const<const volatile tt_prop::empty_class>{});

    EXPECT(not rocm::is_const<int>{});
    EXPECT(not rocm::is_const<volatile int>{});
    EXPECT(not rocm::is_const<int*>{});
    EXPECT(not rocm::is_const<const int*>{});
    EXPECT(not rocm::is_const<int&>{});
    EXPECT(not rocm::is_const<const int&>{});
    EXPECT(not rocm::is_const<int&&>{});
    EXPECT(not rocm::is_const<tt_prop::empty_class>{});
    EXPECT(not rocm::is_const<void>{});
    EXPECT(not rocm::is_const<tt_prop::func_t>{});
}

TEST_CASE(is_volatile)
{
    EXPECT(rocm::is_volatile<volatile int>{});
    EXPECT(rocm::is_volatile<const volatile int>{});
    EXPECT(rocm::is_volatile<int* volatile>{});
    EXPECT(rocm::is_volatile<int* const volatile>{});
    EXPECT(rocm::is_volatile<volatile tt_prop::empty_class>{});
    EXPECT(rocm::is_volatile<volatile int[2]>{});
    EXPECT(rocm::is_volatile<const volatile tt_prop::empty_class>{});

    EXPECT(not rocm::is_volatile<int>{});
    EXPECT(not rocm::is_volatile<const int>{});
    EXPECT(not rocm::is_volatile<int*>{});
    EXPECT(not rocm::is_volatile<volatile int*>{});
    EXPECT(not rocm::is_volatile<int&>{});
    EXPECT(not rocm::is_volatile<volatile int&>{});
    EXPECT(not rocm::is_volatile<int&&>{});
    EXPECT(not rocm::is_volatile<void>{});
    EXPECT(not rocm::is_volatile<tt_prop::func_t>{});
}

TEST_CASE(is_abstract)
{
    EXPECT(rocm::is_abstract<tt_prop::abstract_base>{});
    EXPECT(rocm::is_abstract<const tt_prop::abstract_base>{});
    EXPECT(rocm::is_abstract<volatile tt_prop::abstract_base>{});
    EXPECT(rocm::is_abstract<const volatile tt_prop::abstract_base>{});
    EXPECT(rocm::is_abstract<tt_prop::abstract_derived>{});
    EXPECT(rocm::is_abstract<tt_prop::still_abstract>{});

    EXPECT(not rocm::is_abstract<tt_prop::concrete_override>{});
    EXPECT(not rocm::is_abstract<int>{});
    EXPECT(not rocm::is_abstract<void>{});
    EXPECT(not rocm::is_abstract<tt_prop::empty_class>{});
    EXPECT(not rocm::is_abstract<tt_prop::non_polymorphic>{});
    EXPECT(not rocm::is_abstract<tt_prop::polymorphic_class>{});
    EXPECT(not rocm::is_abstract<int&>{});
    EXPECT(not rocm::is_abstract<int&&>{});
    EXPECT(not rocm::is_abstract<int*>{});
    EXPECT(not rocm::is_abstract<int[2]>{});
    EXPECT(not rocm::is_abstract<tt_prop::test_enum>{});
    EXPECT(not rocm::is_abstract<tt_prop::func_t>{});
    EXPECT(not rocm::is_abstract<tt_prop::func_ptr>{});
}

TEST_CASE(is_aggregate)
{
    EXPECT(rocm::is_aggregate<tt_prop::empty_class>{});
    EXPECT(rocm::is_aggregate<tt_prop::pod_class>{});
    EXPECT(rocm::is_aggregate<tt_prop::non_empty_class>{});
    EXPECT(rocm::is_aggregate<tt_prop::standard_layout_class>{});
    EXPECT(rocm::is_aggregate<tt_prop::test_union>{});
    EXPECT(rocm::is_aggregate<int[2]>{});
    EXPECT(rocm::is_aggregate<int[2][3]>{});
    EXPECT(rocm::is_aggregate<const tt_prop::pod_class>{});

    EXPECT(not rocm::is_aggregate<int>{});
    EXPECT(not rocm::is_aggregate<float>{});
    EXPECT(not rocm::is_aggregate<void>{});
    EXPECT(not rocm::is_aggregate<int*>{});
    EXPECT(not rocm::is_aggregate<int&>{});
    EXPECT(not rocm::is_aggregate<tt_prop::non_aggregate>{});
    EXPECT(not rocm::is_aggregate<tt_prop::test_enum>{});
}

TEST_CASE(is_empty)
{
    EXPECT(rocm::is_empty<tt_prop::empty_class>{});
    EXPECT(rocm::is_empty<const tt_prop::empty_class>{});
    EXPECT(rocm::is_empty<volatile tt_prop::empty_class>{});
    EXPECT(rocm::is_empty<const volatile tt_prop::empty_class>{});

    EXPECT(not rocm::is_empty<tt_prop::non_empty_class>{});
    EXPECT(not rocm::is_empty<tt_prop::pod_class>{});
    EXPECT(not rocm::is_empty<int>{});
    EXPECT(not rocm::is_empty<void>{});
    EXPECT(not rocm::is_empty<int*>{});
    EXPECT(not rocm::is_empty<int&>{});
    EXPECT(not rocm::is_empty<int[2]>{});
    EXPECT(not rocm::is_empty<tt_prop::test_enum>{});
    EXPECT(not rocm::is_empty<tt_prop::test_union>{});
    EXPECT(not rocm::is_empty<tt_prop::polymorphic_class>{});
}

TEST_CASE(is_final)
{
    EXPECT(rocm::is_final<tt_prop::final_class>{});
    EXPECT(rocm::is_final<tt_prop::final_derived>{});
    EXPECT(rocm::is_final<const tt_prop::final_class>{});
    EXPECT(rocm::is_final<volatile tt_prop::final_class>{});
    EXPECT(rocm::is_final<const volatile tt_prop::final_class>{});

    EXPECT(not rocm::is_final<tt_prop::empty_class>{});
    EXPECT(not rocm::is_final<tt_prop::non_polymorphic>{});
    EXPECT(not rocm::is_final<tt_prop::polymorphic_class>{});
    EXPECT(not rocm::is_final<int>{});
    EXPECT(not rocm::is_final<void>{});
    EXPECT(not rocm::is_final<tt_prop::test_enum>{});
}

TEST_CASE(is_polymorphic)
{
    EXPECT(rocm::is_polymorphic<tt_prop::polymorphic_class>{});
    EXPECT(rocm::is_polymorphic<const tt_prop::polymorphic_class>{});
    EXPECT(rocm::is_polymorphic<volatile tt_prop::polymorphic_class>{});
    EXPECT(rocm::is_polymorphic<const volatile tt_prop::polymorphic_class>{});
    EXPECT(rocm::is_polymorphic<tt_prop::polymorphic_derived>{});
    EXPECT(rocm::is_polymorphic<tt_prop::abstract_base>{});
    EXPECT(rocm::is_polymorphic<tt_prop::concrete_override>{});
    EXPECT(rocm::is_polymorphic<tt_prop::non_standard_layout>{});

    EXPECT(not rocm::is_polymorphic<tt_prop::empty_class>{});
    EXPECT(not rocm::is_polymorphic<tt_prop::non_polymorphic>{});
    EXPECT(not rocm::is_polymorphic<tt_prop::final_class>{});
    EXPECT(not rocm::is_polymorphic<int>{});
    EXPECT(not rocm::is_polymorphic<void>{});
    EXPECT(not rocm::is_polymorphic<int*>{});
    EXPECT(not rocm::is_polymorphic<int&>{});
    EXPECT(not rocm::is_polymorphic<int&&>{});
    EXPECT(not rocm::is_polymorphic<tt_prop::test_enum>{});
    EXPECT(not rocm::is_polymorphic<tt_prop::func_t>{});
}

TEST_CASE(is_standard_layout)
{
    EXPECT(rocm::is_standard_layout<int>{});
    EXPECT(rocm::is_standard_layout<float>{});
    EXPECT(rocm::is_standard_layout<double>{});
    EXPECT(rocm::is_standard_layout<tt_prop::standard_layout_class>{});
    EXPECT(rocm::is_standard_layout<tt_prop::pod_class>{});
    EXPECT(rocm::is_standard_layout<tt_prop::empty_class>{});
    EXPECT(rocm::is_standard_layout<int*>{});
    EXPECT(rocm::is_standard_layout<tt_prop::test_enum>{});
    EXPECT(rocm::is_standard_layout<int[2]>{});
    EXPECT(rocm::is_standard_layout<const int>{});
    EXPECT(rocm::is_standard_layout<tt_prop::trivial_class>{});

    EXPECT(not rocm::is_standard_layout<tt_prop::non_standard_layout>{});
    EXPECT(not rocm::is_standard_layout<tt_prop::polymorphic_class>{});
    EXPECT(not rocm::is_standard_layout<int&>{});
}

TEST_CASE(is_trivial)
{
    EXPECT(rocm::is_trivial<int>{});
    EXPECT(rocm::is_trivial<float>{});
    EXPECT(rocm::is_trivial<double>{});
    EXPECT(rocm::is_trivial<tt_prop::trivial_class>{});
    EXPECT(rocm::is_trivial<tt_prop::pod_class>{});
    EXPECT(rocm::is_trivial<tt_prop::empty_class>{});
    EXPECT(rocm::is_trivial<int*>{});
    EXPECT(rocm::is_trivial<tt_prop::test_enum>{});
    EXPECT(rocm::is_trivial<const int>{});
    EXPECT(rocm::is_trivial<int[2]>{});

    EXPECT(not rocm::is_trivial<tt_prop::non_trivial>{});
    EXPECT(not rocm::is_trivial<tt_prop::non_trivially_copyable>{});
    EXPECT(not rocm::is_trivial<int&>{});
    EXPECT(not rocm::is_trivial<int&&>{});
}

TEST_CASE(is_trivially_copyable)
{
    EXPECT(rocm::is_trivially_copyable<int>{});
    EXPECT(rocm::is_trivially_copyable<float>{});
    EXPECT(rocm::is_trivially_copyable<double>{});
    EXPECT(rocm::is_trivially_copyable<tt_prop::trivial_class>{});
    EXPECT(rocm::is_trivially_copyable<tt_prop::pod_class>{});
    EXPECT(rocm::is_trivially_copyable<tt_prop::empty_class>{});
    EXPECT(rocm::is_trivially_copyable<int*>{});
    EXPECT(rocm::is_trivially_copyable<tt_prop::test_enum>{});
    EXPECT(rocm::is_trivially_copyable<const int>{});

    EXPECT(not rocm::is_trivially_copyable<tt_prop::non_trivial>{});
    EXPECT(not rocm::is_trivially_copyable<tt_prop::non_trivially_copyable>{});
    EXPECT(not rocm::is_trivially_copyable<int&>{});
    EXPECT(not rocm::is_trivially_copyable<int&&>{});
}

TEST_CASE(is_trivially_destructible)
{
    EXPECT(rocm::is_trivially_destructible<int>{});
    EXPECT(rocm::is_trivially_destructible<float>{});
    EXPECT(rocm::is_trivially_destructible<double>{});
    EXPECT(rocm::is_trivially_destructible<tt_prop::trivial_class>{});
    EXPECT(rocm::is_trivially_destructible<tt_prop::pod_class>{});
    EXPECT(rocm::is_trivially_destructible<tt_prop::empty_class>{});
    EXPECT(rocm::is_trivially_destructible<int*>{});
    EXPECT(rocm::is_trivially_destructible<tt_prop::test_enum>{});
    EXPECT(rocm::is_trivially_destructible<const int>{});
    EXPECT(rocm::is_trivially_destructible<int[2]>{});

    EXPECT(not rocm::is_trivially_destructible<tt_prop::non_trivial>{});
}

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"

TEST_CASE(is_pod)
{
    EXPECT(rocm::is_pod<int>{});
    EXPECT(rocm::is_pod<float>{});
    EXPECT(rocm::is_pod<double>{});
    EXPECT(rocm::is_pod<tt_prop::pod_class>{});
    EXPECT(rocm::is_pod<int*>{});
    EXPECT(rocm::is_pod<tt_prop::test_enum>{});
    EXPECT(rocm::is_pod<int[2]>{});
    EXPECT(rocm::is_pod<const int>{});
    EXPECT(rocm::is_pod<tt_prop::empty_class>{});
    EXPECT(rocm::is_pod<tt_prop::trivial_class>{});

    EXPECT(not rocm::is_pod<tt_prop::non_trivial>{});
    EXPECT(not rocm::is_pod<tt_prop::polymorphic_class>{});
    EXPECT(not rocm::is_pod<tt_prop::non_standard_layout>{});
    EXPECT(not rocm::is_pod<int&>{});
    EXPECT(not rocm::is_pod<int&&>{});
}

TEST_CASE(is_literal_type)
{
    EXPECT(rocm::is_literal_type<int>{});
    EXPECT(rocm::is_literal_type<float>{});
    EXPECT(rocm::is_literal_type<double>{});
    EXPECT(rocm::is_literal_type<tt_prop::trivial_class>{});
    EXPECT(rocm::is_literal_type<tt_prop::empty_class>{});
    EXPECT(rocm::is_literal_type<int*>{});
    EXPECT(rocm::is_literal_type<tt_prop::test_enum>{});
    EXPECT(rocm::is_literal_type<int&>{});
    EXPECT(rocm::is_literal_type<int[2]>{});
    EXPECT(rocm::is_literal_type<const int>{});

    EXPECT(not rocm::is_literal_type<tt_prop::non_trivial>{});
}

#pragma clang diagnostic pop

TEST_CASE(is_unsigned)
{
    EXPECT(rocm::is_unsigned<bool>{});
    EXPECT(rocm::is_unsigned<unsigned char>{});
    EXPECT(rocm::is_unsigned<unsigned short>{});
    EXPECT(rocm::is_unsigned<unsigned int>{});
    EXPECT(rocm::is_unsigned<unsigned long>{});
    EXPECT(rocm::is_unsigned<unsigned long long>{});
    EXPECT(rocm::is_unsigned<const unsigned int>{});
    EXPECT(rocm::is_unsigned<volatile unsigned int>{});
    EXPECT(rocm::is_unsigned<const volatile unsigned int>{});

    EXPECT(not rocm::is_unsigned<int>{});
    EXPECT(not rocm::is_unsigned<signed char>{});
    EXPECT(not rocm::is_unsigned<short>{});
    EXPECT(not rocm::is_unsigned<long>{});
    EXPECT(not rocm::is_unsigned<long long>{});
    EXPECT(not rocm::is_unsigned<float>{});
    EXPECT(not rocm::is_unsigned<double>{});
    EXPECT(not rocm::is_unsigned<void>{});
    EXPECT(not rocm::is_unsigned<tt_prop::empty_class>{});
    EXPECT(not rocm::is_unsigned<int*>{});
}
