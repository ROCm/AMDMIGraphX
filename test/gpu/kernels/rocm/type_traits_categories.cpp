
#include <rocm/type_traits.hpp>
#include <migraphx/kernels/test.hpp>

namespace tt_cat {

enum test_enum
{
    val_a,
    val_b
};

enum class scoped_enum
{
    x,
    y
};

struct simple_class
{
    int x;
};

union simple_union
{
    int x;
    double y;
};

struct udt
{
    int member_var;
    void member_func();
    int member_func_int(int);
    int member_func_float(int, float);
};

struct incomplete_type;

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wredundant-parens"
using func_ptr    = void (*)();
using func_ptr_i  = int (*)(int);
using mfp_void    = void (udt::*)();
using mfp_int     = int (udt::*)();
using mfp_int_arg = int (udt::*)(int);
using mfp_int_flt = int (udt::*)(int, float);
using mop         = int(udt::*);
using cmfp        = int (udt::*)(int) const;
#pragma clang diagnostic pop

using func_t0 = void();
using func_t1 = void(int);
using func_t2 = void(int&, double);
using func_t3 = void(int&, bool, int, int);
using func_t4 = void(int, bool, int*, int[], int, int, int, int, int);

} // namespace tt_cat

// ---------- primary type categories ----------

TEST_CASE(is_integral)
{
    EXPECT(rocm::is_integral<bool>{});
    EXPECT(rocm::is_integral<char>{});
    EXPECT(rocm::is_integral<signed char>{});
    EXPECT(rocm::is_integral<unsigned char>{});
    EXPECT(rocm::is_integral<wchar_t>{});
    EXPECT(rocm::is_integral<short>{});
    EXPECT(rocm::is_integral<unsigned short>{});
    EXPECT(rocm::is_integral<int>{});
    EXPECT(rocm::is_integral<unsigned int>{});
    EXPECT(rocm::is_integral<long>{});
    EXPECT(rocm::is_integral<unsigned long>{});
    EXPECT(rocm::is_integral<long long>{});
    EXPECT(rocm::is_integral<unsigned long long>{});
    EXPECT(rocm::is_integral<const int>{});
    EXPECT(rocm::is_integral<volatile int>{});
    EXPECT(rocm::is_integral<const volatile int>{});
    EXPECT(rocm::is_integral<const bool>{});
    EXPECT(rocm::is_integral<volatile char>{});
    EXPECT(rocm::is_integral<const volatile unsigned long>{});

    EXPECT(not rocm::is_integral<float>{});
    EXPECT(not rocm::is_integral<double>{});
    EXPECT(not rocm::is_integral<long double>{});
    EXPECT(not rocm::is_integral<void>{});
    EXPECT(not rocm::is_integral<int*>{});
    EXPECT(not rocm::is_integral<int&>{});
    EXPECT(not rocm::is_integral<int&&>{});
    EXPECT(not rocm::is_integral<int[2]>{});
    EXPECT(not rocm::is_integral<tt_cat::simple_class>{});
    EXPECT(not rocm::is_integral<tt_cat::test_enum>{});
    EXPECT(not rocm::is_integral<tt_cat::func_t0>{});
    EXPECT(not rocm::is_integral<tt_cat::func_ptr>{});
    EXPECT(not rocm::is_integral<tt_cat::mop>{});
    EXPECT(not rocm::is_integral<tt_cat::mfp_void>{});
}

TEST_CASE(is_floating_point)
{
    EXPECT(rocm::is_floating_point<float>{});
    EXPECT(rocm::is_floating_point<double>{});
    EXPECT(rocm::is_floating_point<long double>{});
    EXPECT(rocm::is_floating_point<const float>{});
    EXPECT(rocm::is_floating_point<volatile float>{});
    EXPECT(rocm::is_floating_point<const volatile float>{});
    EXPECT(rocm::is_floating_point<const double>{});
    EXPECT(rocm::is_floating_point<volatile double>{});
    EXPECT(rocm::is_floating_point<const volatile double>{});
    EXPECT(rocm::is_floating_point<const long double>{});
    EXPECT(rocm::is_floating_point<volatile long double>{});
    EXPECT(rocm::is_floating_point<const volatile long double>{});

    EXPECT(not rocm::is_floating_point<int>{});
    EXPECT(not rocm::is_floating_point<bool>{});
    EXPECT(not rocm::is_floating_point<char>{});
    EXPECT(not rocm::is_floating_point<void>{});
    EXPECT(not rocm::is_floating_point<float*>{});
    EXPECT(not rocm::is_floating_point<float&>{});
    EXPECT(not rocm::is_floating_point<float&&>{});
    EXPECT(not rocm::is_floating_point<tt_cat::simple_class>{});
    EXPECT(not rocm::is_floating_point<tt_cat::test_enum>{});
    EXPECT(not rocm::is_floating_point<tt_cat::simple_union>{});
}

TEST_CASE(is_array)
{
    EXPECT(rocm::is_array<int[2]>{});
    EXPECT(rocm::is_array<int[2][3]>{});
    EXPECT(rocm::is_array<const int[2]>{});
    EXPECT(rocm::is_array<volatile int[2]>{});
    EXPECT(rocm::is_array<const volatile int[2]>{});
    EXPECT(rocm::is_array<tt_cat::simple_class[5]>{});
    EXPECT(rocm::is_array<int[]>{});
    EXPECT(rocm::is_array<const int[]>{});
    EXPECT(rocm::is_array<double[1]>{});

    EXPECT(not rocm::is_array<int>{});
    EXPECT(not rocm::is_array<int*>{});
    EXPECT(not rocm::is_array<const int*>{});
    EXPECT(not rocm::is_array<int&>{});
    EXPECT(not rocm::is_array<int&&>{});
    EXPECT(not rocm::is_array<void>{});
    EXPECT(not rocm::is_array<tt_cat::simple_class>{});
    EXPECT(not rocm::is_array<tt_cat::func_ptr>{});
    EXPECT(not rocm::is_array<tt_cat::test_enum>{});
    EXPECT(not rocm::is_array<tt_cat::simple_union>{});
    EXPECT(not rocm::is_array<tt_cat::func_t0>{});
}

TEST_CASE(is_class)
{
    EXPECT(rocm::is_class<tt_cat::simple_class>{});
    EXPECT(rocm::is_class<const tt_cat::simple_class>{});
    EXPECT(rocm::is_class<volatile tt_cat::simple_class>{});
    EXPECT(rocm::is_class<const volatile tt_cat::simple_class>{});
    EXPECT(rocm::is_class<tt_cat::udt>{});

    EXPECT(not rocm::is_class<int>{});
    EXPECT(not rocm::is_class<float>{});
    EXPECT(not rocm::is_class<void>{});
    EXPECT(not rocm::is_class<int*>{});
    EXPECT(not rocm::is_class<int&>{});
    EXPECT(not rocm::is_class<int&&>{});
    EXPECT(not rocm::is_class<int[2]>{});
    EXPECT(not rocm::is_class<tt_cat::test_enum>{});
    EXPECT(not rocm::is_class<tt_cat::scoped_enum>{});
    EXPECT(not rocm::is_class<tt_cat::simple_union>{});
    EXPECT(not rocm::is_class<tt_cat::func_t0>{});
    EXPECT(not rocm::is_class<tt_cat::func_ptr>{});
    EXPECT(not rocm::is_class<tt_cat::mfp_void>{});
    EXPECT(not rocm::is_class<tt_cat::mop>{});
}

TEST_CASE(is_enum)
{
    EXPECT(rocm::is_enum<tt_cat::test_enum>{});
    EXPECT(rocm::is_enum<const tt_cat::test_enum>{});
    EXPECT(rocm::is_enum<volatile tt_cat::test_enum>{});
    EXPECT(rocm::is_enum<const volatile tt_cat::test_enum>{});
    EXPECT(rocm::is_enum<tt_cat::scoped_enum>{});
    EXPECT(rocm::is_enum<const tt_cat::scoped_enum>{});

    EXPECT(not rocm::is_enum<int>{});
    EXPECT(not rocm::is_enum<float>{});
    EXPECT(not rocm::is_enum<void>{});
    EXPECT(not rocm::is_enum<tt_cat::simple_class>{});
    EXPECT(not rocm::is_enum<tt_cat::simple_union>{});
    EXPECT(not rocm::is_enum<int*>{});
    EXPECT(not rocm::is_enum<int&>{});
    EXPECT(not rocm::is_enum<int&&>{});
    EXPECT(not rocm::is_enum<tt_cat::func_t0>{});
    EXPECT(not rocm::is_enum<tt_cat::func_ptr>{});
}

TEST_CASE(is_union)
{
    EXPECT(rocm::is_union<tt_cat::simple_union>{});
    EXPECT(rocm::is_union<const tt_cat::simple_union>{});
    EXPECT(rocm::is_union<volatile tt_cat::simple_union>{});
    EXPECT(rocm::is_union<const volatile tt_cat::simple_union>{});

    EXPECT(not rocm::is_union<int>{});
    EXPECT(not rocm::is_union<float>{});
    EXPECT(not rocm::is_union<void>{});
    EXPECT(not rocm::is_union<tt_cat::simple_class>{});
    EXPECT(not rocm::is_union<tt_cat::test_enum>{});
    EXPECT(not rocm::is_union<int*>{});
    EXPECT(not rocm::is_union<int&>{});
    EXPECT(not rocm::is_union<tt_cat::func_t0>{});
}

TEST_CASE(is_function)
{
    EXPECT(rocm::is_function<tt_cat::func_t0>{});
    EXPECT(rocm::is_function<tt_cat::func_t1>{});
    EXPECT(rocm::is_function<tt_cat::func_t2>{});
    EXPECT(rocm::is_function<tt_cat::func_t3>{});
    EXPECT(rocm::is_function<tt_cat::func_t4>{});

    EXPECT(not rocm::is_function<int>{});
    EXPECT(not rocm::is_function<float>{});
    EXPECT(not rocm::is_function<void>{});
    EXPECT(not rocm::is_function<tt_cat::func_ptr>{});
    EXPECT(not rocm::is_function<tt_cat::func_ptr_i>{});
    EXPECT(not rocm::is_function<int*>{});
    EXPECT(not rocm::is_function<int&>{});
    EXPECT(not rocm::is_function<int&&>{});
    EXPECT(not rocm::is_function<tt_cat::simple_class>{});
    EXPECT(not rocm::is_function<tt_cat::test_enum>{});
    EXPECT(not rocm::is_function<int[2]>{});
}

TEST_CASE(is_lvalue_reference)
{
    EXPECT(rocm::is_lvalue_reference<int&>{});
    EXPECT(rocm::is_lvalue_reference<const int&>{});
    EXPECT(rocm::is_lvalue_reference<volatile int&>{});
    EXPECT(rocm::is_lvalue_reference<const volatile int&>{});
    EXPECT(rocm::is_lvalue_reference<tt_cat::simple_class&>{});
    EXPECT(rocm::is_lvalue_reference<int(&)[2]>{});

    EXPECT(not rocm::is_lvalue_reference<int>{});
    EXPECT(not rocm::is_lvalue_reference<int&&>{});
    EXPECT(not rocm::is_lvalue_reference<int*>{});
    EXPECT(not rocm::is_lvalue_reference<void>{});
    EXPECT(not rocm::is_lvalue_reference<tt_cat::simple_class>{});
    EXPECT(not rocm::is_lvalue_reference<tt_cat::func_ptr>{});
    EXPECT(not rocm::is_lvalue_reference<tt_cat::test_enum>{});
}

TEST_CASE(is_rvalue_reference)
{
    EXPECT(rocm::is_rvalue_reference<int&&>{});
    EXPECT(rocm::is_rvalue_reference<const int&&>{});
    EXPECT(rocm::is_rvalue_reference<volatile int&&>{});
    EXPECT(rocm::is_rvalue_reference<const volatile int&&>{});
    EXPECT(rocm::is_rvalue_reference<tt_cat::simple_class&&>{});
    EXPECT(rocm::is_rvalue_reference<int(&&)[2]>{});

    EXPECT(not rocm::is_rvalue_reference<int>{});
    EXPECT(not rocm::is_rvalue_reference<int&>{});
    EXPECT(not rocm::is_rvalue_reference<int*>{});
    EXPECT(not rocm::is_rvalue_reference<void>{});
    EXPECT(not rocm::is_rvalue_reference<tt_cat::simple_class>{});
    EXPECT(not rocm::is_rvalue_reference<tt_cat::func_ptr>{});
}

TEST_CASE(is_member_object_pointer)
{
    EXPECT(rocm::is_member_object_pointer<tt_cat::mop>{});
    EXPECT(rocm::is_member_object_pointer<const tt_cat::mop>{});
    EXPECT(rocm::is_member_object_pointer<volatile tt_cat::mop>{});

    EXPECT(not rocm::is_member_object_pointer<tt_cat::mfp_void>{});
    EXPECT(not rocm::is_member_object_pointer<tt_cat::mfp_int>{});
    EXPECT(not rocm::is_member_object_pointer<tt_cat::cmfp>{});
    EXPECT(not rocm::is_member_object_pointer<int>{});
    EXPECT(not rocm::is_member_object_pointer<int*>{});
    EXPECT(not rocm::is_member_object_pointer<tt_cat::func_ptr>{});
    EXPECT(not rocm::is_member_object_pointer<void>{});
    EXPECT(not rocm::is_member_object_pointer<tt_cat::func_t0>{});
}

TEST_CASE(is_member_function_pointer)
{
    EXPECT(rocm::is_member_function_pointer<tt_cat::mfp_void>{});
    EXPECT(rocm::is_member_function_pointer<tt_cat::mfp_int>{});
    EXPECT(rocm::is_member_function_pointer<tt_cat::mfp_int_arg>{});
    EXPECT(rocm::is_member_function_pointer<tt_cat::mfp_int_flt>{});
    EXPECT(rocm::is_member_function_pointer<tt_cat::cmfp>{});
    EXPECT(rocm::is_member_function_pointer<const tt_cat::mfp_void>{});

    EXPECT(not rocm::is_member_function_pointer<tt_cat::mop>{});
    EXPECT(not rocm::is_member_function_pointer<int>{});
    EXPECT(not rocm::is_member_function_pointer<int*>{});
    EXPECT(not rocm::is_member_function_pointer<tt_cat::func_ptr>{});
    EXPECT(not rocm::is_member_function_pointer<void>{});
    EXPECT(not rocm::is_member_function_pointer<tt_cat::func_t0>{});
    EXPECT(not rocm::is_member_function_pointer<tt_cat::simple_class>{});
}

// ---------- composite type categories ----------

TEST_CASE(is_reference)
{
    EXPECT(rocm::is_reference<int&>{});
    EXPECT(rocm::is_reference<int&&>{});
    EXPECT(rocm::is_reference<const int&>{});
    EXPECT(rocm::is_reference<volatile int&&>{});
    EXPECT(rocm::is_reference<tt_cat::simple_class&>{});
    EXPECT(rocm::is_reference<tt_cat::simple_class&&>{});
    EXPECT(rocm::is_reference<int(&)[2]>{});
    EXPECT(rocm::is_reference<int(&&)[2]>{});

    EXPECT(not rocm::is_reference<int>{});
    EXPECT(not rocm::is_reference<int*>{});
    EXPECT(not rocm::is_reference<void>{});
    EXPECT(not rocm::is_reference<tt_cat::simple_class>{});
    EXPECT(not rocm::is_reference<tt_cat::func_ptr>{});
    EXPECT(not rocm::is_reference<tt_cat::test_enum>{});
}

TEST_CASE(is_member_pointer)
{
    EXPECT(rocm::is_member_pointer<tt_cat::mop>{});
    EXPECT(rocm::is_member_pointer<tt_cat::mfp_void>{});
    EXPECT(rocm::is_member_pointer<tt_cat::mfp_int>{});
    EXPECT(rocm::is_member_pointer<tt_cat::mfp_int_arg>{});
    EXPECT(rocm::is_member_pointer<tt_cat::cmfp>{});
    EXPECT(rocm::is_member_pointer<const tt_cat::mop>{});

    EXPECT(not rocm::is_member_pointer<int>{});
    EXPECT(not rocm::is_member_pointer<int*>{});
    EXPECT(not rocm::is_member_pointer<void>{});
    EXPECT(not rocm::is_member_pointer<tt_cat::func_ptr>{});
    EXPECT(not rocm::is_member_pointer<tt_cat::simple_class>{});
    EXPECT(not rocm::is_member_pointer<tt_cat::func_t0>{});
}

TEST_CASE(is_fundamental)
{
    EXPECT(rocm::is_fundamental<void>{});
    EXPECT(rocm::is_fundamental<int>{});
    EXPECT(rocm::is_fundamental<float>{});
    EXPECT(rocm::is_fundamental<double>{});
    EXPECT(rocm::is_fundamental<long double>{});
    EXPECT(rocm::is_fundamental<bool>{});
    EXPECT(rocm::is_fundamental<char>{});
    EXPECT(rocm::is_fundamental<wchar_t>{});
    EXPECT(rocm::is_fundamental<long>{});
    EXPECT(rocm::is_fundamental<unsigned long long>{});
    EXPECT(rocm::is_fundamental<const int>{});
    EXPECT(rocm::is_fundamental<volatile float>{});
    EXPECT(rocm::is_fundamental<const volatile double>{});

    EXPECT(not rocm::is_fundamental<int*>{});
    EXPECT(not rocm::is_fundamental<int&>{});
    EXPECT(not rocm::is_fundamental<int&&>{});
    EXPECT(not rocm::is_fundamental<int[2]>{});
    EXPECT(not rocm::is_fundamental<tt_cat::simple_class>{});
    EXPECT(not rocm::is_fundamental<tt_cat::test_enum>{});
    EXPECT(not rocm::is_fundamental<tt_cat::simple_union>{});
    EXPECT(not rocm::is_fundamental<tt_cat::func_t0>{});
    EXPECT(not rocm::is_fundamental<tt_cat::func_ptr>{});
    EXPECT(not rocm::is_fundamental<tt_cat::mop>{});
}

TEST_CASE(is_compound)
{
    EXPECT(rocm::is_compound<int*>{});
    EXPECT(rocm::is_compound<int&>{});
    EXPECT(rocm::is_compound<int&&>{});
    EXPECT(rocm::is_compound<int[2]>{});
    EXPECT(rocm::is_compound<tt_cat::simple_class>{});
    EXPECT(rocm::is_compound<tt_cat::test_enum>{});
    EXPECT(rocm::is_compound<tt_cat::scoped_enum>{});
    EXPECT(rocm::is_compound<tt_cat::simple_union>{});
    EXPECT(rocm::is_compound<tt_cat::func_t0>{});
    EXPECT(rocm::is_compound<tt_cat::func_ptr>{});
    EXPECT(rocm::is_compound<tt_cat::mop>{});
    EXPECT(rocm::is_compound<tt_cat::mfp_void>{});

    EXPECT(not rocm::is_compound<void>{});
    EXPECT(not rocm::is_compound<int>{});
    EXPECT(not rocm::is_compound<float>{});
    EXPECT(not rocm::is_compound<double>{});
    EXPECT(not rocm::is_compound<bool>{});
    EXPECT(not rocm::is_compound<char>{});
    EXPECT(not rocm::is_compound<long>{});
}

TEST_CASE(is_object)
{
    EXPECT(rocm::is_object<int>{});
    EXPECT(rocm::is_object<float>{});
    EXPECT(rocm::is_object<int*>{});
    EXPECT(rocm::is_object<int[2]>{});
    EXPECT(rocm::is_object<tt_cat::simple_class>{});
    EXPECT(rocm::is_object<tt_cat::test_enum>{});
    EXPECT(rocm::is_object<tt_cat::simple_union>{});
    EXPECT(rocm::is_object<const int>{});
    EXPECT(rocm::is_object<tt_cat::mop>{});
    EXPECT(rocm::is_object<tt_cat::mfp_void>{});
    EXPECT(rocm::is_object<tt_cat::func_ptr>{});

    EXPECT(not rocm::is_object<void>{});
    EXPECT(not rocm::is_object<int&>{});
    EXPECT(not rocm::is_object<int&&>{});
    EXPECT(not rocm::is_object<tt_cat::func_t0>{});
    EXPECT(not rocm::is_object<tt_cat::func_t1>{});
    EXPECT(not rocm::is_object<tt_cat::func_t2>{});
}
