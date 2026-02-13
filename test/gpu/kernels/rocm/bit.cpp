#include <rocm/bit.hpp>
#include <migraphx/kernels/test.hpp>

// SFINAE detection: checks whether rocm::bit_cast<To>(From{}) is well-formed
template <class To, class From, class = void>
struct can_bit_cast : rocm::false_type
{
};

template <class To, class From>
struct can_bit_cast<To, From, rocm::void_t<decltype(rocm::bit_cast<To>(From{}))>>
    : rocm::true_type
{
};

// Trivially copyable POD structs (4 bytes each)
struct pod_a
{
    int x;
};

struct pod_b
{
    unsigned int y;
};

// Non-trivially copyable type (user-defined copy constructor)
struct non_trivial
{
    int x;
    constexpr non_trivial() : x(0) {}
    constexpr non_trivial(const non_trivial& other) : x(other.x) {}
};

// 8-byte non-trivially copyable type
struct non_trivial_8
{
    double d;
    constexpr non_trivial_8() : d(0.0) {}
    constexpr non_trivial_8(const non_trivial_8& other) : d(other.d) {}
};

// Trivially copyable but NOT trivially constructible
// (user-defined default ctor; copy/move/dtor are implicitly trivial)
struct trivial_copy_nontrivial_ctor
{
    int x;
    constexpr trivial_copy_nontrivial_ctor() : x(42) {}
};

// Enum type (trivially copyable, same size as underlying type)
enum class color : int
{
    red   = 0,
    green = 1,
    blue  = 2
};

// ---- Verify test type sizes ----
static_assert(sizeof(pod_a) == sizeof(int));
static_assert(sizeof(pod_b) == sizeof(unsigned int));
static_assert(sizeof(non_trivial) == sizeof(int));
static_assert(sizeof(non_trivial_8) == sizeof(double));
static_assert(sizeof(trivial_copy_nontrivial_ctor) == sizeof(int));
static_assert(sizeof(color) == sizeof(int));
static_assert(sizeof(double) == sizeof(unsigned long long));

// Verify trivial_copy_nontrivial_ctor properties
static_assert(rocm::is_trivially_copyable<trivial_copy_nontrivial_ctor>{});
static_assert(not rocm::is_trivially_constructible<trivial_copy_nontrivial_ctor>{});

// ---- SFINAE: valid same-size trivially-copyable pairs ----
// Same type
static_assert(can_bit_cast<int, int>{});
static_assert(can_bit_cast<float, float>{});
static_assert(can_bit_cast<double, double>{});
static_assert(can_bit_cast<unsigned int, unsigned int>{});
static_assert(can_bit_cast<char, char>{});

// Same-size different types
static_assert(can_bit_cast<int, float>{});
static_assert(can_bit_cast<float, int>{});
static_assert(can_bit_cast<int, unsigned int>{});
static_assert(can_bit_cast<unsigned int, int>{});
static_assert(can_bit_cast<char, unsigned char>{});
static_assert(can_bit_cast<unsigned char, char>{});
static_assert(can_bit_cast<short, unsigned short>{});
static_assert(can_bit_cast<unsigned short, short>{});

// POD struct pairs
static_assert(can_bit_cast<pod_a, pod_b>{});
static_assert(can_bit_cast<pod_b, pod_a>{});
static_assert(can_bit_cast<int, pod_a>{});
static_assert(can_bit_cast<pod_a, int>{});
static_assert(can_bit_cast<float, pod_a>{});
static_assert(can_bit_cast<pod_a, float>{});

// Enum types
static_assert(can_bit_cast<int, color>{});
static_assert(can_bit_cast<color, int>{});
static_assert(can_bit_cast<unsigned int, color>{});
static_assert(can_bit_cast<color, unsigned int>{});
static_assert(can_bit_cast<float, color>{});
static_assert(can_bit_cast<color, float>{});

// Trivially copyable but not trivially constructible (accepted)
static_assert(can_bit_cast<trivial_copy_nontrivial_ctor, int>{});
static_assert(can_bit_cast<int, trivial_copy_nontrivial_ctor>{});
static_assert(can_bit_cast<trivial_copy_nontrivial_ctor, float>{});
static_assert(can_bit_cast<float, trivial_copy_nontrivial_ctor>{});
static_assert(can_bit_cast<trivial_copy_nontrivial_ctor, unsigned int>{});
static_assert(can_bit_cast<trivial_copy_nontrivial_ctor, pod_a>{});
static_assert(can_bit_cast<pod_a, trivial_copy_nontrivial_ctor>{});
static_assert(can_bit_cast<trivial_copy_nontrivial_ctor, trivial_copy_nontrivial_ctor>{});

// ---- SFINAE: sizeof mismatch (rejected) ----
static_assert(not can_bit_cast<int, char>{});
static_assert(not can_bit_cast<char, int>{});
static_assert(not can_bit_cast<int, short>{});
static_assert(not can_bit_cast<short, int>{});
static_assert(not can_bit_cast<int, double>{});
static_assert(not can_bit_cast<double, int>{});
static_assert(not can_bit_cast<float, double>{});
static_assert(not can_bit_cast<double, float>{});
static_assert(not can_bit_cast<char, short>{});
static_assert(not can_bit_cast<short, char>{});
static_assert(not can_bit_cast<float, char>{});
static_assert(not can_bit_cast<char, float>{});
static_assert(not can_bit_cast<pod_a, double>{});
static_assert(not can_bit_cast<double, pod_a>{});

// ---- SFINAE: non-trivially-copyable (rejected) ----
// Same size but not trivially copyable
static_assert(not can_bit_cast<non_trivial, int>{});
static_assert(not can_bit_cast<int, non_trivial>{});
static_assert(not can_bit_cast<non_trivial, non_trivial>{});
static_assert(not can_bit_cast<non_trivial, float>{});
static_assert(not can_bit_cast<float, non_trivial>{});
static_assert(not can_bit_cast<non_trivial, pod_a>{});
static_assert(not can_bit_cast<pod_a, non_trivial>{});
// Different size AND not trivially copyable
static_assert(not can_bit_cast<non_trivial_8, int>{});
static_assert(not can_bit_cast<int, non_trivial_8>{});
static_assert(not can_bit_cast<non_trivial_8, float>{});
static_assert(not can_bit_cast<float, non_trivial_8>{});
// Same size non-trivially-copyable pair
static_assert(not can_bit_cast<non_trivial_8, double>{});
static_assert(not can_bit_cast<double, non_trivial_8>{});

// ---- Constexpr value correctness ----
// Identity
static_assert(rocm::bit_cast<int>(0) == 0);
static_assert(rocm::bit_cast<int>(42) == 42);
static_assert(rocm::bit_cast<int>(-1) == -1);

// int <-> unsigned int
static_assert(rocm::bit_cast<unsigned int>(0) == 0u);
static_assert(rocm::bit_cast<int>(0u) == 0);
static_assert(rocm::bit_cast<unsigned int>(-1) == ~0u);

// float <-> unsigned int (IEEE 754 single-precision)
static_assert(rocm::bit_cast<unsigned int>(0.0f) == 0x00000000u);
static_assert(rocm::bit_cast<unsigned int>(1.0f) == 0x3F800000u);
static_assert(rocm::bit_cast<unsigned int>(-1.0f) == 0xBF800000u);
static_assert(rocm::bit_cast<unsigned int>(0.5f) == 0x3F000000u);
static_assert(rocm::bit_cast<unsigned int>(2.0f) == 0x40000000u);
static_assert(rocm::bit_cast<float>(0x3F800000u) == 1.0f);
static_assert(rocm::bit_cast<float>(0x00000000u) == 0.0f);

// Constexpr roundtrip
static_assert(rocm::bit_cast<float>(rocm::bit_cast<unsigned int>(1.0f)) == 1.0f);
static_assert(rocm::bit_cast<int>(rocm::bit_cast<unsigned int>(42)) == 42);
static_assert(rocm::bit_cast<unsigned int>(rocm::bit_cast<int>(123u)) == 123u);

TEST_CASE(bit_cast_identity)
{
    EXPECT(rocm::bit_cast<int>(0) == 0);
    EXPECT(rocm::bit_cast<int>(42) == 42);
    EXPECT(rocm::bit_cast<int>(-1) == -1);
    EXPECT(rocm::bit_cast<unsigned int>(0u) == 0u);
    EXPECT(rocm::bit_cast<unsigned int>(123u) == 123u);
    EXPECT(rocm::bit_cast<char>('A') == 'A');

    // Float/double identity verified via integer bit patterns
    EXPECT(rocm::bit_cast<unsigned int>(rocm::bit_cast<float>(1.0f)) == 0x3F800000u);
    EXPECT(rocm::bit_cast<unsigned int>(rocm::bit_cast<float>(0.0f)) == 0x00000000u);
    EXPECT(rocm::bit_cast<unsigned long long>(rocm::bit_cast<double>(1.0)) == 0x3FF0000000000000ull);
}

TEST_CASE(bit_cast_int_unsigned)
{
    EXPECT(rocm::bit_cast<unsigned int>(0) == 0u);
    EXPECT(rocm::bit_cast<unsigned int>(1) == 1u);
    EXPECT(rocm::bit_cast<unsigned int>(-1) == ~0u);
    EXPECT(rocm::bit_cast<int>(0u) == 0);
    EXPECT(rocm::bit_cast<int>(1u) == 1);

    EXPECT(rocm::bit_cast<unsigned short>(static_cast<short>(0)) == 0u);
    EXPECT(rocm::bit_cast<unsigned short>(static_cast<short>(1)) == 1u);
    EXPECT(rocm::bit_cast<short>(static_cast<unsigned short>(0)) == 0);
    EXPECT(rocm::bit_cast<short>(static_cast<unsigned short>(1)) == 1);

    EXPECT(rocm::bit_cast<unsigned char>(static_cast<char>(0)) == 0);
    EXPECT(rocm::bit_cast<char>(static_cast<unsigned char>(65)) == 'A');
}

TEST_CASE(bit_cast_float_to_int)
{
    // IEEE 754 single-precision bit patterns
    EXPECT(rocm::bit_cast<unsigned int>(0.0f) == 0x00000000u);
    EXPECT(rocm::bit_cast<unsigned int>(1.0f) == 0x3F800000u);
    EXPECT(rocm::bit_cast<unsigned int>(-1.0f) == 0xBF800000u);
    EXPECT(rocm::bit_cast<unsigned int>(0.5f) == 0x3F000000u);
    EXPECT(rocm::bit_cast<unsigned int>(2.0f) == 0x40000000u);
    EXPECT(rocm::bit_cast<unsigned int>(-0.0f) == 0x80000000u);
}

TEST_CASE(bit_cast_int_to_float)
{
    // Verify uint -> float -> uint preserves all bits
    EXPECT(rocm::bit_cast<unsigned int>(rocm::bit_cast<float>(0x3F800000u)) == 0x3F800000u);
    EXPECT(rocm::bit_cast<unsigned int>(rocm::bit_cast<float>(0x00000000u)) == 0x00000000u);
    EXPECT(rocm::bit_cast<unsigned int>(rocm::bit_cast<float>(0x40000000u)) == 0x40000000u);
    EXPECT(rocm::bit_cast<unsigned int>(rocm::bit_cast<float>(0x3F000000u)) == 0x3F000000u);
    EXPECT(rocm::bit_cast<unsigned int>(rocm::bit_cast<float>(0xBF800000u)) == 0xBF800000u);
    EXPECT(rocm::bit_cast<unsigned int>(rocm::bit_cast<float>(0x80000000u)) == 0x80000000u);
}

TEST_CASE(bit_cast_roundtrip)
{
    // float -> unsigned int -> float (verified via bit pattern)
    EXPECT(rocm::bit_cast<unsigned int>(
               rocm::bit_cast<float>(rocm::bit_cast<unsigned int>(1.0f))) == 0x3F800000u);
    EXPECT(rocm::bit_cast<unsigned int>(
               rocm::bit_cast<float>(rocm::bit_cast<unsigned int>(0.0f))) == 0x00000000u);
    EXPECT(rocm::bit_cast<unsigned int>(
               rocm::bit_cast<float>(rocm::bit_cast<unsigned int>(-1.0f))) == 0xBF800000u);
    EXPECT(rocm::bit_cast<unsigned int>(
               rocm::bit_cast<float>(rocm::bit_cast<unsigned int>(3.14f))) ==
           rocm::bit_cast<unsigned int>(3.14f));

    // int -> unsigned int -> int
    EXPECT(rocm::bit_cast<int>(rocm::bit_cast<unsigned int>(0)) == 0);
    EXPECT(rocm::bit_cast<int>(rocm::bit_cast<unsigned int>(42)) == 42);
    EXPECT(rocm::bit_cast<int>(rocm::bit_cast<unsigned int>(-42)) == -42);

    // unsigned int -> int -> unsigned int
    EXPECT(rocm::bit_cast<unsigned int>(rocm::bit_cast<int>(0u)) == 0u);
    EXPECT(rocm::bit_cast<unsigned int>(rocm::bit_cast<int>(100u)) == 100u);
}

TEST_CASE(bit_cast_pod_struct)
{
    pod_a a{42};
    pod_b b = rocm::bit_cast<pod_b>(a);
    EXPECT(b.y == 42u);

    pod_b b2{100u};
    pod_a a2 = rocm::bit_cast<pod_a>(b2);
    EXPECT(a2.x == 100);

    // Roundtrip through pod structs
    pod_a a3{-1};
    pod_a a4 = rocm::bit_cast<pod_a>(rocm::bit_cast<pod_b>(a3));
    EXPECT(a4.x == -1);
}

TEST_CASE(bit_cast_enum)
{
    EXPECT(rocm::bit_cast<int>(color::red) == 0);
    EXPECT(rocm::bit_cast<int>(color::green) == 1);
    EXPECT(rocm::bit_cast<int>(color::blue) == 2);

    EXPECT(rocm::bit_cast<color>(0) == color::red);
    EXPECT(rocm::bit_cast<color>(1) == color::green);
    EXPECT(rocm::bit_cast<color>(2) == color::blue);
}

TEST_CASE(bit_cast_trivial_copy_nontrivial_ctor)
{
    // Trivially copyable but not trivially constructible: bit_cast should still work
    trivial_copy_nontrivial_ctor tc;
    EXPECT(tc.x == 42);

    // To int and back
    int i = rocm::bit_cast<int>(tc);
    EXPECT(i == 42);

    trivial_copy_nontrivial_ctor tc2 = rocm::bit_cast<trivial_copy_nontrivial_ctor>(99);
    EXPECT(tc2.x == 99);

    // Roundtrip
    trivial_copy_nontrivial_ctor tc3 =
        rocm::bit_cast<trivial_copy_nontrivial_ctor>(rocm::bit_cast<int>(tc));
    EXPECT(tc3.x == 42);

    // To/from float (same size, both trivially copyable)
    unsigned int bits = rocm::bit_cast<unsigned int>(rocm::bit_cast<float>(tc));
    EXPECT(bits == rocm::bit_cast<unsigned int>(tc));

    // To/from pod struct
    pod_a pa = rocm::bit_cast<pod_a>(tc);
    EXPECT(pa.x == 42);

    trivial_copy_nontrivial_ctor tc4 = rocm::bit_cast<trivial_copy_nontrivial_ctor>(pod_b{7u});
    EXPECT(tc4.x == 7);
}
