#include <rocm/bit.hpp>
#include <migraphx/kernels/test.hpp>

// SFINAE detection: checks whether rocm::bit_cast<To>(From{}) is well-formed
template <class To, class From, class = void>
struct can_bit_cast : rocm::false_type
{
};

template <class To, class From>
struct can_bit_cast<To, From, rocm::void_t<decltype(rocm::bit_cast<To>(From{}))>> : rocm::true_type
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
    EXPECT(rocm::bit_cast<unsigned long long>(rocm::bit_cast<double>(1.0)) ==
           0x3FF0000000000000ull);
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
    EXPECT(rocm::bit_cast<unsigned int>(rocm::bit_cast<float>(
               rocm::bit_cast<unsigned int>(3.14f))) == rocm::bit_cast<unsigned int>(3.14f));

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

// ============================================================================
// SFINAE detection for unsigned-only bit functions
// ============================================================================

template <class T, class = void>
struct can_popcount : rocm::false_type
{
};

template <class T>
struct can_popcount<T, rocm::void_t<decltype(rocm::popcount(T{}))>> : rocm::true_type
{
};

template <class T, class = void>
struct can_countl_zero : rocm::false_type
{
};

template <class T>
struct can_countl_zero<T, rocm::void_t<decltype(rocm::countl_zero(T{}))>> : rocm::true_type
{
};

template <class T, class = void>
struct can_rotl : rocm::false_type
{
};

template <class T>
struct can_rotl<T, rocm::void_t<decltype(rocm::rotl(T{}, 0))>> : rocm::true_type
{
};

// Unsigned types accepted
static_assert(can_popcount<unsigned char>{});
static_assert(can_popcount<unsigned short>{});
static_assert(can_popcount<unsigned int>{});
static_assert(can_popcount<unsigned long long>{});
static_assert(can_countl_zero<unsigned char>{});
static_assert(can_countl_zero<unsigned short>{});
static_assert(can_countl_zero<unsigned int>{});
static_assert(can_countl_zero<unsigned long long>{});
static_assert(can_rotl<unsigned char>{});
static_assert(can_rotl<unsigned short>{});
static_assert(can_rotl<unsigned int>{});
static_assert(can_rotl<unsigned long long>{});

// Signed and non-integer types rejected
static_assert(not can_popcount<signed char>{});
static_assert(not can_popcount<short>{});
static_assert(not can_popcount<int>{});
static_assert(not can_popcount<long long>{});
static_assert(not can_popcount<float>{});
static_assert(not can_popcount<double>{});
static_assert(not can_countl_zero<int>{});
static_assert(not can_countl_zero<float>{});
static_assert(not can_rotl<int>{});
static_assert(not can_rotl<float>{});

// ============================================================================
// Constexpr correctness: countl_zero
// ============================================================================
// unsigned int (32-bit)
static_assert(rocm::countl_zero(0u) == 32);
static_assert(rocm::countl_zero(1u) == 31);
static_assert(rocm::countl_zero(2u) == 30);
static_assert(rocm::countl_zero(0x80000000u) == 0);
static_assert(rocm::countl_zero(0xFFFFFFFFu) == 0);
static_assert(rocm::countl_zero(0x0000FFFFu) == 16);
static_assert(rocm::countl_zero(0x00010000u) == 15);
// unsigned char (8-bit)
static_assert(rocm::countl_zero(static_cast<unsigned char>(0)) == 8);
static_assert(rocm::countl_zero(static_cast<unsigned char>(1)) == 7);
static_assert(rocm::countl_zero(static_cast<unsigned char>(0x80)) == 0);
static_assert(rocm::countl_zero(static_cast<unsigned char>(0xFF)) == 0);
// unsigned long long (64-bit)
static_assert(rocm::countl_zero(0ull) == 64);
static_assert(rocm::countl_zero(1ull) == 63);
static_assert(rocm::countl_zero(0x8000000000000000ull) == 0);

// ============================================================================
// Constexpr correctness: countl_one
// ============================================================================
static_assert(rocm::countl_one(0u) == 0);
static_assert(rocm::countl_one(0xFFFFFFFFu) == 32);
static_assert(rocm::countl_one(0x80000000u) == 1);
static_assert(rocm::countl_one(0xF0000000u) == 4);
static_assert(rocm::countl_one(0xFFF00000u) == 12);
// unsigned char (8-bit)
static_assert(rocm::countl_one(static_cast<unsigned char>(0xFF)) == 8);
static_assert(rocm::countl_one(static_cast<unsigned char>(0xF0)) == 4);
static_assert(rocm::countl_one(static_cast<unsigned char>(0x00)) == 0);

// ============================================================================
// Constexpr correctness: countr_zero
// ============================================================================
static_assert(rocm::countr_zero(0u) == 32);
static_assert(rocm::countr_zero(1u) == 0);
static_assert(rocm::countr_zero(2u) == 1);
static_assert(rocm::countr_zero(0x80000000u) == 31);
static_assert(rocm::countr_zero(0x00010000u) == 16);
static_assert(rocm::countr_zero(0xFFFFFFFFu) == 0);
static_assert(rocm::countr_zero(static_cast<unsigned char>(0)) == 8);
static_assert(rocm::countr_zero(static_cast<unsigned char>(0x80)) == 7);

// ============================================================================
// Constexpr correctness: countr_one
// ============================================================================
static_assert(rocm::countr_one(0u) == 0);
static_assert(rocm::countr_one(1u) == 1);
static_assert(rocm::countr_one(0xFFFFFFFFu) == 32);
static_assert(rocm::countr_one(0x0000000Fu) == 4);
static_assert(rocm::countr_one(0x000000FFu) == 8);
// unsigned char (8-bit)
static_assert(rocm::countr_one(static_cast<unsigned char>(0xFF)) == 8);
static_assert(rocm::countr_one(static_cast<unsigned char>(0x0F)) == 4);
static_assert(rocm::countr_one(static_cast<unsigned char>(0x00)) == 0);

// ============================================================================
// Constexpr correctness: popcount
// ============================================================================
static_assert(rocm::popcount(0u) == 0);
static_assert(rocm::popcount(1u) == 1);
static_assert(rocm::popcount(0xFFFFFFFFu) == 32);
static_assert(rocm::popcount(0x55555555u) == 16);
static_assert(rocm::popcount(0xAAAAAAAAu) == 16);
static_assert(rocm::popcount(0x000000FFu) == 8);
static_assert(rocm::popcount(0x0F0F0F0Fu) == 16);
static_assert(rocm::popcount(static_cast<unsigned char>(0)) == 0);
static_assert(rocm::popcount(static_cast<unsigned char>(0xFF)) == 8);
static_assert(rocm::popcount(static_cast<unsigned char>(0xAA)) == 4);
static_assert(rocm::popcount(0xFFFFFFFFFFFFFFFFull) == 64);

// ============================================================================
// Constexpr correctness: bit_width
// ============================================================================
static_assert(rocm::bit_width(0u) == 0);
static_assert(rocm::bit_width(1u) == 1);
static_assert(rocm::bit_width(2u) == 2);
static_assert(rocm::bit_width(3u) == 2);
static_assert(rocm::bit_width(4u) == 3);
static_assert(rocm::bit_width(7u) == 3);
static_assert(rocm::bit_width(8u) == 4);
static_assert(rocm::bit_width(255u) == 8);
static_assert(rocm::bit_width(256u) == 9);
static_assert(rocm::bit_width(0xFFFFFFFFu) == 32);
static_assert(rocm::bit_width(static_cast<unsigned char>(0)) == 0);
static_assert(rocm::bit_width(static_cast<unsigned char>(0xFF)) == 8);

// ============================================================================
// Constexpr correctness: bit_floor
// ============================================================================
static_assert(rocm::bit_floor(0u) == 0u);
static_assert(rocm::bit_floor(1u) == 1u);
static_assert(rocm::bit_floor(2u) == 2u);
static_assert(rocm::bit_floor(3u) == 2u);
static_assert(rocm::bit_floor(4u) == 4u);
static_assert(rocm::bit_floor(5u) == 4u);
static_assert(rocm::bit_floor(7u) == 4u);
static_assert(rocm::bit_floor(8u) == 8u);
static_assert(rocm::bit_floor(9u) == 8u);
static_assert(rocm::bit_floor(255u) == 128u);
static_assert(rocm::bit_floor(0xFFFFFFFFu) == 0x80000000u);

// ============================================================================
// Constexpr correctness: bit_ceil
// ============================================================================
static_assert(rocm::bit_ceil(0u) == 1u);
static_assert(rocm::bit_ceil(1u) == 1u);
static_assert(rocm::bit_ceil(2u) == 2u);
static_assert(rocm::bit_ceil(3u) == 4u);
static_assert(rocm::bit_ceil(4u) == 4u);
static_assert(rocm::bit_ceil(5u) == 8u);
static_assert(rocm::bit_ceil(7u) == 8u);
static_assert(rocm::bit_ceil(8u) == 8u);
static_assert(rocm::bit_ceil(9u) == 16u);
static_assert(rocm::bit_ceil(128u) == 128u);
static_assert(rocm::bit_ceil(129u) == 256u);

// ============================================================================
// Constexpr correctness: has_single_bit
// ============================================================================
static_assert(not rocm::has_single_bit(0u));
static_assert(rocm::has_single_bit(1u));
static_assert(rocm::has_single_bit(2u));
static_assert(not rocm::has_single_bit(3u));
static_assert(rocm::has_single_bit(4u));
static_assert(not rocm::has_single_bit(5u));
static_assert(not rocm::has_single_bit(6u));
static_assert(not rocm::has_single_bit(7u));
static_assert(rocm::has_single_bit(8u));
static_assert(rocm::has_single_bit(0x80000000u));
static_assert(not rocm::has_single_bit(0xFFFFFFFFu));

// ============================================================================
// Constexpr correctness: rotl
// ============================================================================
static_assert(rocm::rotl(1u, 0) == 1u);
static_assert(rocm::rotl(1u, 1) == 2u);
static_assert(rocm::rotl(1u, 4) == 16u);
static_assert(rocm::rotl(1u, 31) == 0x80000000u);
static_assert(rocm::rotl(0x80000000u, 1) == 1u);
static_assert(rocm::rotl(0x12345678u, 4) == 0x23456781u);
static_assert(rocm::rotl(0x12345678u, 8) == 0x34567812u);
static_assert(rocm::rotl(1u, 32) == 1u);
static_assert(rocm::rotl(1u, -1) == 0x80000000u);

// ============================================================================
// Constexpr correctness: rotr
// ============================================================================
static_assert(rocm::rotr(1u, 0) == 1u);
static_assert(rocm::rotr(1u, 1) == 0x80000000u);
static_assert(rocm::rotr(0x80000000u, 1) == 0x40000000u);
static_assert(rocm::rotr(0x12345678u, 4) == 0x81234567u);
static_assert(rocm::rotr(0x12345678u, 8) == 0x78123456u);
static_assert(rocm::rotr(1u, 32) == 1u);
static_assert(rocm::rotr(1u, -1) == 2u);

// rotl/rotr are inverses
static_assert(rocm::rotl(rocm::rotr(0xDEADBEEFu, 7), 7) == 0xDEADBEEFu);
static_assert(rocm::rotr(rocm::rotl(0xDEADBEEFu, 13), 13) == 0xDEADBEEFu);

// ============================================================================
// Runtime tests
// ============================================================================

TEST_CASE(countl_zero_test)
{
    // unsigned int
    EXPECT(rocm::countl_zero(0u) == 32);
    EXPECT(rocm::countl_zero(1u) == 31);
    EXPECT(rocm::countl_zero(0x80000000u) == 0);
    EXPECT(rocm::countl_zero(0xFFFFFFFFu) == 0);
    EXPECT(rocm::countl_zero(0x0000FFFFu) == 16);

    // unsigned char
    EXPECT(rocm::countl_zero(static_cast<unsigned char>(0)) == 8);
    EXPECT(rocm::countl_zero(static_cast<unsigned char>(1)) == 7);
    EXPECT(rocm::countl_zero(static_cast<unsigned char>(0x80)) == 0);
    EXPECT(rocm::countl_zero(static_cast<unsigned char>(0x0F)) == 4);

    // unsigned long long
    EXPECT(rocm::countl_zero(0ull) == 64);
    EXPECT(rocm::countl_zero(1ull) == 63);
    EXPECT(rocm::countl_zero(0x8000000000000000ull) == 0);
    EXPECT(rocm::countl_zero(0x00000000FFFFFFFFull) == 32);
}

TEST_CASE(countl_one_test)
{
    EXPECT(rocm::countl_one(0u) == 0);
    EXPECT(rocm::countl_one(0xFFFFFFFFu) == 32);
    EXPECT(rocm::countl_one(0x80000000u) == 1);
    EXPECT(rocm::countl_one(0xF0000000u) == 4);
    EXPECT(rocm::countl_one(0xFFF00000u) == 12);
    EXPECT(rocm::countl_one(0xFFFFFFFEu) == 31);

    // unsigned char
    EXPECT(rocm::countl_one(static_cast<unsigned char>(0xFF)) == 8);
    EXPECT(rocm::countl_one(static_cast<unsigned char>(0xF0)) == 4);
    EXPECT(rocm::countl_one(static_cast<unsigned char>(0x00)) == 0);
    EXPECT(rocm::countl_one(static_cast<unsigned char>(0xFE)) == 7);

    // unsigned long long
    EXPECT(rocm::countl_one(0xFFFFFFFFFFFFFFFFull) == 64);
    EXPECT(rocm::countl_one(0xFFFFFFFF00000000ull) == 32);
}

TEST_CASE(countr_zero_test)
{
    EXPECT(rocm::countr_zero(0u) == 32);
    EXPECT(rocm::countr_zero(1u) == 0);
    EXPECT(rocm::countr_zero(2u) == 1);
    EXPECT(rocm::countr_zero(4u) == 2);
    EXPECT(rocm::countr_zero(0x80000000u) == 31);
    EXPECT(rocm::countr_zero(0x00010000u) == 16);
    EXPECT(rocm::countr_zero(0xFFFFFFFFu) == 0);
    EXPECT(rocm::countr_zero(0xFFFF0000u) == 16);

    // unsigned char
    EXPECT(rocm::countr_zero(static_cast<unsigned char>(0)) == 8);
    EXPECT(rocm::countr_zero(static_cast<unsigned char>(1)) == 0);
    EXPECT(rocm::countr_zero(static_cast<unsigned char>(0x80)) == 7);
    EXPECT(rocm::countr_zero(static_cast<unsigned char>(0x10)) == 4);

    // unsigned long long
    EXPECT(rocm::countr_zero(0ull) == 64);
    EXPECT(rocm::countr_zero(0x0000000100000000ull) == 32);
}

TEST_CASE(countr_one_test)
{
    EXPECT(rocm::countr_one(0u) == 0);
    EXPECT(rocm::countr_one(1u) == 1);
    EXPECT(rocm::countr_one(3u) == 2);
    EXPECT(rocm::countr_one(0x0000000Fu) == 4);
    EXPECT(rocm::countr_one(0x000000FFu) == 8);
    EXPECT(rocm::countr_one(0xFFFFFFFFu) == 32);
    EXPECT(rocm::countr_one(0x0000FFFEu) == 0);

    // unsigned char
    EXPECT(rocm::countr_one(static_cast<unsigned char>(0xFF)) == 8);
    EXPECT(rocm::countr_one(static_cast<unsigned char>(0x0F)) == 4);
    EXPECT(rocm::countr_one(static_cast<unsigned char>(0x00)) == 0);
    EXPECT(rocm::countr_one(static_cast<unsigned char>(0x7F)) == 7);

    // unsigned long long
    EXPECT(rocm::countr_one(0xFFFFFFFFFFFFFFFFull) == 64);
    EXPECT(rocm::countr_one(0x00000000FFFFFFFFull) == 32);
}

TEST_CASE(popcount_test)
{
    EXPECT(rocm::popcount(0u) == 0);
    EXPECT(rocm::popcount(1u) == 1);
    EXPECT(rocm::popcount(0xFFFFFFFFu) == 32);
    EXPECT(rocm::popcount(0x55555555u) == 16);
    EXPECT(rocm::popcount(0xAAAAAAAAu) == 16);
    EXPECT(rocm::popcount(0x000000FFu) == 8);
    EXPECT(rocm::popcount(0x0F0F0F0Fu) == 16);
    EXPECT(rocm::popcount(0x80000001u) == 2);

    // unsigned char
    EXPECT(rocm::popcount(static_cast<unsigned char>(0)) == 0);
    EXPECT(rocm::popcount(static_cast<unsigned char>(0xFF)) == 8);
    EXPECT(rocm::popcount(static_cast<unsigned char>(0xAA)) == 4);
    EXPECT(rocm::popcount(static_cast<unsigned char>(0x55)) == 4);

    // unsigned long long
    EXPECT(rocm::popcount(0ull) == 0);
    EXPECT(rocm::popcount(0xFFFFFFFFFFFFFFFFull) == 64);
    EXPECT(rocm::popcount(0x5555555555555555ull) == 32);
}

TEST_CASE(bit_width_test)
{
    EXPECT(rocm::bit_width(0u) == 0);
    EXPECT(rocm::bit_width(1u) == 1);
    EXPECT(rocm::bit_width(2u) == 2);
    EXPECT(rocm::bit_width(3u) == 2);
    EXPECT(rocm::bit_width(4u) == 3);
    EXPECT(rocm::bit_width(7u) == 3);
    EXPECT(rocm::bit_width(8u) == 4);
    EXPECT(rocm::bit_width(15u) == 4);
    EXPECT(rocm::bit_width(16u) == 5);
    EXPECT(rocm::bit_width(255u) == 8);
    EXPECT(rocm::bit_width(256u) == 9);
    EXPECT(rocm::bit_width(0xFFFFFFFFu) == 32);

    // unsigned char
    EXPECT(rocm::bit_width(static_cast<unsigned char>(0)) == 0);
    EXPECT(rocm::bit_width(static_cast<unsigned char>(1)) == 1);
    EXPECT(rocm::bit_width(static_cast<unsigned char>(0xFF)) == 8);
    EXPECT(rocm::bit_width(static_cast<unsigned char>(0x80)) == 8);

    // unsigned long long
    EXPECT(rocm::bit_width(0ull) == 0);
    EXPECT(rocm::bit_width(1ull) == 1);
    EXPECT(rocm::bit_width(0xFFFFFFFFFFFFFFFFull) == 64);
}

TEST_CASE(bit_floor_test)
{
    EXPECT(rocm::bit_floor(0u) == 0u);
    EXPECT(rocm::bit_floor(1u) == 1u);
    EXPECT(rocm::bit_floor(2u) == 2u);
    EXPECT(rocm::bit_floor(3u) == 2u);
    EXPECT(rocm::bit_floor(4u) == 4u);
    EXPECT(rocm::bit_floor(5u) == 4u);
    EXPECT(rocm::bit_floor(7u) == 4u);
    EXPECT(rocm::bit_floor(8u) == 8u);
    EXPECT(rocm::bit_floor(9u) == 8u);
    EXPECT(rocm::bit_floor(255u) == 128u);
    EXPECT(rocm::bit_floor(256u) == 256u);
    EXPECT(rocm::bit_floor(0xFFFFFFFFu) == 0x80000000u);

    // unsigned char
    EXPECT(rocm::bit_floor(static_cast<unsigned char>(0)) == 0);
    EXPECT(rocm::bit_floor(static_cast<unsigned char>(1)) == 1);
    EXPECT(rocm::bit_floor(static_cast<unsigned char>(3)) == 2);
    EXPECT(rocm::bit_floor(static_cast<unsigned char>(0xFF)) == 0x80);

    // unsigned long long
    EXPECT(rocm::bit_floor(0ull) == 0ull);
    EXPECT(rocm::bit_floor(1ull) == 1ull);
    EXPECT(rocm::bit_floor(0xFFFFFFFFFFFFFFFFull) == 0x8000000000000000ull);
}

TEST_CASE(bit_ceil_test)
{
    EXPECT(rocm::bit_ceil(0u) == 1u);
    EXPECT(rocm::bit_ceil(1u) == 1u);
    EXPECT(rocm::bit_ceil(2u) == 2u);
    EXPECT(rocm::bit_ceil(3u) == 4u);
    EXPECT(rocm::bit_ceil(4u) == 4u);
    EXPECT(rocm::bit_ceil(5u) == 8u);
    EXPECT(rocm::bit_ceil(7u) == 8u);
    EXPECT(rocm::bit_ceil(8u) == 8u);
    EXPECT(rocm::bit_ceil(9u) == 16u);
    EXPECT(rocm::bit_ceil(128u) == 128u);
    EXPECT(rocm::bit_ceil(129u) == 256u);
    EXPECT(rocm::bit_ceil(0x40000000u) == 0x40000000u);

    // unsigned char (max safe value is 128, since 129 would need 256 which overflows)
    EXPECT(rocm::bit_ceil(static_cast<unsigned char>(0)) == 1);
    EXPECT(rocm::bit_ceil(static_cast<unsigned char>(1)) == 1);
    EXPECT(rocm::bit_ceil(static_cast<unsigned char>(3)) == 4);
    EXPECT(rocm::bit_ceil(static_cast<unsigned char>(64)) == 64);
    EXPECT(rocm::bit_ceil(static_cast<unsigned char>(128)) == 128);

    // unsigned long long
    EXPECT(rocm::bit_ceil(0ull) == 1ull);
    EXPECT(rocm::bit_ceil(1ull) == 1ull);
    EXPECT(rocm::bit_ceil(3ull) == 4ull);
    EXPECT(rocm::bit_ceil(0x100ull) == 0x100ull);
    EXPECT(rocm::bit_ceil(0x101ull) == 0x200ull);
}

TEST_CASE(has_single_bit_test)
{
    EXPECT(not rocm::has_single_bit(0u));
    EXPECT(rocm::has_single_bit(1u));
    EXPECT(rocm::has_single_bit(2u));
    EXPECT(not rocm::has_single_bit(3u));
    EXPECT(rocm::has_single_bit(4u));
    EXPECT(not rocm::has_single_bit(5u));
    EXPECT(not rocm::has_single_bit(6u));
    EXPECT(not rocm::has_single_bit(7u));
    EXPECT(rocm::has_single_bit(8u));
    EXPECT(rocm::has_single_bit(16u));
    EXPECT(rocm::has_single_bit(0x80000000u));
    EXPECT(not rocm::has_single_bit(0xFFFFFFFFu));
    EXPECT(not rocm::has_single_bit(0x80000001u));

    // unsigned char
    EXPECT(not rocm::has_single_bit(static_cast<unsigned char>(0)));
    EXPECT(rocm::has_single_bit(static_cast<unsigned char>(1)));
    EXPECT(rocm::has_single_bit(static_cast<unsigned char>(0x80)));
    EXPECT(not rocm::has_single_bit(static_cast<unsigned char>(0xFF)));

    // unsigned long long
    EXPECT(rocm::has_single_bit(1ull));
    EXPECT(rocm::has_single_bit(0x8000000000000000ull));
    EXPECT(not rocm::has_single_bit(0xFFFFFFFFFFFFFFFFull));
}

TEST_CASE(rotl_test)
{
    // Zero rotation
    EXPECT(rocm::rotl(0xABCDEF01u, 0) == 0xABCDEF01u);

    // Basic rotations
    EXPECT(rocm::rotl(1u, 1) == 2u);
    EXPECT(rocm::rotl(1u, 4) == 16u);
    EXPECT(rocm::rotl(1u, 31) == 0x80000000u);
    EXPECT(rocm::rotl(0x80000000u, 1) == 1u);

    // Multi-nibble rotations
    EXPECT(rocm::rotl(0x12345678u, 4) == 0x23456781u);
    EXPECT(rocm::rotl(0x12345678u, 8) == 0x34567812u);
    EXPECT(rocm::rotl(0x12345678u, 16) == 0x56781234u);
    EXPECT(rocm::rotl(0x12345678u, 24) == 0x78123456u);

    // Full rotation (identity)
    EXPECT(rocm::rotl(0xDEADBEEFu, 32) == 0xDEADBEEFu);

    // Negative rotation (equivalent to rotr)
    EXPECT(rocm::rotl(1u, -1) == 0x80000000u);
    EXPECT(rocm::rotl(0x12345678u, -4) == 0x81234567u);

    // unsigned char (8-bit)
    EXPECT(rocm::rotl(static_cast<unsigned char>(1), 1) == 2);
    EXPECT(rocm::rotl(static_cast<unsigned char>(0x80), 1) == 1);
    EXPECT(rocm::rotl(static_cast<unsigned char>(0x0F), 4) == 0xF0);

    // unsigned long long (64-bit)
    EXPECT(rocm::rotl(1ull, 63) == 0x8000000000000000ull);
    EXPECT(rocm::rotl(0x8000000000000000ull, 1) == 1ull);
}

TEST_CASE(rotr_test)
{
    // Zero rotation
    EXPECT(rocm::rotr(0xABCDEF01u, 0) == 0xABCDEF01u);

    // Basic rotations
    EXPECT(rocm::rotr(1u, 1) == 0x80000000u);
    EXPECT(rocm::rotr(2u, 1) == 1u);
    EXPECT(rocm::rotr(0x80000000u, 1) == 0x40000000u);

    // Multi-nibble rotations
    EXPECT(rocm::rotr(0x12345678u, 4) == 0x81234567u);
    EXPECT(rocm::rotr(0x12345678u, 8) == 0x78123456u);
    EXPECT(rocm::rotr(0x12345678u, 16) == 0x56781234u);
    EXPECT(rocm::rotr(0x12345678u, 24) == 0x34567812u);

    // Full rotation (identity)
    EXPECT(rocm::rotr(0xDEADBEEFu, 32) == 0xDEADBEEFu);

    // Negative rotation (equivalent to rotl)
    EXPECT(rocm::rotr(1u, -1) == 2u);
    EXPECT(rocm::rotr(0x12345678u, -4) == 0x23456781u);

    // unsigned char (8-bit)
    EXPECT(rocm::rotr(static_cast<unsigned char>(1), 1) == 0x80);
    EXPECT(rocm::rotr(static_cast<unsigned char>(0x80), 1) == 0x40);
    EXPECT(rocm::rotr(static_cast<unsigned char>(0xF0), 4) == 0x0F);

    // unsigned long long (64-bit)
    EXPECT(rocm::rotr(1ull, 1) == 0x8000000000000000ull);
    EXPECT(rocm::rotr(0x8000000000000000ull, 63) == 1ull);
}

TEST_CASE(rotl_rotr_inverse)
{
    // rotl(rotr(x, s), s) == x
    EXPECT(rocm::rotl(rocm::rotr(0xDEADBEEFu, 1), 1) == 0xDEADBEEFu);
    EXPECT(rocm::rotl(rocm::rotr(0xDEADBEEFu, 7), 7) == 0xDEADBEEFu);
    EXPECT(rocm::rotl(rocm::rotr(0xDEADBEEFu, 13), 13) == 0xDEADBEEFu);
    EXPECT(rocm::rotl(rocm::rotr(0xDEADBEEFu, 31), 31) == 0xDEADBEEFu);

    // rotr(rotl(x, s), s) == x
    EXPECT(rocm::rotr(rocm::rotl(0xCAFEBABEu, 1), 1) == 0xCAFEBABEu);
    EXPECT(rocm::rotr(rocm::rotl(0xCAFEBABEu, 7), 7) == 0xCAFEBABEu);
    EXPECT(rocm::rotr(rocm::rotl(0xCAFEBABEu, 13), 13) == 0xCAFEBABEu);
    EXPECT(rocm::rotr(rocm::rotl(0xCAFEBABEu, 31), 31) == 0xCAFEBABEu);

    // rotl(x, s) == rotr(x, -s)
    EXPECT(rocm::rotl(0x12345678u, 5) == rocm::rotr(0x12345678u, -5));
    EXPECT(rocm::rotl(0x12345678u, 17) == rocm::rotr(0x12345678u, -17));
}

TEST_CASE(count_relationships)
{
    // countl_zero(x) + countl_one(~x) covers all digits (complement relationship)
    // popcount(x) + popcount(~x) == digits
    EXPECT(rocm::popcount(0x12345678u) + rocm::popcount(~0x12345678u) == 32);
    EXPECT(rocm::popcount(0xDEADBEEFu) + rocm::popcount(~0xDEADBEEFu) == 32);

    // countl_zero(x) == countl_one(~x)
    EXPECT(rocm::countl_zero(0x12345678u) == rocm::countl_one(~0x12345678u));
    EXPECT(rocm::countr_zero(0x12345678u) == rocm::countr_one(~0x12345678u));

    // bit_width(x) == digits - countl_zero(x)
    EXPECT(rocm::bit_width(0x12345678u) == 32 - rocm::countl_zero(0x12345678u));
    EXPECT(rocm::bit_width(42u) == 32 - rocm::countl_zero(42u));

    // has_single_bit(x) iff popcount(x) == 1
    EXPECT(rocm::has_single_bit(64u) == (rocm::popcount(64u) == 1));
    EXPECT(rocm::has_single_bit(65u) == (rocm::popcount(65u) == 1));

    // For powers of 2: bit_floor == bit_ceil == x
    EXPECT(rocm::bit_floor(64u) == 64u);
    EXPECT(rocm::bit_ceil(64u) == 64u);

    // bit_floor(x) <= x <= bit_ceil(x) for non-zero x
    EXPECT(rocm::bit_floor(100u) <= 100u);
    EXPECT(100u <= rocm::bit_ceil(100u));
    EXPECT(rocm::bit_floor(100u) == 64u);
    EXPECT(rocm::bit_ceil(100u) == 128u);
}
