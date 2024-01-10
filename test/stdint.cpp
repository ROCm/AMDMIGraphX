#include <dual_test.hpp>
#include <rocm/stdint.hpp>

ROCM_DUAL_TEST_CASE()
{
    static_assert(sizeof(rocm::int8_t) == 1, "int8_t must be 1 bytes");
    static_assert(sizeof(rocm::uint8_t) == 1, "uint8_t must be 1 bytes");
    static_assert(sizeof(rocm::int16_t) == 2, "int16_t must be 2 bytes");
    static_assert(sizeof(rocm::uint16_t) == 2, "uint16_t must be 2 bytes");
    static_assert(sizeof(rocm::int32_t) == 4, "int32_t must be 4 bytes");
    static_assert(sizeof(rocm::uint32_t) == 4, "uint32_t must be 4 bytes");
    static_assert(sizeof(rocm::int64_t) == 8, "int64_t must be 8 bytes");
    static_assert(sizeof(rocm::uint64_t) == 8, "uint64_t must be 8 bytes");
}
