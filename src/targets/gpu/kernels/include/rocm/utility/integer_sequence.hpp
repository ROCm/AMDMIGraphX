#ifndef ROCM_GUARD_UTILITY_INTEGER_SEQUENCE_HPP
#define ROCM_GUARD_UTILITY_INTEGER_SEQUENCE_HPP

#include <rocm/config.hpp>
#include <rocm/stdint.hpp>

namespace rocm {
inline namespace ROCM_INLINE_NS {

template <class T, T... Ints>
struct integer_sequence
{
    using value_type = T;

    static constexpr size_t size() noexcept { return sizeof...(Ints); }
};

template <size_t... Ints>
using index_sequence = integer_sequence<size_t, Ints...>;

template <class T, T N>
using make_integer_sequence = __make_integer_seq<integer_sequence, T, N>;

template <size_t N>
using make_index_sequence = make_integer_sequence<size_t, N>;

template <class... Ts>
using index_sequence_for = make_index_sequence<sizeof...(Ts)>;

} // namespace ROCM_INLINE_NS
} // namespace rocm
#endif // ROCM_GUARD_UTILITY_INTEGER_SEQUENCE_HPP
