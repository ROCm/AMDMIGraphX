#ifndef ROCM_GUARD_ITERATOR_ITERATOR_TRAITS_HPP
#define ROCM_GUARD_ITERATOR_ITERATOR_TRAITS_HPP

#include <rocm/config.hpp>
#include <rocm/stdint.hpp>
#include <rocm/type_traits.hpp>

namespace rocm {
inline namespace ROCM_INLINE_NS {

struct input_iterator_tag
{
};

struct output_iterator_tag
{
};

struct forward_iterator_tag : input_iterator_tag
{
};

struct bidirectional_iterator_tag : forward_iterator_tag
{
};

struct random_access_iterator_tag : bidirectional_iterator_tag
{
};

template <class Iterator>
struct iterator_traits
{
    using difference_type   = typename Iterator::difference_type;
    using value_type        = typename Iterator::value_type;
    using pointer           = typename Iterator::pointer;
    using reference         = typename Iterator::reference;
    using iterator_category = typename Iterator::iterator_category;
};

template <class T>
struct iterator_traits<T*>
{
    using difference_type   = ptrdiff_t;
    using value_type        = remove_cv_t<T>;
    using pointer           = T*;
    using reference         = T&;
    using iterator_category = random_access_iterator_tag;
};

} // namespace ROCM_INLINE_NS
} // namespace rocm
#endif // ROCM_GUARD_ITERATOR_ITERATOR_TRAITS_HPP
