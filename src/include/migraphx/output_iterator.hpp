#ifndef MIGRAPHX_GUARD_MIGRAPHX_OUTPUT_ITERATOR_HPP
#define MIGRAPHX_GUARD_MIGRAPHX_OUTPUT_ITERATOR_HPP

#include <migraphx/config.hpp>
#include <iterator>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

template <class F>
struct function_output_iterator
{
    F f;

    using self              = function_output_iterator;
    using difference_type   = void;
    using reference         = void;
    using value_type        = void;
    using pointer           = void;
    using iterator_category = std::output_iterator_tag;

    struct output_proxy
    {
        template <class T>
        output_proxy& operator=(const T& value)
        {
            assert(f);
            (*f)(value);
            return *this;
        }
        F* f;
    };
    output_proxy operator*() { return output_proxy{&f}; }
    self& operator++() { return *this; }
    self& operator++(int) { return *this; } // NOLINT
};

template <class F>
function_output_iterator<F> make_function_output_iterator(F f)
{
    return {std::move(f)};
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
#endif // MIGRAPHX_GUARD_MIGRAPHX_OUTPUT_ITERATOR_HPP
