#ifndef MIGRAPHX_GUARD_MIGRAPHX_PAR_HPP
#define MIGRAPHX_GUARD_MIGRAPHX_PAR_HPP

#include <migraphx/config.hpp>
#ifdef MIGRAPHX_HAS_EXECUTORS
#include <execution>
#else
#include <migraphx/simple_par_for.hpp>
#endif
#include <algorithm>
#include <mutex>
#include <vector>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

namespace detail {

struct exception_list
{
    std::vector<std::exception_ptr> exceptions;
    std::mutex m;
    void add_exception()
    {
        std::lock_guard<std::mutex> guard(m);
        exceptions.push_back(std::current_exception());
    }
    void throw_if_exception() const
    {
        if(not exceptions.empty())
            std::rethrow_exception(exceptions.front());
    }
};

template<class F>
auto par_collect_exceptions(exception_list& ex, F f)
{
    return [=, &ex](auto&&... xs) {
        try
        {
            f(std::forward<decltype(xs)>(xs)...);
        }
        catch(...)
        {
            ex.add_exception();
        }
    };
}

} // namespace detail

template <class InputIt, class OutputIt, class UnaryOperation>
OutputIt par_transform(InputIt first1, InputIt last1, OutputIt d_first, UnaryOperation unary_op)
{
#ifdef MIGRAPHX_HAS_EXECUTORS
    return std::transform(std::execution::par, first1, last1, d_first, std::move(unary_op));
#else
    simple_par_for(last1 - first1, [&](auto i) { d_first[i] = unary_op(first1[i]); });
    return d_first + (last1 - first1);
#endif
}

template <class InputIt1, class InputIt2, class OutputIt, class BinaryOperation>
OutputIt par_transform(
    InputIt1 first1, InputIt1 last1, InputIt2 first2, OutputIt d_first, BinaryOperation binary_op)
{
#ifdef MIGRAPHX_HAS_EXECUTORS
    return std::transform(
        std::execution::par, first1, last1, first2, d_first, std::move(binary_op));
#else
    simple_par_for(last1 - first1, [&](auto i) { d_first[i] = binary_op(first1[i], first2[i]); });
    return d_first + (last1 - first1);
#endif
}

template <class InputIt, class UnaryFunction>
void par_for_each(InputIt first, InputIt last, UnaryFunction f)
{
#ifdef MIGRAPHX_HAS_EXECUTORS
    // Propagate the exception
    detail::exception_list ex;
    std::for_each(std::execution::par, first, last, detail::par_collect_exceptions(ex, std::move(f)));
    ex.throw_if_exception();
#else
    simple_par_for(last - first, [&](auto i) { f(first[i]); });
#endif
}

template <class... Ts>
auto par_copy_if(Ts&&... xs)
{
#ifdef MIGRAPHX_HAS_EXECUTORS
    return std::copy_if(std::execution::par, std::forward<Ts>(xs)...);
#else
    return std::copy_if(std::forward<Ts>(xs)...);
#endif
}

template <class... Ts>
auto par_sort(Ts&&... xs)
{
#ifdef MIGRAPHX_HAS_EXECUTORS
    return std::sort(std::execution::par, std::forward<Ts>(xs)...);
#else
    return std::sort(std::forward<Ts>(xs)...);
#endif
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
#endif // MIGRAPHX_GUARD_MIGRAPHX_PAR_HPP
