#ifndef MIGRAPHX_GUARD_MIGRAPHLIB_ARGUMENT_HPP
#define MIGRAPHX_GUARD_MIGRAPHLIB_ARGUMENT_HPP

#include <migraphx/shape.hpp>
#include <migraphx/raw_data.hpp>
#include <migraphx/config.hpp>
#include <migraphx/make_shared_array.hpp>
#include <functional>
#include <utility>

// clang-format off
namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

/**
 * @brief Arguments passed to instructions
 *
 * An `argument` can represent a raw buffer of data that either be referenced from another element
 * or it can be owned by the argument.
 *
 */
struct argument : raw_data<argument>
{
    argument() = default;

    argument(const shape& s);

    template <class F, MIGRAPHX_REQUIRES(std::is_pointer<decltype(std::declval<F>()())>{})>
    argument(shape s, F d)
        : m_shape(std::move(s)),
          m_data({[f = std::move(d)]() mutable { return reinterpret_cast<char*>(f()); }})

    {
    }
    template <class T>
    argument(shape s, T* d)
        : m_shape(std::move(s)), m_data({[d] { return reinterpret_cast<char*>(d); }})
    {
    }

    template <class T>
    argument(shape s, std::shared_ptr<T> d)
        : m_shape(std::move(s)), m_data({[d] { return reinterpret_cast<char*>(d.get()); }})
    {
    }

    argument(shape s, std::nullptr_t);
    
    argument(const std::vector<argument>& args);

    static argument load(const shape& s, char* buffer);

    /// Provides a raw pointer to the data
    char* data() const;

    /// Whether data is available
    bool empty() const;

    const shape& get_shape() const;

    argument reshape(const shape& s) const;

    argument copy() const;

    /// Make copy of the argument that is always sharing the data
    argument share() const;

    std::vector<argument> get_sub_objects() const;

    private:
    struct data_t
    {
        std::function<char*()> get = nullptr;
        std::vector<data_t> sub = {};
        data_t share() const;
        static data_t from_args(const std::vector<argument>& args);
    };
    argument(const shape& s, const data_t& d);
    shape m_shape;
    data_t m_data{};
};

void migraphx_to_value(value& v, const argument& a);
void migraphx_from_value(const value& v, argument& a);

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
// clang-format on

#endif
