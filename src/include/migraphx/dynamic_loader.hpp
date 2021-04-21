#ifndef MIGRAPHX_GUARD_MIGRAPHX_DYNAMIC_LOADER_HPP
#define MIGRAPHX_GUARD_MIGRAPHX_DYNAMIC_LOADER_HPP

#include <migraphx/config.hpp>
#include <migraphx/filesystem.hpp>
#include <functional>
#include <memory>
#include <vector>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct dynamic_loader_impl;

struct dynamic_loader
{
    dynamic_loader() = default;

    dynamic_loader(const fs::path& p);

    dynamic_loader(const char* image, std::size_t size);

    dynamic_loader(const std::vector<char>& buffer);

    std::shared_ptr<void> get_symbol(const std::string& name) const;

    template <class F>
    std::function<F> get_function(const std::string& name) const
    {
        auto s = get_symbol(name);
        return [=](auto&&... xs) -> decltype(auto) {
            auto f = reinterpret_cast<std::add_pointer_t<F>>(s.get());
            return f(std::forward<decltype(xs)>(xs)...);
        };
    }

    private:
    std::shared_ptr<dynamic_loader_impl> impl;
};

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
#endif // MIGRAPHX_GUARD_MIGRAPHX_DYNAMIC_LOADER_HPP
