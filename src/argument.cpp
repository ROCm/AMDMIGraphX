#include <migraphx/argument.hpp>
#include <migraphx/functional.hpp>
#include <unordered_map>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

argument::argument(const shape& s) : m_shape(s)
{
    auto buffer = make_shared_array<char>(s.bytes());
    m_data      = {[=]() mutable { return buffer.get(); }};
}

argument::argument(shape s, std::nullptr_t)
    : m_shape(std::move(s)), m_data({[] { return nullptr; }})
{
}

argument::argument(const shape& s, const argument::data_t& d) : m_shape(s), m_data(d) {}

argument argument::load(const shape& s, char* buffer)
{
    if(s.type() != shape::tuple_type)
        return argument{s, buffer};
    // Collect all shapes
    std::unordered_map<std::size_t, shape> shapes;
    {
        // cppcheck-suppress variableScope
        std::size_t i = 0;
        fix([&](auto self, auto ss) {
            if(ss.sub_shapes().empty())
            {
                shapes[i] = ss;
                i++;
            }
            else
            {
                for(auto&& child : ss.sub_shapes())
                    self(child);
            }
        })(s);
    }
    // Sort by type size
    std::vector<std::size_t> order(shapes.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), by(std::greater<>{}, [&](auto i) {
                  return shapes[i].type_size();
              }));
    // Compute offsets
    std::unordered_map<std::size_t, std::size_t> offsets;
    std::size_t offset = 0;
    for(auto i : order)
    {
        offsets[i] = offset;
        offset += shapes[i].bytes();
    }
    assert(offset == s.bytes());

    // cppcheck-suppress variableScope
    std::size_t i = 0;
    return fix<argument>([&](auto self, auto ss) {
        if(ss.sub_shapes().empty())
        {
            argument r{shapes[i], buffer + offsets[i]};
            i++;
            return r;
        }
        std::vector<argument> subs;
        std::transform(ss.sub_shapes().begin(),
                       ss.sub_shapes().end(),
                       std::back_inserter(subs),
                       [&](auto child) { return self(child); });
        return argument{subs};
    })(s);
}

std::vector<shape> to_shapes(const std::vector<argument>& args)
{
    std::vector<shape> shapes;
    std::transform(args.begin(), args.end(), std::back_inserter(shapes), [](auto&& arg) {
        return arg.get_shape();
    });
    return shapes;
}

argument::argument(const std::vector<argument>& args)
    : m_shape(to_shapes(args)), m_data(data_t::from_args(args))
{
}

char* argument::data() const
{
    assert(m_shape.type() != shape::tuple_type);
    assert(not this->empty());
    return m_data.get();
}

bool argument::empty() const { return not m_data.get and m_data.sub.empty(); }

const shape& argument::get_shape() const { return this->m_shape; }

argument argument::reshape(const shape& s) const { return {s, this->m_data}; }

argument::data_t argument::data_t::share() const
{
    data_t result;
    if(this->get)
    {
        auto self  = std::make_shared<data_t>(*this);
        result.get = [self]() mutable { return self->get(); };
    }
    std::transform(sub.begin(), sub.end(), std::back_inserter(result.sub), [](const auto& d) {
        return d.share();
    });
    return result;
}

argument::data_t argument::data_t::from_args(const std::vector<argument>& args)
{
    data_t result;
    std::transform(args.begin(), args.end(), std::back_inserter(result.sub), [](auto&& arg) {
        return arg.m_data;
    });
    return result;
}

argument argument::copy() const
{
    argument result{this->get_shape()};
    auto* src = this->data();
    std::copy(src, src + this->get_shape().bytes(), result.data());
    return result;
}

argument argument::share() const { return {m_shape, m_data.share()}; }

std::vector<argument> argument::get_sub_objects() const
{
    std::vector<argument> result;
    assert(m_shape.sub_shapes().size() == m_data.sub.size());
    std::transform(m_shape.sub_shapes().begin(),
                   m_shape.sub_shapes().end(),
                   m_data.sub.begin(),
                   std::back_inserter(result),
                   [](auto&& s, auto&& d) {
                       return argument{s, d};
                   });
    return result;
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
