#include <migraphx/argument.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

argument::argument(const shape& s) : m_shape(s)
{
    auto buffer = make_shared_array<char>(s.bytes());
    m_data      = [=]() mutable { return buffer.get(); };
}

argument::argument(shape s, std::nullptr_t) : m_shape(std::move(s)), m_data([] { return nullptr; })
{
}

char* argument::data() const { return m_data(); }

bool argument::empty() const { return not m_data; }

const shape& argument::get_shape() const { return this->m_shape; }

argument argument::reshape(const shape& s) const
{
    argument self = *this;
    return {s, [=]() mutable { return self.data(); }};
}

argument argument::share() const
{
    auto self = std::make_shared<argument>(*this);
    return {m_shape, [self]() mutable { return self->data(); }};
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
