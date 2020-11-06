#ifndef MIGRAPHX_GUARD_AMDMIGRAPHX_DNNL_HPP
#define MIGRAPHX_GUARD_AMDMIGRAPHX_DNNL_HPP

#include <migraphx/config.hpp>
#ifdef USE_DNNL
#include <dnnl.hpp>
#include <migraphx/errors.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace cpu {



inline dnnl::memory::data_type to_dnnl_memory_data_type(shape::type_t t)
{
    using dt = dnnl::memory::data_type;
    using st = shape::type_t;
    switch(t)
    {
        case st::half_type: return dt::f16;
        case st::float_type: return dt::f32;
        case st::int32_type: return dt::s32;
        case st::int8_type: return dt::s8;
        case st::uint8_type: return dt::u8;
        default: MIGRAPHX_THROW("Unsupported data type");
    }
}

inline dnnl::memory::format_tag to_dnnl_memory_format_tag(std::size_t n)
{
    switch(n)
    {
        case 1: return dnnl::memory::format_tag::a;
        case 2: return dnnl::memory::format_tag::ab;
        case 3: return dnnl::memory::format_tag::abc;
        case 4: return dnnl::memory::format_tag::abcd;
        case 5: return dnnl::memory::format_tag::abcde;
        case 6: return dnnl::memory::format_tag::abcdef;
        default: MIGRAPHX_THROW("Unsupported tensor size: " + std::to_string(n));
    }
}

inline dnnl::memory::desc to_dnnl_memory_desc(const shape& s)
{
    if (not s.standard())
        MIGRAPHX_THROW("Unsupported layout");
    dnnl::memory::dims lens(s.lens().begin(), s.lens().end());
    return dnnl::memory::desc(lens, to_dnnl_memory_data_type(s.type()), to_dnnl_memory_format_tag(lens.size()));
}

inline dnnl::memory to_dnnl_memory(const argument& a, dnnl::engine& engine)
{
    return dnnl::memory(to_dnnl_memory_desc(a.get_shape()), engine, a.data());
}

} // namespace cpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif

#endif
