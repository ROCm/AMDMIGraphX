#ifndef MIGRAPHX_GUARD_RTGLIB_MARKER_ROCTX_HPP
#define MIGRAPHX_GUARD_RTGLIB_MARKER_ROCTX_HPP

#include <migraphx/marker.hpp>
#include <migraphx/dynamic_loader.hpp>

namespace migraphx {
namespace driver {
inline namespace MIGRAPHX_INLINE_NS {

class marker_roctx
{
    private:
    dynamic_loader lib;
    std::function<void(const char*)> sym_roctx_mark;
    std::function<uint64_t(const char*)> sym_roctx_range_start;
    std::function<void(uint64_t)> sym_roctx_range_stop;

    std::function<int(const char*)> sym_roctx_range_push;
    std::function<int()> sym_roctx_range_pop;

    public:
    void initalize_roctx()
    {
        lib                   = migraphx::dynamic_loader{"libroctx64.so"};
        sym_roctx_mark        = lib.get_function<void(const char*)>("roctxMarkA");
        sym_roctx_range_start = lib.get_function<uint64_t(const char*)>("roctxRangeStartA");
        sym_roctx_range_stop  = lib.get_function<void(uint64_t)>("roctxRangeStop");

        sym_roctx_range_push = lib.get_function<int(const char*)>("roctxRangePushA");
        sym_roctx_range_pop  = lib.get_function<int()>("roctxRangePop");

        sym_roctx_mark("rocTX marker created.");
    }

    void mark_start(instruction_ref ins)
    {
        std::string text = "Marker start: " + ins->name();
        sym_roctx_range_push(text.c_str());
    }
    void mark_stop(instruction_ref ins __attribute__((unused))) { sym_roctx_range_pop(); }
    uint64_t mark_start(const program& p __attribute__((unused)))
    {
        sym_roctx_mark("rocTX marker created: ");
        return sym_roctx_range_start("0");
    }
    void mark_stop(const program& p __attribute__((unused)), uint64_t range_id)
    {
        sym_roctx_range_stop(range_id);
    }
};

} // namespace MIGRAPHX_INLINE_NS
} // namespace driver
} // namespace migraphx

#endif
