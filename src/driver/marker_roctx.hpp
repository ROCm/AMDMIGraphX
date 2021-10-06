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

    bool init = false;

    public:
    void initalize_roctx()
    {
        lib                   = migraphx::dynamic_loader{"libroctx64.so"};
        sym_roctx_mark        = lib.get_function<void(const char*)>("roctxMarkA");
        sym_roctx_range_start = lib.get_function<uint64_t(const char*)>("roctxRangeStartA");
        sym_roctx_range_stop  = lib.get_function<void(uint64_t)>("roctxRangeStop");

        sym_roctx_range_push = lib.get_function<int(const char*)>("roctxRangePushA");
        sym_roctx_range_pop  = lib.get_function<int()>("roctxRangePop");
    }

    uint64_t mark_range_start(uint64_t range_id)
    {
        if(init)
        {
            sym_roctx_mark("rocTX marker created.");
            init = true;
        }
        return sym_roctx_range_start(std::to_string(range_id).c_str());
    }

    void mark_range_finish(uint64_t range_id) { return sym_roctx_range_stop(range_id); }

    void mark_ins_start(std::string start_log) { sym_roctx_range_push(start_log.c_str()); }

    void mark_ins_finish() { sym_roctx_range_pop(); }

    void mark_program_start() {}
    void mark_program_finish() {}
};

} // namespace MIGRAPHX_INLINE_NS
} // namespace driver
} // namespace migraphx

#endif
