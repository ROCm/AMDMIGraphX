#include <iterator>
#include <migraphx/run_loop.hpp>
#include <migraphx/gpu/loop.hpp>
#include <migraphx/gpu/context.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

shape hip_loop::compute_shape(std::vector<shape> inputs, std::vector<module_ref> mods) const
{
    inputs.pop_back();
    inputs.pop_back();
    inputs.erase(inputs.begin() + 3);
    inputs.erase(inputs.begin() + 1);
    return op.compute_shape(inputs, mods);
}

struct gpu_loop
{
    int64_t max_iter_num = 0;

    template <class T>
    void copy_arg(context& ctx, const argument& arg, T& var, bool from_to_var) const
    {
        argument arg_var{arg.get_shape(), &var};
        if(from_to_var)
        {
            copy_to_gpu(ctx, arg_var, arg);
        }
        else
        {
            copy_from_gpu(ctx, arg, arg_var);
        }
    }

    void
    concat_scan_outputs(const std::vector<argument>&, const std::vector<argument>&, const int) const
    {
    }

    void set_zero(const std::vector<argument>& scan_outputs, const int iter) const
    {
        if(iter >= max_iter_num)
            return;

        auto elem_num = max_iter_num - iter;
        for(auto& out : scan_outputs)
        {
            auto s    = out.get_shape();
            auto size = s.bytes() / max_iter_num;
            (void)hipMemset(out.data() + iter * size, 0, size * elem_num);
        }
    }
};

argument
hip_loop::compute(context& ctx,
                  const shape&,
                  const std::vector<argument>& args,
                  const std::vector<module_ref>& mods,
                  const std::function<std::vector<argument>(
                      module_ref&, const std::unordered_map<std::string, argument>&)>& run) const
{
    return run_loop(gpu_loop{op.max_iter_num}, ctx, args, mods, run);
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
