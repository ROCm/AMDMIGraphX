#include <migraphx/config.hpp>
#include <migraphx/register_op.hpp>
#include <migraphx/reflect.hpp>
#include <migraphx/par_for.hpp>
#include <migraphx/context.hpp>
#include <migraphx/cpu/context.hpp>
#include <migraphx/cpu/dnnl.hpp>
#include <migraphx/op/convolution.hpp>
#include <migraphx/op/quant_convolution.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace cpu {

template <class V, class T, class... Ts>
void visit_quantize_impl(V&& v, T&& x, Ts&&... xs)
{
    x.visit([&](auto y) { visit_all(xs...)([&](auto... ys) { v(y, ys...); }); });
}

template <class T, class... Ts>
auto visit_quantize(T&& x, Ts&&... xs)
{
    return [&](auto v) {
        // Workaround for https://gcc.gnu.org/bugzilla/show_bug.cgi?id=70100
        visit_quantize_impl(v, x, xs...);
    };
}

template <class Op>
struct cpu_convolution : auto_register_op<cpu_convolution<Op>>
{
    Op op;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return migraphx::reflect(self.op, f);
    }

    std::string name() const { return "cpu::" + op.name(); }
    shape compute_shape(std::vector<shape> inputs) const
    {
        inputs.pop_back();
        return op.compute_shape(inputs);
    }

    std::ptrdiff_t output_alias(const std::vector<shape>& shapes) const
    {
        return shapes.size() - 1;
    }

    argument compute(context&, shape output_shape, std::vector<argument> args) const
    {
        visit_quantize(args.back(), args[0], args[1])([&](auto output, auto input, auto weights) {
            auto in_lens = input.get_shape().lens();

            auto wei_lens = weights.get_shape().lens();
            auto wei_n    = wei_lens[0];
            auto wei_c    = wei_lens[1];
            std::vector<std::size_t> win_size(wei_lens.begin() + 1, wei_lens.end());

            par_for(output_shape.elements(), [&](auto i) {
                auto idx_o = output_shape.multi(i);
                auto w     = idx_o[1];
                auto n_dim = idx_o.size();

                std::vector<std::ptrdiff_t> win_start;
                for(std::size_t dim = 2; dim < n_dim; ++dim)
                {
                    auto d_2 = dim - 2;
                    win_start.push_back(std::ptrdiff_t(idx_o[dim] * op.stride[d_2]) -
                                        std::ptrdiff_t(op.padding[d_2]));
                }
                const auto group_id = w / (wei_n / op.group);

                shape win_shape{output_shape.type(), win_size};

                double acc = 0.0;
                shape_for_each(win_shape, [&](auto idx_win) {
                    auto k           = idx_win[0];
                    const auto in_ch = group_id * wei_c + k;
                    std::vector<std::ptrdiff_t> idx(idx_o.begin(), idx_o.end());
                    idx[1] = in_ch;
                    std::transform(idx_win.begin() + 1,
                                   idx_win.end(),
                                   win_start.begin(),
                                   idx.begin() + 2,
                                   [](std::ptrdiff_t ii, std::ptrdiff_t jj) { return ii + jj; });
                    std::vector<std::ptrdiff_t> idx_wei(idx_o.size());
                    idx_wei[0] = w;
                    std::copy(idx_win.begin(), idx_win.end(), idx_wei.begin() + 1);
                    if(std::all_of(idx.begin() + 2, idx.end(), [&](auto ii) { return ii >= 0; }) and
                       std::equal(idx.begin(),
                                  idx.end(),
                                  in_lens.begin(),
                                  in_lens.end(),
                                  std::less<std::ptrdiff_t>{}))
                    {
                        acc +=
                            input(idx.begin(), idx.end()) * weights(idx_wei.begin(), idx_wei.end());
                    }
                });

                output[i] = acc;
            });
        });
        return args.back();
    }
};
template struct cpu_convolution<op::quant_convolution>;
template struct cpu_convolution<op::convolution>;

#if USE_DNNL
struct dnnl_convolution
    : dnnl_extend_op<dnnl_convolution, dnnl::convolution_forward, op::convolution>
{
    std::vector<int> arg_map(int) const { return {DNNL_ARG_SRC, DNNL_ARG_WEIGHTS}; }

    shape adjust_shape(shape x, int i) const
    {
        auto s = base_adjust_shape(std::move(x));
        if(i == 1 and op.group > 1)
        {
            // TODO: Add support for transposed weights
            if(not s.standard())
                MIGRAPHX_THROW("Weights for grouped convolution must be standard");
            auto lens = s.lens();
            lens.insert(lens.begin(), op.group);
            lens.at(1) /= op.group;
            return shape{s.type(), lens};
        }
        return s;
    }

    dnnl::convolution_forward::desc
    get_desc(const std::unordered_map<int, dnnl::memory::desc>& m) const
    {
        // In DNNL dilation is zero-based
        auto dilation = op.dilation;
        std::transform(
            dilation.begin(), dilation.end(), dilation.begin(), [](auto x) { return x - 1; });
        return dnnl::convolution_forward::desc(dnnl::prop_kind::forward_inference,
                                               dnnl::algorithm::convolution_auto,
                                               m.at(DNNL_ARG_SRC),
                                               m.at(DNNL_ARG_WEIGHTS),
                                               m.at(DNNL_ARG_DST),
                                               to_dnnl_dims(op.stride),
                                               to_dnnl_dims(dilation),
                                               to_dnnl_dims(op.padding),
                                               to_dnnl_dims(op.padding));
    }
};
#endif

} // namespace cpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
