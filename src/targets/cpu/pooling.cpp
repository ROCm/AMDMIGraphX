#include <migraphx/config.hpp>
#include <migraphx/register_op.hpp>
#include <migraphx/reflect.hpp>
#include <migraphx/par_for.hpp>
#include <migraphx/context.hpp>
#include <migraphx/cpu/context.hpp>
#include <migraphx/cpu/dnnl.hpp>
#include <migraphx/op/pooling.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace cpu {

struct max_pool
{
    static std::string name() { return "max"; }
    template <class T>
    static T start()
    {
        return std::numeric_limits<T>::lowest();
    }

    static double apply(double x, double y)
    {
        double m = std::max(x, y);
        return (m);
    }

    static double final(double x, std::size_t) { return (x); }
};

struct avg_pool
{
    static std::string name() { return "average"; }

    template <class T>
    static double start()
    {
        return 0.0;
    }

    static double apply(double x, double y) { return x + y; }

    static double final(double x, std::size_t y) { return (y == 0) ? 0.0 : (x / y); }
};

template <class Op>
struct cpu_pooling : auto_register_op<cpu_pooling<Op>>
{
    cpu_pooling() = default;

    cpu_pooling(op::pooling pop) : op(std::move(pop)) {}

    op::pooling op;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return migraphx::reflect(self.op, f);
    }

    std::string name() const { return "cpu::pooling_" + Op::name(); }
    shape compute_shape(std::vector<shape> inputs) const
    {
        inputs.pop_back();
        return op.normalize_compute_shape(inputs);
    }

    std::ptrdiff_t output_alias(const std::vector<shape>& shapes) const
    {
        return shapes.size() - 1;
    }

    argument compute(context&, const shape& output_shape, std::vector<argument> args) const
    {
        visit_all(args.back(), args[0])([&](auto output, auto input) {
            using type   = typename decltype(output)::value_type;
            auto in_s    = input.get_shape();
            auto in_lens = in_s.lens();
            std::vector<std::size_t> vec_len(in_lens.begin() + 2, in_lens.end());

            par_for(output_shape.elements(), [&](auto i) {
                auto idx_o = output_shape.multi(i);
                auto n_dim = idx_o.size();
                std::vector<std::size_t> win_start;
                std::vector<std::size_t> win_size;
                for(std::size_t dim = 2; dim < n_dim; ++dim)
                {
                    auto d_2  = dim - 2;
                    int start = static_cast<int>(idx_o[dim] * op.stride[d_2]) -
                                static_cast<int>(op.padding[d_2]);
                    int end = std::min(start + op.lengths[d_2], in_lens[dim]);
                    start   = std::max(start, 0);
                    win_start.push_back(start);
                    win_size.push_back(end - start);
                }

                shape win_shape{output_shape.type(), win_size};
                auto pool_size = win_shape.elements();
                double acc     = Op::template start<type>();
                shape_for_each(win_shape, [&](auto idx_w) {
                    auto idx = idx_o;
                    std::transform(idx_w.begin(),
                                   idx_w.end(),
                                   win_start.begin(),
                                   idx.begin() + 2,
                                   [](auto ii, auto jj) { return ii + jj; });
                    if(std::all_of(idx.begin() + 2, idx.end(), [&](auto ii) { return ii >= 0; }) and
                       idx < in_lens)
                    {
                        acc = Op::apply(acc, input[in_s.index(idx)]);
                    }
                });

                output[i] = type(Op::final(acc, pool_size));
            });
        });

        return args.back();
    }
};

template struct cpu_pooling<avg_pool>;
template struct cpu_pooling<max_pool>;

struct dnnl_pooling : dnnl_extend_op<dnnl_pooling, dnnl::pooling_forward, op::pooling>
{
    std::vector<int> arg_map(int) const { return {DNNL_ARG_SRC}; }

    dnnl::pooling_forward::desc get_desc(const std::unordered_map<int, dnnl::memory::desc>& m) const
    {
        auto algo  = op.mode == "max" ? dnnl::algorithm::pooling_max : dnnl::algorithm::pooling_avg;
        auto kdims = op.kdims();
        std::vector<size_t> padding_l(op.padding.begin(), op.padding.begin() + kdims);
        std::vector<size_t> padding_r(op.padding.begin() + kdims, op.padding.end());
        return {dnnl::prop_kind::forward_inference,
                algo,
                m.at(DNNL_ARG_SRC),
                m.at(DNNL_ARG_DST),
                to_dnnl_dims(op.stride),
                to_dnnl_dims(op.lengths),
                to_dnnl_dims(padding_l),
                to_dnnl_dims(padding_r)};
    }
};

} // namespace cpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
