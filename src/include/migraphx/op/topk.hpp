#ifndef MIGRAPHX_GUARD_OPERATORS_GATHER_HPP
#define MIGRAPHX_GUARD_OPERATORS_GATHER_HPP

#include <algorithm>
#include <migraphx/check_shapes.hpp>
#include <migraphx/config.hpp>
#include <migraphx/value.hpp>
#include <migraphx/op/normalize_attribute.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/par_for.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

struct topk
{
    int64_t k;
    int64_t axis = 0;
    bool largest = true;
    bool sorted  = true;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.k, "k"),
                    f(self.axis, "axis"),
                    f(self.largest, "largest"),
                    f(self.sorted, "sorted"));
    }

    value attributes() const
    {
        value normalize;
        normalize["axis"] = value::array{normalize_attribute::include_min};
        return {{"normalize_axes", normalize}};
    }

    std::string name() const { return "topk"; }

    shape normalize_compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this}.has(1).standard();
        auto lens = inputs.at(0).lens();
        auto type = inputs.at(0).type();

        lens[axis] = k;

        shape s_val{type, lens};
        shape s_ind{shape::int64_type, lens};

        return shape({s_val, s_ind});
    }

    // template <class T, class Op>
    // void heapify(const T& data,
    //              const shape& iss,
    //              const std::vector<std::size_t>& sidx,
    //              std::vector<int>& indices,
    //              int n,
    //              int i,
    //              Op op) const
    // {
    //     int index = i;
    //     auto idx  = sidx;
    //     auto idxl = sidx;
    //     auto idxr = sidx;
    //     while(index < n)
    //     {
    //         auto pre_index = index;
    //         int l          = 2 * index + 1;
    //         int r          = 2 * index + 2;
    //         idx[axis]      = indices[index];
    //         if(l < n)
    //         {
    //             idxl[axis] = indices[l];
    //         }
    //         if(r < n)
    //         {
    //             idxr[axis] = indices[r];
    //         }

    //         if(l < n && op(data[iss.index(idxl)], data[iss.index(idx)]))
    //         {
    //             index = l;
    //         }

    //         if(r < n && op(data[iss.index(idxr)], data[iss.index(idx)]))
    //         {
    //             index = r;
    //             if(op(data[iss.index(idxl)], data[iss.index(idxr)]))
    //             {
    //                 index = l;
    //             }
    //         }

    //         if(index == pre_index)
    //         {
    //             break;
    //         }
    //         std::swap(indices[index], indices[pre_index]);
    //     }
    // }

    template <class T, class Op>
    void build_heap(const T& data,
                    const shape& iss,
                    const std::vector<std::size_t>& sidx,
                    std::vector<int>& indices,
                    int n,
                    Op op) const
    {
        std::make_heap(indices.begin(), indices.begin() + n, [&](auto i1, auto i2) {
            auto idx1  = sidx;
            auto idx2  = sidx;
            idx1[axis] = i1;
            idx2[axis] = i2;
            return op(data[iss.index(idx1)], data[iss.index(idx2)]);
        });
    }

    template <class T, class Op>
    void heap_add(const T& data,
                  const shape& iss,
                  std::vector<std::size_t> sidx,
                  std::vector<int>& indices,
                  int n,
                  const int& val,
                  Op op) const
    {
        auto comp = [&](auto i1, auto i2) {
            auto idx1  = sidx;
            auto idx2  = sidx;
            idx1[axis] = i1;
            idx2[axis] = i2;
            return op(data[iss.index(idx1)], data[iss.index(idx2)]);
        };

        std::pop_heap(indices.begin(), indices.end(), comp);

        auto idx1  = sidx;
        auto idx2  = sidx;
        idx1[axis] = indices.back();
        idx2[axis] = val;
        if(op(data[iss.index(idx2)], data[iss.index(idx1)]))
        {
            indices.back() = val;
        }

        std::push_heap(indices.begin(), indices.end(), comp);
    }



    template <class T>
    void heap_add(std::vector<int>& indices,
                  const int& val,
                  T comp) const
    {
        std::pop_heap(indices.begin(), indices.end(), comp);
        if(comp(val, indices.back()))
        {
            indices.back() = val;
        }
        std::push_heap(indices.begin(), indices.end(), comp);
    }


    template <class T, class Op>
    void heap_sort(const T& data,
                   const shape& iss,
                   const std::vector<std::size_t>& sidx,
                   std::vector<int>& indices,
                   int n,
                   Op op) const
    {
        auto comp = [&](auto i1, auto i2) {
            auto idx1  = sidx;
            auto idx2  = sidx;
            idx1[axis] = i1;
            idx2[axis] = i2;
            return op(data[iss.index(idx1)], data[iss.index(idx2)]);
        };

        std::make_heap(indices.begin(), indices.end(), comp);
        std::sort_heap(indices.begin(), indices.end(), comp);
    }


    template <class T>
    void heap_sort(std::vector<int>& indices,
                   T comp) const
    {
        std::make_heap(indices.begin(), indices.end(), comp);
        std::sort_heap(indices.begin(), indices.end(), comp);
    }

    template <class T>
    void topk_value(std::vector<int>& indices,
                    std::size_t n,
                    T comp) const
    {
        std::make_heap(indices.begin(), indices.end(), comp);

        for(int i = indices.size(); i < n; ++i)
        {
            heap_add(indices, i, comp);
        }
    }


    template <class T, class Op>
    void topk_value(const T& data,
                    const shape& iss,
                    const std::vector<std::size_t>& sidx,
                    std::vector<int>& indices,
                    int n,
                    int kk,
                    Op op) const
    {
        auto comp = [&](auto i1, auto i2) {
            auto idx1  = sidx;
            auto idx2  = sidx;
            idx1[axis] = i1;
            idx2[axis] = i2;
            return op(data[iss.index(idx1)], data[iss.index(idx2)]);
        };

        std::make_heap(indices.begin(), indices.end(), comp);

        for(int i = kk; i < n; ++i)
        {
            heap_add(data, iss, sidx, indices, kk, i, op);
        }
    }


    argument compute(const shape& output_shape, std::vector<argument> args) const
    {
        auto vec_ss = output_shape.sub_shapes();
        argument res_val{vec_ss.front()};
        argument res_ind{vec_ss.back()};
        auto in_s      = args.front().get_shape();
        auto out_s     = vec_ss.front();
        auto comp_lens = in_s.lens();
        auto axis_dim  = comp_lens[axis];

        // compute shape
        comp_lens[axis] = 1;
        shape comp_s{in_s.type(), comp_lens};
        auto op = [](auto largest1)
        {
            return largest1 ? std::greater<>{} : std::less<>{};
        };

        visit_all(res_val, args.front())([&](auto out_val, auto input) {
            auto* out_ind = res_ind.cast<int64_t>();
            par_for(comp_s.elements(), [&](auto i) {
                auto idx = comp_s.multi(i);
                std::vector<int> indices(k);
                std::iota(indices.begin(), indices.end(), 0);

                auto comp = [&](auto i1, auto i2) {
                        auto idx1  = idx;
                        auto idx2  = idx;
                        idx1[axis] = i1;
                        idx2[axis] = i2;
                        return this->largest ? std::greater<>{}(input[in_s.index(idx1)], input[in_s.index(idx2)]) : std::less<>{}(input[in_s.index(idx1)], input[in_s.index(idx2)]);
                    };

                topk_value(indices, axis_dim, comp);
                heap_sort(indices, comp);


                // largest ? this->topk_value(input, in_s, idx, indices, axis_dim, k, std::greater<>{})
                //         : this->topk_value(input, in_s, idx, indices, axis_dim, k, std::less<>{});

                // if(sorted)
                // {
                //     largest ? this->heap_sort(input, in_s, idx, indices, k, std::greater<>{})
                //             : this->heap_sort(input, in_s, idx, indices, k, std::less<>{});
                // }

                auto out_idx = idx;
                auto in_idx  = idx;
                for(auto j : range(indices.size()))
                {
                    out_idx[axis]                 = j;
                    in_idx[axis]                  = indices[j];
                    out_val[out_s.index(out_idx)] = input[in_s.index(in_idx)];
                    out_ind[out_s.index(out_idx)] = indices[j];
                }
            });
        });

        return argument({res_val, res_ind});
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
