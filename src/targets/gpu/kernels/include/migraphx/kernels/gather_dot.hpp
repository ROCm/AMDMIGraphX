/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 */

#ifndef MIGRAPHX_GUARD_KERNELS_GATHER_DOT_HPP
#define MIGRAPHX_GUARD_KERNELS_GATHER_DOT_HPP

#include <migraphx/kernels/index.hpp>
#include <migraphx/kernels/tensor_view.hpp>
#include <migraphx/kernels/uninitialized_buffer.hpp>
#include <migraphx/kernels/vec.hpp>
#include <migraphx/kernels/functional.hpp>
#include <migraphx/kernels/bit.hpp>
#include <migraphx/kernels/dpp.hpp>
#include <migraphx/kernels/tuple.hpp>
#include <migraphx/kernels/unpack_int4.hpp>
#include <migraphx/kernels/matmul.hpp>
#include <migraphx/kernels/ops.hpp>

namespace migraphx {
namespace gather_dot_detail {

// N elements at p, which need not be aligned to the vector
template <index_int N, class T>
__device__ vec<T, N> load_run(const T* p)
{
    vec<T, N> result;
    __builtin_memcpy(&result, p, sizeof(result));
    return result;
}

template <class T, index_int N>
__device__ void store_run(T* p, vec<T, N> x)
{
    __builtin_memcpy(p, &x, sizeof(x));
}

// The arguments Begin..Begin+N-1 packed
template <index_int Begin, index_int N, class... Ts>
constexpr auto pack_range(Ts... xs)
{
    return sequence_c<N>([&](auto... is) { return pack(arg_c<is + Begin>()(xs...)...); });
}

// The number of set bits of mask below bit i
constexpr index_int popcount_below(uint64_t mask, index_int i)
{
    return popcount(mask & ((uint64_t{1} << i) - 1));
}

// The index of a weight tensor at m in the weight space of Settings: the
// axes of the matrices with the pair axis split in pairs, of which a packed
// tensor holds one element each. The k and the n axes of the space are
// k_space and n_space.
template <class Settings, class W, class M>
__device__ auto weight_index(W, M m)
{
    constexpr index_int rank = get_shape_c<W>{}.lens.size();
    constexpr index_int axis = Settings::pair_axis;
    array<index_int, rank> result{};
    for(index_int d = 0; d < axis; d++)
        result[d] = m[d];
    if constexpr(pack_factor<typename W::type>{} > 1)
        result[axis] = m[axis];
    else
        result[axis] = m[axis] * Settings::pair + m[axis + 1];
    for(index_int d = axis + 1; d < rank; d++)
        result[d] = m[d + 1];
    return result;
}

template <class Settings, class W, class M>
__device__ auto weight_element(W w, M m)
{
    const auto i = weight_index<Settings>(w, m);
    if constexpr(pack_factor<typename W::type>{} > 1)
    {
        return unpack_int4(*remove_packed(&w[i]))[m[Settings::pair_axis + 1]];
    }
    else
    {
        return w[i];
    }
}

// The step of a weight tensor along the innermost k axis of the matrices
template <class Settings, class W>
constexpr index_int weight_step()
{
    constexpr auto s = get_shape_c<W>{};
    return s.strides[Settings::run_axis];
}

// N elements of a weight tensor along the innermost k axis of the matrices
// as they are stored: the bytes of a packed tensor, and one element of a
// tensor broadcast along the run
template <index_int N, class Settings, class W, class M>
__device__ auto load_weights(W w, M m)
{
    using type          = typename W::type;
    constexpr auto step = weight_step<Settings, W>();
    const auto i        = weight_index<Settings>(w, m);
    auto* p             = &w[i];
    if constexpr(pack_factor<type>{} > 1)
    {
        return load_run<N / pack_factor<type>{}>(remove_packed(p));
    }
    else if constexpr(step == 1)
    {
        return load_run<N>(p);
    }
    else
    {
        static_assert(step == 0, "a run is stored adjacent or broadcast");
        return *p;
    }
}

// The N elements from what load_weights returned
template <index_int N, class Settings, class W, class Stored>
__device__ auto expand_weights(W, Stored x)
{
    using type = typename W::type;
    if constexpr(pack_factor<type>{} > 1)
    {
        return unpack_int4(x);
    }
    else if constexpr(weight_step<Settings, W>() == 0)
    {
        return vec_generate<N>([&](auto) { return x; });
    }
    else
    {
        return x;
    }
}

// Computes the matrix of one table entry: element (k, j) is column n0 + j of
// the entry, computed by f from the weight tensors
template <class Settings, class T, class F, class Weights>
struct weight_loader
{
    F f;
    Weights weights;
    index_int entry;
    index_int n0;

    // Whether k..k+N-1 of a column are loaded as a run: every weight tensor
    // stores them adjacent or broadcast along the innermost k axis and they
    // stay inside it
    template <index_int N>
    static constexpr bool is_run()
    {
        return Settings::run_loads and Settings::run_len % N == 0;
    }

    // The last columns of a workgroup may reach past the matrix: the loads
    // stay inside and the store drops the columns
    __device__ auto position(index_int k, index_int j) const
    {
        const index_int col      = min(n0 + j, Settings::n - 1);
        const auto mk            = Settings::k_space.multi(k);
        const auto mn            = Settings::n_space.multi(col);
        constexpr index_int nk   = decltype(mk){}.size();
        constexpr index_int rank = 1 + nk + decltype(mn){}.size();
        array<index_int, rank> m{};
        m[0] = entry;
        for(index_int d = 0; d < nk; d++)
            m[1 + d] = mk[d];
        for(index_int d = 1 + nk; d < rank; d++)
            m[d] = mn[d - 1 - nk];
        return m;
    }

    __device__ T operator()(index_int k, index_int j) const
    {
        const auto m = position(k, j);
        return weights(
            [&](auto... ws) { return convert<T>(f(weight_element<Settings>(ws, m)...)); });
    }

    // The stored weights of k..k+N-1 of a column, to compute the elements
    // from once the loads of other runs are under way
    template <index_int N>
    __device__ auto load(index_int k, index_int j) const
    {
        const auto m = position(k, j);
        return weights([&](auto... ws) { return make_tuple(load_weights<N, Settings>(ws, m)...); });
    }

    template <index_int N, class Stored>
    __device__ vec<T, N> compute(Stored stored) const
    {
        return weights([&](auto... ws) {
            return stored(
                [&](auto... xs) { return convert<T>(f(expand_weights<N, Settings>(ws, xs)...)); });
        });
    }
};

// The longest run up to N the loader computes the chunks of k in: it must
// divide the chunk and what is left of k in the last chunk
template <class Loader, index_int Chunk, index_int K, index_int N>
constexpr index_int stage_run()
{
    if constexpr(N < 2)
        return 1;
    else if constexpr(Loader::template is_run<N>() and Chunk % N == 0 and (K % Chunk) % N == 0)
        return N;
    else
        return stage_run<Loader, Chunk, K, N / 2>();
}

// One row of the rows {rows, k axes...} along its k axes: the tensor, the
// index of the first element of the row and, for a contiguous row, its address
template <class A>
struct row_view
{
    A a;
    index_int first;
    decltype(A{}.data()) start;

    // Whether the elements of a row are stored in order without gaps
    static constexpr bool is_packed()
    {
        constexpr auto s = get_shape_c<A>{};
        index_int stride = 1;
        for(index_int d = s.lens.size() - 1; d > 0; d--)
        {
            if(s.strides[d] != stride)
                return false;
            stride *= s.lens[d];
        }
        return true;
    }

    __device__ auto& operator[](index_int k) const
    {
        if constexpr(is_packed())
            return start[k];
        else
            return a[first + k];
    }
};

} // namespace gather_dot_detail

// out[r] = a[r] . b[indices[r]] for rows a {rows, k axes...} and a table of
// matrices b {entries, k axes..., n axes...} that the weight function
// computes from the weight tensors: k and n here are the elements of their
// axes in row-major order. Every sum adds the bias of its entry when there
// is one, and the output function combines the columns of a group into one
// output column with its other inputs.
//
// Each workgroup takes one entry and one block of 16 columns of its matrix
// per wave. Its first wave lists the rows gathering the entry, so the entry
// is read once for all of them. The rows go in tiles of 16, a batch of tiles
// at a time: for every chunk of k the workgroup computes the columns of the
// entry into local memory, then each wave multiplies the tiles of rows with
// the columns of its block by matrix instructions, accumulating over the
// chunks. The columns of a group are in adjacent lanes, so the lane of the
// first one reads the sums of the others from their lanes. Without a matrix
// instruction every lane sums one row times one group of columns at a time.
template <class Settings, class Functions, class Out, class A, class Indices, class Inputs>
struct gather_dot_group
{
    using value_type = typename A::type;
    using acc_t      = typename acc_type<value_type>::type;
    using loader     = gather_dot_detail::weight_loader<Settings,
                                                        value_type,
                                                        typename Functions::weight_type,
                                                        typename Inputs::weights_type>;
    static constexpr index_int bs    = 16;
    static constexpr index_int rows  = Settings::rows;
    static constexpr index_int group = Settings::group;
    static constexpr index_int n_out = Settings::n / group;
    // the columns of the matrices and the output columns of a workgroup
    static constexpr index_int cols     = bs * Settings::col_blocks;
    static constexpr index_int out_cols = cols / group;
    static constexpr index_int chunk    = Settings::k_chunk;
    static_assert(bs % group == 0, "a block of columns must hold whole groups");
    // the tiles of rows that take the columns in local memory together
    static constexpr index_int tile_batch = 4;
    static constexpr auto grid =
        make_shape(index_ints<Settings::entries, ceil_div(Settings::n, cols)>{});
    // a slot of the list of rows and an output column of the workgroup
    static constexpr auto sums = make_shape(index_ints<ceil_div(rows, bs) * bs, out_cols>{});
    // a lane as (lane group, row or column of the block)
    static constexpr auto lane_groups = make_shape(index_ints<MIGRAPHX_WAVEFRONTSIZE / bs, bs>{});
    // a column of a block as (output column of the block, column of its group)
    static constexpr auto block_groups = make_shape(index_ints<bs / group, group>{});
    // the k of one matrix instruction on a block, 0 when there is none
    static constexpr index_int ki = matmul_block<value_type, bs, bs, Settings::k>() == bs
                                        ? matmul_detail::mma_k<acc_t, value_type, bs, Settings::k>()
                                        : 0;
    static constexpr bool
        has_mma = ki > 0 and
                  matmul_detail::frag_k<bs, ki>() > 1 and chunk % (ki == 0 ? 1 : ki) == 0;
    // the columns of a chunk in local memory, computed a run of k at a time
    static constexpr auto tile     = make_shape(index_ints<cols, chunk>{});
    static constexpr index_int run = gather_dot_detail::stage_run<loader, chunk, Settings::k, 32>();
    static constexpr auto runs     = make_shape(index_ints<cols, chunk / run>{});
    // the run of k a lane sums without a matrix instruction
    static constexpr index_int sum_run =
        gather_dot_detail::stage_run<loader, Settings::k, Settings::k, 8>();

    Functions fs;
    Out out;
    A a;
    Indices indices;
    Inputs inputs;
    index idx;
    index_int entry;
    // the first column of the matrices and the first output column
    index_int col0;
    index_int out0;
    index_int* list;
    index_int* count;
    value_type* columns;

    // The rows gathering the entry, in order, by the first wave: each lane
    // tests one row and places it after the matches of the lanes before it
    __device__ void find_rows() const
    {
        if(idx.wave() != 0)
            return;
        const index_int lane = idx.local_wave();
        index_int found      = 0;
        for(index_int r0 = 0; r0 < rows; r0 += MIGRAPHX_WAVEFRONTSIZE)
        {
            const index_int r = r0 + lane;
            bool match        = false;
            if(r < rows)
            {
                // a negative index counts from the end of the table
                const int64_t i = indices[r];
                match           = i == entry or i + Settings::entries == entry;
            }
            const uint64_t mask = __ballot(match ? 1 : 0);
            if(match)
                list[found + gather_dot_detail::popcount_below(mask, lane)] = r;
            found += popcount(mask);
        }
        // the lanes of a partial tile read a listed row and drop the result
        if(lane < bs)
            list[found + lane] = list[0];
        if(lane == 0)
            *count = found;
    }

    template <class View>
    __device__ static auto row(View x, index_int r)
    {
        MIGRAPHX_ASSERT(r < rows);
        const index_int first = r * Settings::k;
        return gather_dot_detail::row_view<View>{x, first, &x[first]};
    }

    // N elements of a row along k
    template <index_int N, class Row>
    __device__ static auto load_row(Row r, index_int k)
    {
        if constexpr(Row::is_packed())
        {
            return gather_dot_detail::load_run<N>(&r[k]);
        }
        else
        {
            return vec_generate<N>([&](auto i) { return r[k + i]; });
        }
    }

    // The matrix of the entry from its column n0
    __device__ loader matrix(index_int n0) const { return {fs.weight, inputs.weights, entry, n0}; }

    // The sum of a row and column n of the matrix with the bias of the entry
    // for the column, when there is one
    __device__ auto column(index_int n, acc_t x) const
    {
        return inputs.bias([&](auto... biases) {
            return (value_type(x) + ... + biases[entry * Settings::n + n]);
        });
    }

    // The output of a row and an output column from the sums of its group of
    // columns, which sum(g) gives
    template <class Sum>
    __device__ void store(index_int r, index_int col, Sum sum) const
    {
        const index_int i = r * n_out + col;
        out[i]            = sequence_c<group>([&](auto... gs) {
            return inputs.output([&](auto... es) {
                return fs.output(
                    make_array(column(col * group + index_int{gs}, sum(index_int{gs}))...),
                    i,
                    es...);
            });
        });
    }

    // The run i of the lane as (column, k), the k being len when the lane
    // has no such run
    __device__ auto run_place(index_int i, index_int len) const
    {
        const index_int r = idx.local + i * MIGRAPHX_NLOCAL;
        auto m            = runs.multi(r < runs.elements() ? r : 0);
        m[1]              = r < runs.elements() ? m[1] * run : len;
        return m;
    }

    // The columns of the workgroup for len of k from k0 into local memory.
    // A lane loads the stored weights of all its runs before it computes
    // any, so the loads are in flight together and it waits for memory once.
    __device__ void stage(index_int k0, index_int len) const
    {
        const auto b = matrix(col0);
        if constexpr(run > 1)
        {
            constexpr index_int per_lane = ceil_div(runs.elements(), MIGRAPHX_NLOCAL);
            array<decltype(b.template load<run>(0, 0)), per_lane> stored{};
            repeat_c<per_lane>([&](auto ic) {
                const index_int i = ic;
                const auto m      = run_place(i, len);
                if(m[1] < len)
                    stored[i] = b.template load<run>(k0 + m[1], m[0]);
            });
            repeat_c<per_lane>([&](auto ic) {
                const index_int i = ic;
                const auto m      = run_place(i, len);
                if(m[1] < len)
                    gather_dot_detail::store_run(columns + tile.index(m),
                                                 b.template compute<run>(stored[i]));
            });
        }
        else
        {
            idx.local_stride(tile.elements(), [&](auto i) {
                const auto m = tile.multi(i);
                if(m[1] < len)
                    columns[i] = b(k0 + m[1], m[0]);
            });
        }
    }

    // `Tiles` tiles of rows times the columns of the block of the wave over the
    // chunk of k from k0, whose columns are in local memory. The rows come
    // from global memory: those of a few steps are loaded together, so the
    // wave waits for them once.
    template <index_int Tiles, class Rows, class Acc>
    __device__ void mma_chunk(Acc& acc, const Rows& arow, index_int k0, index_int len) const
    {
        constexpr index_int kpl   = matmul_detail::frag_k<bs, ki>();
        constexpr index_int steps = 4;
        const auto lm             = lane_groups.multi(idx.local_wave());
        const index_int kl        = matmul_detail::frag_k0(kpl, lm[0]);
        const auto* column =
            columns + tile.index(make_array<index_int>(idx.wave() * bs + lm[1], 0));
        for(index_int k = kl; k < len; k += ki * steps)
        {
            array<array<vec<value_type, kpl>, Tiles>, steps> av{};
            repeat_c<steps>([&](auto si) {
                const index_int s = si;
                if(k + s * ki >= len)
                    return;
                repeat_c<Tiles>([&](auto j) {
                    const index_int t = j;
                    av[s][t]          = load_row<kpl>(arow[t], k0 + k + s * ki);
                });
            });
            repeat_c<steps>([&](auto si) {
                const index_int s = si;
                if(k + s * ki >= len)
                    return;
                const auto bv = gather_dot_detail::load_run<kpl>(column + k + s * ki);
                repeat_c<Tiles>([&](auto j) {
                    const index_int t = j;
                    acc[t] = matmul_detail::mma_intrinsic<acc_t, value_type, bs, bs, ki>{}(
                        av[s][t], bv, acc[t]);
                });
            });
        }
    }

    // The sums a lane holds, or those of the lane g columns after it
    template <class Frag, class G>
    __device__ static Frag lane_sums(Frag x, G)
    {
        constexpr index_int g = G{};
        if constexpr(g == 0)
            return x;
        else
            return readlane_xor<g>(x);
    }

    // The sums of the columns of a group, which adjacent lanes hold: valid
    // in the lane of the first column
    template <class Frag>
    __device__ static auto group_sums(Frag x)
    {
        return sequence_c<group>([&](auto... gs) { return make_array(lane_sums(x, gs)...); });
    }

    // The chunk for a run-time number of tiles: one loop for every number up
    // to Tiles, so the loop does not test the tiles at each step
    template <index_int Tiles, class Rows, class Acc>
    __device__ void
    mma_chunk(Acc& acc, const Rows& arow, index_int k0, index_int len, index_int tiles) const
    {
        if(tiles == Tiles)
            mma_chunk<Tiles>(acc, arow, k0, len);
        else if constexpr(Tiles > 1)
            mma_chunk<Tiles - 1>(acc, arow, k0, len, tiles);
    }

    // The tiles from slot r0 of the list times all the columns of the
    // workgroup, a chunk of k at a time
    __device__ void mma_batch(index_int r0, index_int tiles, index_int found) const
    {
        using frag           = matmul_detail::acc_frag<acc_t, bs>;
        const auto lm        = lane_groups.multi(idx.local_wave());
        const auto bg        = block_groups.multi(lm[1]);
        const index_int col  = out0 + idx.wave() * (bs / group) + bg[0];
        const bool has_block = col0 + idx.wave() * bs < Settings::n;
        // the lanes of a tile outside the batch are not read
        array<decltype(row(a, 0)), tile_batch> arow{};
        repeat_c<tile_batch>([&](auto j) {
            const index_int t = j;
            arow[t]           = row(a, list[r0 + (t < tiles ? t * bs : 0) + lm[1]]);
        });
        array<frag, tile_batch> acc{};
        for(index_int k0 = 0; k0 < Settings::k; k0 += chunk)
        {
            const index_int len = min(Settings::k - k0, chunk);
            stage(k0, len);
            __syncthreads();
            if(has_block)
                mma_chunk<tile_batch>(acc, arow, k0, len, tiles);
            __syncthreads();
        }
        repeat_c<tile_batch>([&](auto j) {
            const index_int t = j;
            if(t >= tiles)
                return;
            // read from the other lanes before the lanes take different paths
            const auto parts = group_sums(acc[t]);
            // the lane of the first column of a group stores it; a partial
            // tile and the last block drop what lies outside
            if(bg[1] != 0 or col >= n_out)
                return;
            for(index_int i = 0; i < vec_size<frag>(); i++)
            {
                const index_int slot = r0 + t * bs + matmul_detail::frag_row<bs>(i, lm[0]);
                if(slot < found)
                    store(list[slot], col, [&](index_int g) { return parts[g][i]; });
            }
        });
    }

    // The sum of a row times a column of the entry, a run of k at a time
    template <class Row>
    __device__ acc_t dot_column(Row arow, const loader& b, index_int n) const
    {
        acc_t result = 0;
        for(index_int k = 0; k < Settings::k; k += sum_run)
        {
            if constexpr(sum_run > 1)
            {
                result += vec_reduce(
                    convert<acc_t>(load_row<sum_run>(arow, k)) *
                        convert<acc_t>(b.template compute<sum_run>(b.template load<sum_run>(k, n))),
                    op::sum{});
            }
            else
            {
                result += acc_t(arow[k]) * acc_t(b(k, n));
            }
        }
        return result;
    }

    // Without a matrix instruction: every lane sums the listed rows times
    // the groups of columns of the workgroup one at a time
    __device__ void dot_rows(index_int found) const
    {
        const auto b = matrix(0);
        idx.local_stride(found * out_cols, [&](auto i) {
            const auto m        = sums.multi(i);
            const index_int col = out0 + m[1];
            if(col >= n_out)
                return;
            const index_int r = list[m[0]];
            array<acc_t, group> x{};
            repeat_c<group>([&](auto gi) {
                const index_int g = gi;
                x[g]              = dot_column(row(a, r), b, col * group + g);
            });
            store(r, col, [&](index_int g) { return x[g]; });
        });
    }

    __device__ void run_group() const
    {
        find_rows();
        __syncthreads();
        const index_int found = *count;
        if constexpr(has_mma)
        {
            for(index_int r0 = 0; r0 < found; r0 += bs * tile_batch)
            {
                mma_batch(r0, min(ceil_div(found - r0, bs), tile_batch), found);
            }
        }
        else
        {
            dot_rows(found);
        }
    }
};

// The functions computing the matrices and an output column from the columns
// of a group
template <class Weight, class Output>
struct gather_dot_functions
{
    using weight_type = Weight;
    Weight weight;
    Output output;
};

// The weight tensors, the bias if there is one and the other inputs of the
// output function, each packed
template <class Weights, class Bias, class Output>
struct gather_dot_inputs
{
    using weights_type = Weights;
    Weights weights;
    Bias bias;
    Output output;
};

template <class Settings, class Weight, class Output, class A, class Indices, class... Ts>
__device__ void gather_dot(Weight weight, Output output, A a, Indices indices, Ts... xs)
{
    // the weight tensors, the bias, the inputs of the output function, then
    // the output
    constexpr index_int nweights = Settings::weights;
    constexpr index_int nbias    = Settings::bias;
    static_assert(sizeof...(Ts) >= 1 + nweights + nbias, "missing inputs");
    constexpr index_int noutput = sizeof...(Ts) - 1 - nweights - nbias;
    auto weights                = gather_dot_detail::pack_range<0, nweights>(xs...);
    auto bias                   = gather_dot_detail::pack_range<nweights, nbias>(xs...);
    auto output_inputs          = gather_dot_detail::pack_range<nweights + nbias, noutput>(xs...);
    auto out                    = arg_c<sizeof...(Ts) - 1>()(xs...);
    using functions             = gather_dot_functions<Weight, Output>;
    using inputs = gather_dot_inputs<decltype(weights), decltype(bias), decltype(output_inputs)>;
    using group  = gather_dot_group<Settings, functions, decltype(out), A, Indices, inputs>;
    // the listed rows, padded to a whole tile
    __shared__ uninitialized_buffer<index_int, group::sums.lens[0] + group::bs, 16> list;
    __shared__ uninitialized_buffer<index_int, 1, 16> count;
    // 16-byte aligned so the loads of a lane's run of k merge
    __shared__ uninitialized_buffer<typename A::type, group::tile.elements(), 16> columns;
    auto idx     = make_index();
    const auto g = group::grid.multi(idx.group);
    group{functions{weight, output},
          out,
          a,
          indices,
          inputs{weights, bias, output_inputs},
          idx,
          g[0],
          g[1] * group::cols,
          g[1] * group::out_cols,
          list.data(),
          count.data(),
          columns.data()}
        .run_group();
}

} // namespace migraphx
#endif // MIGRAPHX_GUARD_KERNELS_GATHER_DOT_HPP
