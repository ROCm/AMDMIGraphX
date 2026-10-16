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
 */
#include <migraphx/gpu/write_literals.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/gpu/hip.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/program.hpp>
#include <migraphx/env.hpp>
#include <migraphx/register_op.hpp>
#include <migraphx/liveness.hpp>
#include <migraphx/algorithm.hpp>
#include <migraphx/functional.hpp>
#include <migraphx/ranges.hpp>
#include <algorithm>
#include <cstddef>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_COPY_LITERALS)

struct gpu_literal
{
    argument data{};
    bool host = false;

    argument gpu_data{};

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.data, "data"), f(self.host, "host"));
    }

    std::string name() const { return "gpu::literal"; }

    shape compute_shape(const std::vector<shape>&) const { return data.get_shape(); }

    argument compute(const shape&, const std::vector<argument>&) const { return gpu_data; }

    void finalize(context&, const shape&, const std::vector<shape>&)
    {
        if(host)
            gpu_data = register_on_gpu(data);
        else
            gpu_data = to_gpu(data);
    }

    friend std::ostream& operator<<(std::ostream& os, const gpu_literal& x)
    {
        os << x.name();
        if(x.host)
            os << ":host";
        else
            os << ":gpu";
        return os;
    }
};
MIGRAPHX_REGISTER_OP(gpu_literal);

static bool is_allocate(instruction_ref ins)
{
    return contains({"hip::allocate", "allocate"}, ins->name());
}

static std::size_t
estimate_scratch_size(const module& m, std::size_t overhead_percent, std::size_t alignment = 32)
{
    std::size_t scratch_size = 0;
    liveness(m, [&](instruction_ref ins, const auto& live_set) {
        std::size_t n =
            transform_accumulate(live_set.begin(),
                                 live_set.end(),
                                 ins->get_shape().bytes(),
                                 std::plus<>{},
                                 [&](instruction_ref i) -> std::size_t {
                                     if(not is_allocate(i))
                                         return 0;
                                     auto b = (i->get_shape().bytes() + alignment - 1) / alignment;
                                     return b * alignment;
                                 });
        scratch_size = std::max(scratch_size, n);
    });
    // Pad the estimate by overhead_percent since memory coloring is NP-hard and liveness is
    // incomplete without the scheduler, so we might need more space
    return scratch_size + scratch_size * overhead_percent / 100;
}

static std::size_t get_total_literals(const module& m)
{
    return transform_accumulate(m.begin(),
                                m.end(),
                                std::size_t{0},
                                std::plus<>{},
                                [&](const instruction& ins) -> std::size_t {
                                    // each code obj takes 2mb of gpu memory
                                    if(ins.name() == "gpu::code_object")
                                        return 1024 * 1024 * 2;
                                    if(not contains({"@literal", "@param"}, ins.name()))
                                        return 0;
                                    return ins.get_shape().bytes();
                                });
}

static std::size_t get_total_memory(const module& m, std::size_t scratch_overhead_percent)
{
    return get_total_literals(m) + estimate_scratch_size(m, scratch_overhead_percent);
}

static std::size_t get_available_memory()
{
    std::size_t free_memory = 0;
    auto status             = hipMemGetInfo(&free_memory, nullptr);
    if(status != hipSuccess)
        MIGRAPHX_THROW("Failed to get GPU memory info: " + std::string(hipGetErrorString(status)));
    return free_memory;
}

static std::size_t extra_needed(std::size_t available, std::size_t used)
{
    if(available > used)
        return 0;
    return used - available;
}

static std::size_t literal_bytes(instruction_ref ins) { return ins->get_shape().bytes(); }

// True when every use of lit in user is an input the kernel reads at index-selected positions
static bool is_indexed_use(instruction_ref lit, instruction_ref user)
{
    auto indexed =
        user->get_operator().attributes().get("indexed_inputs", std::vector<std::size_t>{});
    if(indexed.empty())
        return false;
    const auto& inputs = user->inputs();
    auto positions     = range(inputs.size());
    return std::all_of(positions.begin(), positions.end(), [&](std::size_t i) {
        return inputs[i] != lit or contains(indexed, i);
    });
}

// A literal only read a few elements at a time can stay in host memory without a staging copy
static bool is_zero_copy_literal(instruction_ref ins)
{
    const auto& outputs = ins->outputs();
    return not outputs.empty() and std::all_of(outputs.begin(), outputs.end(), [&](auto user) {
        return is_indexed_use(ins, user);
    });
}

template <class Pred>
static std::vector<instruction_ref> find_literals(const module& m, Pred pred)
{
    std::vector<instruction_ref> result;
    copy_if(iterator_for(m), std::back_inserter(result), [&](instruction_ref ins) {
        return ins->name() == "@literal" and pred(ins);
    });
    return result;
}

struct literal_plan
{
    // Host literals the kernels read directly
    std::unordered_set<instruction_ref> zero_copy{};
    // Host literals copied into scratch before each use
    std::unordered_set<instruction_ref> staged{};
};

// Moving a literal to the host frees its bytes, but a staged copy adds them back to scratch while
// it is live, so staging only nets the bytes of every staged literal except the largest. Zero-copy
// literals are freed in full and taken first. Of the rest, literals within the staging peak are
// taken largest-first for the fewest copies and any beyond it smallest-first to grow the peak as
// little as possible, then the smallest are dropped again while the need is still covered.
static literal_plan plan_literals(const module& m, std::size_t needed, std::size_t staging_peak)
{
    literal_plan plan;
    if(needed == 0)
        return plan;

    auto zero_copy = find_literals(m, &is_zero_copy_literal);
    std::stable_sort(zero_copy.begin(), zero_copy.end(), by(std::greater<>{}, &literal_bytes));
    auto zero_copy_last =
        std::find_if(zero_copy.begin(), zero_copy.end(), [&](instruction_ref ins) {
            if(needed == 0)
                return true;
            needed -= std::min(needed, literal_bytes(ins));
            return false;
        });
    plan.zero_copy.insert(zero_copy.begin(), zero_copy_last);
    if(needed == 0)
        return plan;

    auto staged =
        find_literals(m, [](instruction_ref ins) { return not is_zero_copy_literal(ins); });
    auto beyond = std::stable_partition(staged.begin(), staged.end(), [&](instruction_ref ins) {
        return literal_bytes(ins) <= staging_peak;
    });
    std::stable_sort(staged.begin(), beyond, by(std::greater<>{}, &literal_bytes));
    std::stable_sort(beyond, staged.end(), by(std::less<>{}, &literal_bytes));
    std::size_t sum     = 0;
    std::size_t largest = 0;
    auto last           = std::find_if(staged.begin(), staged.end(), [&](instruction_ref ins) {
        sum += literal_bytes(ins);
        largest = std::max(largest, literal_bytes(ins));
        return sum - largest >= needed;
    });
    if(last == staged.end())
    {
        // Short of memory either way, so stage everything like the overhead pad does
        plan.staged.insert(staged.begin(), staged.end());
        return plan;
    }
    ++last;
    // Drop the smallest taken literals while the rest still cover the need
    std::vector<instruction_ref> taken(staged.begin(), last);
    std::stable_sort(taken.begin(), taken.end(), by(std::less<>{}, &literal_bytes));
    std::size_t freed = sum - largest;
    auto first        = std::find_if(taken.begin(), taken.end(), [&](instruction_ref ins) {
        if(freed < needed + literal_bytes(ins))
            return true;
        freed -= literal_bytes(ins);
        return false;
    });
    plan.staged.insert(first, taken.end());
    return plan;
}

void write_literals::apply(module& m) const
{
    // Sort module to get better liveness analysis
    m.sort();
    std::size_t available = max_memory == 0 ? get_available_memory() : max_memory;
    std::size_t needed    = extra_needed(available, get_total_memory(m, scratch_overhead_percent));
    auto plan             = plan_literals(m, needed, needed * staging_peak_percent / 100);

    for(auto ins : iterator_for(m))
    {
        if(ins->name() != "@literal")
            continue;
        auto data = ins->get_literal().get_argument();
        if(enabled(MIGRAPHX_COPY_LITERALS{}) or contains(plan.staged, ins))
        {
            auto lit = m.insert_instruction(ins, gpu_literal{.data = data, .host = true});
            auto a   = m.insert_instruction(ins, hip_allocate{data.get_shape()});
            m.replace_instruction(ins, hip_copy{}, lit, a);
        }
        else
        {
            m.replace_instruction(ins,
                                  gpu_literal{.data = data, .host = contains(plan.zero_copy, ins)});
        }
    }
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
