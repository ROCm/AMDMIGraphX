/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2022 Advanced Micro Devices, Inc. All rights reserved.
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
#ifndef MIGRAPHX_GUARD_MIGRAPHX_SUPPORTED_INSTRUCTIONS_HPP
#define MIGRAPHX_GUARD_MIGRAPHX_SUPPORTED_INSTRUCTIONS_HPP

#include <cassert>
#include <vector>

#include <migraphx/ranges.hpp>
#include <migraphx/instruction_ref.hpp>
// #include <migraphx/iota_iterator.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct supported_instructions;

template <class T>
struct supported_instructions_iterator_read
{
    static_assert(std::is_base_of<supported_instructions, T>::value,
                  "Template parameter must be a subclass of supported_instructions");
    T* view;
    auto operator()(std::size_t n) const
    {
        assert(view != nullptr);
        const auto& ranges  = view->get_ranges();
        const auto& metrics = view->get_metrics();
        return std::pair<iterator_range<instruction_ref>, float>{ranges[n], metrics[n]};
    }
};

struct supported_instructions
{
    using ranges  = std::vector<iterator_range<instruction_ref>>;
    using metrics = std::vector<float>;

    using iterator =
        basic_iota_iterator<supported_instructions_iterator_read<supported_instructions>,
                            std::size_t>;
    using const_iterator =
        basic_iota_iterator<supported_instructions_iterator_read<const supported_instructions>,
                            std::size_t>;

    void add(iterator_range<instruction_ref> range, float metric);

    const ranges& get_ranges() const&;
    ranges get_ranges() &&;
    const metrics& get_metrics() const&;
    metrics get_metrics() &&;

    iterator begin() { return {0, {this}}; }
    iterator end() { return {this->r.size(), {this}}; }

    const_iterator begin() const { return {0, {this}}; }
    const_iterator end() const { return {this->r.size(), {this}}; }

    private:
    ranges r;
    metrics m;
};

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
#endif // MIGRAPHX_GUARD_MIGRAPHX_SUPPORTED_INSTRUCTIONS_HPP
