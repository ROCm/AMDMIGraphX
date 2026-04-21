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
#include <migraphx/compile_modes.hpp>
#include <migraphx/errors.hpp>
#include <migraphx/logger.hpp>
#include <algorithm>
#include <array>
#include <stdexcept>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

static constexpr std::array<compile_modes, 3> all_compile_modes = {
    compile_modes::EAGER, compile_modes::BALANCED, compile_modes::MAX};

compile_modes convert_to_compile_mode(uint8_t mode)
{
    // Exact match first
    auto it = std::find_if(all_compile_modes.begin(), all_compile_modes.end(), [&](compile_modes m) {
        return static_cast<uint8_t>(m) == mode;
    });
    if(it != all_compile_modes.end())
        return *it;

    // Clip out-of-range values and warn
    if(mode > 100)
    {
        log::warn() << "compile_mode value " << static_cast<int>(mode)
                    << " is out of range [0, 100], clamping to 100 (MAX)";
        return compile_modes::MAX;
    }

    // Snap to nearest defined mode using signed arithmetic to avoid uint8_t wrap-around
    auto nearest = std::min_element(
        all_compile_modes.begin(), all_compile_modes.end(), [&](compile_modes a, compile_modes b) {
            return std::abs(static_cast<int>(mode) - static_cast<int>(a)) <
                   std::abs(static_cast<int>(mode) - static_cast<int>(b));
        });
    log::warn() << "compile_mode value " << static_cast<int>(mode)
                << " does not map to a defined mode, snapping to nearest: "
                << static_cast<int>(*nearest);
    return *nearest;
}

compile_modes convert_to_compile_mode(const std::string& mode)
{
    if(mode == "eager")
        return compile_modes::EAGER;
    if(mode == "balanced")
        return compile_modes::BALANCED;
    if(mode == "max")
        return compile_modes::MAX;

    // Try parsing as integer
    try
    {
        std::size_t pos  = 0;
        int val          = std::stoi(mode, &pos);
        if(pos != mode.size())
            MIGRAPHX_THROW("Invalid compile_mode string: '" + mode + "'");
        if(val < 0 || val > 255)
            MIGRAPHX_THROW("compile_mode integer value out of uint8 range: " + mode);
        return convert_to_compile_mode(static_cast<uint8_t>(val));
    }
    catch(const std::invalid_argument&)
    {
        MIGRAPHX_THROW("Unknown compile_mode string: '" + mode +
                       "'. Valid values: 'eager', 'balanced', 'max', or an integer 0-100.");
    }
    catch(const std::out_of_range&)
    {
        MIGRAPHX_THROW("compile_mode integer value out of range: " + mode);
    }
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
