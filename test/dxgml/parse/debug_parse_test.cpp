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

// Quick diagnostic: parse each mlir file and print any exception messages.
// NOT a production test — for development use only.

#include <migraphx/dxgml.hpp>
#include <migraphx/program.hpp>
#include <migraphx/instruction.hpp>
#include <dxgml_files.hpp>
#include <iostream>
#include <stdexcept>

int main()
{
    auto files = ::dxgml_files();

    for(const auto& kv : files)
    {
        std::cout << "\n=== Parsing: " << kv.first << " ===" << std::endl;
        try
        {
            auto prog = migraphx::parse_dxgml_string(std::string{kv.second});
            auto* mm  = prog.get_main_module();
            std::size_t n = std::distance(mm->begin(), mm->end());
            std::cout << "OK: " << n << " instructions" << std::endl;
            for(auto it = mm->begin(); it != mm->end(); ++it)
                std::cout << "  [" << std::distance(mm->begin(), it) << "] "
                          << it->name() << "  " << it->get_shape() << std::endl;
        }
        catch(const std::exception& e)
        {
            std::cerr << "EXCEPTION: " << e.what() << std::endl;
        }
        catch(...)
        {
            std::cerr << "UNKNOWN EXCEPTION" << std::endl;
        }
    }
    return 0;
}
