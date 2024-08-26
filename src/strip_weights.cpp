/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2024 Advanced Micro Devices, Inc. All rights reserved.
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
#include <migraphx/strip_weights.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/program.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/register_op.hpp>
#include <msgpack.hpp>
#include <sys/stat.h>
#include <fstream>

// Define if want to test reading literals back into instructions

#define TEST_READ

// Currently run using:
// "MIGRAPHX_COPY_LITERALS=true ./build/bin/driver compile <model> --strip-weights -o <.mxr>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {


// Operation used to fetch literal arguments from file during runtime
struct fetch_literal
{
    size_t id;
    shape l_shape;
    std::string file_header;
    std::string name() const { return "fetch_literal"; }

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.l_shape, "shape"), f(self.id, "id"), f(self.file_header, "file_header"));
    }

    shape compute_shape(const std::vector<shape>&) const { return l_shape; }
    argument compute(context&, const shape&, const std::vector<argument>&) const { 

        // read literal argument from file
        std::ifstream is;
        is.open(file_header + "/literal_" + std::to_string(id), std::ios::binary);
        if(not is.is_open())
        {
            MIGRAPHX_THROW("Could not open file: " + file_header + "/literal_" + std::to_string(id));
        }

        auto buffer = make_shared_array<char>(l_shape.bytes());

        if(not is.read(buffer.get(), l_shape.bytes()))
        {
            MIGRAPHX_THROW("Failed to read file: " + file_header + "/literal_" + std::to_string(id));
        }
        is.close();
            
        return argument(l_shape, buffer); 
    
    }
    friend std::ostream& operator<<(std::ostream& os, const fetch_literal& x)
    {
        os << x.name() << "[id=" << x.id << "]";
        return os;
    }
};
MIGRAPHX_REGISTER_OP(fetch_literal);


// strips the weights (might eventually move to write_literals.cpp, but keep here now)
// writes each literal to its own file in literals folder
void strip_weights::apply(module& m) const
{
    

    #ifdef TEST_READ
        std::vector<argument> testing; // vector used to compare saved literal arguments to original
    #endif


    size_t n = 0;
    std::ofstream fs;
    std::string output_file_header = output.substr(0, output.find(".")) + "_weights";

    if(mkdir(output_file_header.c_str(), 0777) == -1){
       std::cout << "Weights already created, delete weights folder and try again.\n\n";
       return;
    }


    for(auto ins : iterator_for(m))
    {
        if(ins->name() == "@literal")
        {
            ins->debug_print();
            // save current literal to it's own file
            std::vector<char> cur_literal = std::vector<char>(ins->get_literal().data(), ins->get_literal().data() + ins->get_shape().bytes());
            auto buffer = make_shared_array<char>(cur_literal.begin(), cur_literal.end());

            std::string output_file = output_file_header + "/literal_" + std::to_string(n);
            std::cout << output_file << "\n";
            fs.open(output_file, std::ofstream::out | std::ofstream::trunc | std::ios::binary);
            fs.write(buffer.get(), ins->get_shape().bytes());
            fs.close();
            
            #ifdef TEST_READ
                testing.push_back(ins->get_literal().get_argument());
            #endif

            m.replace_instruction(ins, fetch_literal{n, ins->get_shape(), output_file_header});
            n++;
        }
    }

    #ifdef TEST_READ
        int i = 0;
        for(auto ins : iterator_for(m))
        {
            if(ins->name() == "fetch_literal")
            {
                std::ifstream is;
                is.open(output_file_header + "/literal_" + std::to_string(i), std::ios::binary);
                if(not is.is_open())
                {
                    // MIGRAPHX_THROW("Could not open file: " + file_header + "/literal_" + std::to_string(id));
                }

                auto buffer = make_shared_array<char>(ins->get_shape().bytes());

                if(not is.read(buffer.get(), ins->get_shape().bytes()))
                {
                    // MIGRAPHX_THROW("Failed to read file: " + file_header + "/literal_" + std::to_string(id));
                }
                is.close();

                argument test(ins->get_shape(), buffer);
                if(test.to_string() == testing[i].to_string()){
                    std::cout << "worked" + std::to_string(i) << "\n";
                }
                else {
                    std::cout << "\n\n" << test.to_string() << "\n\n";
                    std::cout << "\n\n" << testing[i].to_string() << "\n\n";
                }
                i++;
            }
        }
    #endif
    
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
