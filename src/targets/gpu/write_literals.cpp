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
#include <migraphx/gpu/write_literals.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/gpu/hip.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/program.hpp>
#include <migraphx/env.hpp>
#include <migraphx/register_op.hpp>
#include <sys/stat.h>
#include <fstream>
#include <functional>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_COPY_LITERALS)


struct fetch_literal
{
    std::string id{};
    shape l_shape;
    std::string literal_file;
    std::string name() const { return "fetch_literal"; }

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.l_shape, "shape"), f(self.id, "id"), f(self.literal_file, "literal_file"));
    }

    shape compute_shape(const std::vector<shape>&) const { return l_shape; }
    argument compute(context&, const shape&, const std::vector<argument>&) const { 

        std::ifstream is;
        is.open(literal_file);
        if(not is.is_open())
        {
            MIGRAPHX_THROW("Could not open file: " + literal_file);
        }

        auto buffer = make_shared_array<char>(l_shape.bytes());

        if(not is.read(buffer.get(), l_shape.bytes()))
        {
            MIGRAPHX_THROW("Failed to read file: " + literal_file);
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





void write_literals::apply(module& m) const
{
    assert(ctx != nullptr);
    std::size_t n = 0;
    std::ofstream fs;
    std::string output_file_header;
    bool create_file = true;
    bool save_literals = strip_weights;
    
    if(output == "")
    {
        save_literals = false;
    }

    if(save_literals)
    {
        output_file_header = output + "_literals";
        if(mkdir(output_file_header.c_str(), 0777) == -1)
        {
            std::cout << "Weights already created, using folder already created.\n\n";
            create_file = false;
        }
    }

    for(auto ins : iterator_for(m))
    {
        if(ins->name() == "@literal")
        {
            if(enabled(MIGRAPHX_COPY_LITERALS{}))
            {
                if(save_literals)
                {
                    std::string id = m.name() + ":@literal:" + std::to_string(n);
                    // save current literal to it's own file (would need to change if literal.hpp is updated)
                    std::string buffer(ins->get_literal().data(), ins->get_shape().bytes());
                    std::hash<std::string> create_hash;
                    // can use better hashing method, this is just the easiest that worked
                    std::size_t hash_v = create_hash(buffer);
                    std::string output_file = output_file_header + "/" + std::to_string(hash_v) + ".mxr_literal";
                    if(create_file)
                    {
                        fs.open(output_file, std::ofstream::out | std::ofstream::trunc | std::ios::binary);
                        fs.write(buffer.data(), ins->get_shape().bytes());
                        fs.close();
                    }

                
                    auto pre = m.insert_instruction(ins, fetch_literal{id, ins->get_shape(), output_file});
                    auto alloc = m.insert_instruction(std::next(pre), hip_allocate{ins->get_literal().get_shape()});
                    m.replace_instruction(ins, hip_copy_to_gpu{}, pre, alloc);
                    n++;

                }
                else
                {
                    literal l  = ins->get_literal();
                    auto pre   = m.add_literal(l);
                    auto alloc = m.insert_instruction(std::next(pre), hip_allocate{l.get_shape()});
                    m.replace_instruction(ins, hip_copy_to_gpu{}, pre, alloc);
                }
            }
            else
            {
                if(save_literals)
                {
                    std::string id = m.name() + ":@literal:" + std::to_string(n);
                    std::string buffer(ins->get_literal().data(), ins->get_shape().bytes());
                    std::hash<std::string> create_hash;
                    // can use better hashing method, this is just the easiest that worked
                    std::size_t hash_v = create_hash(buffer);
                    std::string output_file = output_file_header + "/" + std::to_string(hash_v) + ".mxr_literal";
                    if(create_file)
                    {
                        fs.open(output_file, std::ofstream::out | std::ofstream::trunc | std::ios::binary);
                        fs.write(buffer.data(), ins->get_shape().bytes());
                        fs.close();
                    }

                    m.replace_instruction(ins, hip_copy_fetch_literal{ins->get_shape(), output_file, id});
                    // m.replace_instruction(ins, hip_copy_fetch_literal_test{ins->get_shape(), output_file, id, ins->get_literal()});
                    n++;
                }
                else
                {
                    std::string id = m.name() + ":@literal:" + std::to_string(n);
                    m.replace_instruction(ins, hip_copy_literal{ins->get_literal(), id});
                    n++;
                }
            }
        }
    }
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
