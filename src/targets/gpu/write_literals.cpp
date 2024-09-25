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
#include <fstream>
#include <filesystem>
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

namespace fs = std::filesystem;
namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_COPY_LITERALS)

struct fetch_literal
{
    std::string id{};
    shape l_shape;
    std::string literal_file;
    argument data;
    std::string name() const { return "fetch_literal"; }

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(
            f(self.l_shape, "shape"), f(self.id, "id"), f(self.literal_file, "literal_file"));
    }

    shape compute_shape(const std::vector<shape>&) const { return l_shape; }
    argument compute(context&, const shape&, const std::vector<argument>&) const
    {
        return data;
    }

    void finalize(context&, const shape&, const std::vector<shape>&)
    {
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

        data = {l_shape, buffer};
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
    bool save_literals = strip_weights;

    if(output.empty())
        save_literals = false;

    if(save_literals)
    {
        output_file_header = output + "_literals";
        if(not fs::exists(output_file_header))
        {
            fs::create_directory(output_file_header);
        }
    }

    for(auto ins : iterator_for(m))
    {
        if(ins->name() == "@literal")
        {
            if(save_literals)
            {
                std::string id = m.name() + ":@literal:" + std::to_string(n);
                std::string buffer(ins->get_literal().data(), ins->get_shape().bytes());
                std::hash<std::string> create_hash;
                // can use better hashing method, this is just the easiest that worked
                std::size_t hash_v      = create_hash(buffer);
                std::string output_file = output_file_header + "/" + std::to_string(hash_v) +
                                          ".mxr_literal" +
                                          ins->get_literal().get_shape().type_string();
                // check and see if new file needs to be saved
                if(not fs::exists(output_file))
                {
                    fs.open(output_file,
                            std::ofstream::out | std::ofstream::trunc | std::ios::binary);
                    fs.write(buffer.data(), ins->get_shape().bytes());
                    fs.close();
                }

                if(enabled(MIGRAPHX_COPY_LITERALS{}))
                {
                    auto pre =
                        m.insert_instruction(ins, fetch_literal{id, ins->get_shape(), output_file, argument()});
                    auto alloc = m.insert_instruction(std::next(pre),
                                                      hip_allocate{ins->get_literal().get_shape()});
                    m.replace_instruction(ins, hip_copy_to_gpu{}, pre, alloc);
                }
                else
                {
                    m.replace_instruction(
                        ins, hip_copy_fetch_literal{ins->get_shape(), output_file, id});
                }
                n++;
            }

            // base strategy without stripping weights
            else
            {
                if(enabled(MIGRAPHX_COPY_LITERALS{}))
                {
                    literal l  = ins->get_literal();
                    auto pre   = m.add_literal(l);
                    auto alloc = m.insert_instruction(std::next(pre), hip_allocate{l.get_shape()});
                    m.replace_instruction(ins, hip_copy_to_gpu{}, pre, alloc);
                }
                else
                {
                    std::string id = m.name() + ":@literal:" + std::to_string(n);
                    m.replace_instruction(ins, hip_copy_literal{ins->get_literal(), id});
                }
                n++;
            }
        }
    }
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
