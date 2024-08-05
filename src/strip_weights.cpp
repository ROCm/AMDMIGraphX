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
#include <fstream>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct fetch_literal
{
    size_t id;
    shape l_shape;
    argument l_argument;
    std::string name() const { return "fetch_literal"; }

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.l_shape, "shape"), f(self.id, "id"));
    }

    shape compute_shape(const std::vector<shape>&) const { return l_shape; }
    argument compute(const std::vector<argument>&) const { return l_argument; }
    friend std::ostream& operator<<(std::ostream& os, const fetch_literal& x)
    {
        os << x.name() << "[id=" << x.id << "]";
        return os;
    }
};
MIGRAPHX_REGISTER_OP(fetch_literal);

struct vector_stream
{
    std::vector<char> buffer{};
    vector_stream& write(const char* b, std::size_t n)
    {
        buffer.insert(buffer.end(), b, b + n);
        return *this;
    }
};

void strip_weights::apply(module& m) const
{
    std::vector<instruction_ref> ins_list;
    std::vector<std::string> vec;
    size_t n = 0;
    for(auto ins : iterator_for(m))
    {
        if(ins->name() == "@literal")
        {
            ins_list.push_back(ins);
            vec.push_back("@" + std::to_string(n) + ": " + ins->get_literal().to_string());
            m.replace_instruction(
                ins, fetch_literal{n, ins->get_shape(), ins->get_literal().get_argument()});
            n++;
        }
    }

    vector_stream vs;
    // msgpack::pack(vs, ins_list);
    msgpack::pack(vs, vec);

    auto* os = &std::cout;
    std::ofstream fs;
    fs.open("models/mnist.mxr_wgts", std::ios::binary);
    os = &fs;
    (*os).write(vs.buffer.data(), vs.buffer.size());
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
