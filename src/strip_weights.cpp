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

class test_literal : public raw_data<test_literal>
{
    private:
    size_t l_id;
    shape l_shape;
    std::vector<char> l_data;

    public:
    MSGPACK_DEFINE(l_id, l_data);
    test_literal(size_t id = 0, shape s = shape{}, const char* data = nullptr)
        : l_id(id), l_shape(s)
    {
        l_data = std::vector<char>(data, data + s.bytes());
    }

    bool empty() const { return l_data.empty(); }

    const char* data() const { return &l_data[0]; }

    const shape& get_shape() const { return l_shape; }

    std::vector<test_literal> get_sub_objects() const { return {}; }

    void print(std::ostream& os) const
    {
        os << "test_literal[id=" << l_id << ", data={";
        if(l_data.size() < 10)
        {
            os << this;
        }
        else
        {
            os << " ... ";
        }
        os << ", size=" << l_data.size() << "}";
    }
};

void strip_weights::apply(module& m) const
{
    std::vector<test_literal> vec;
    size_t n = 0;
    for(auto ins : iterator_for(m))
    {
        if(ins->name() == "@literal")
        {
            vec.push_back(test_literal{n, ins->get_shape(), ins->get_literal().data()});
            m.replace_instruction(
                ins, fetch_literal{n, ins->get_shape(), ins->get_literal().get_argument()});
            n++;
        }
    }

    for(auto literal : vec)
    {
        literal.print(std::cout);
        std::cout << std::endl;
    }

    // Pack and write to file
    vector_stream vs;
    msgpack::pack(vs, vec);

    auto* os = &std::cout;
    std::ofstream fs;
    fs.open("models/mnist.mxr_weights", std::ios::binary);
    os = &fs;
    (*os).write(vs.buffer.data(), vs.buffer.size());

    // Read and unpack from file
    vector_stream vs2;
    std::ifstream is;
    is.open("models/mnist.mxr_weights", std::ios::binary | std::ios::ate);
    if(not is.is_open())
    {
        std::cout << "Failed to open file" << std::endl;
    }

    size_t nbytes = is.tellg();
    is.seekg(0, std::ios::beg);

    std::vector<char> buffer(nbytes, 0);

    if(not is.read(&buffer[0], nbytes))
    {
        std::cout << "Failed to read file" << std::endl;
    }

    msgpack::object_handle oh = msgpack::unpack(buffer.data(), buffer.size());
    msgpack::object obj       = oh.get();

    std::vector<test_literal> vec2;
    obj.convert(vec2);

    for(auto literal : vec2)
    {
        literal.print(std::cout);
        std::cout << std::endl;
    }
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
