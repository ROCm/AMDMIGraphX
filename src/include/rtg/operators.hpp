#ifndef RTG_GUARD_OPERATORS_HPP
#define RTG_GUARD_OPERATORS_HPP

#include <array>
#include <rtg/operation.hpp>
#include <rtg/stringutils.hpp>
#include <rtg/streamutils.hpp>
#include <cmath>

namespace rtg {

struct check_shapes
{
    const std::vector<shape>* shapes;

    check_shapes(const std::vector<shape>& s) : shapes(&s) {}

    const check_shapes& has(std::size_t n) const
    {
        assert(shapes != nullptr);
        if(shapes->size() != n)
            RTG_THROW("Wrong number of arguments: expected " + std::to_string(n) + " but given " +
                      std::to_string(shapes->size()));
        return *this;
    }

    const check_shapes& only_dims(std::size_t n) const
    {
        assert(shapes != nullptr);
        if(!shapes->empty())
        {
            if(shapes->front().lens().size() != n)
                RTG_THROW("Only " + std::to_string(n) + "d supported");
        }
        return *this;
    }

    const check_shapes& same_shape() const
    {
        if(!this->same([](const shape& s) { return s; }))
            RTG_THROW("Shapes do not match");
        return *this;
    }

    const check_shapes& same_type() const
    {
        if(!this->same([](const shape& s) { return s.type(); }))
            RTG_THROW("Types do not match");
        return *this;
    }

    const check_shapes& same_dims() const
    {
        if(!this->same([](const shape& s) { return s.lens(); }))
            RTG_THROW("Dimensions do not match");
        return *this;
    }

    template <class F>
    bool same(F f) const
    {
        assert(shapes != nullptr);
        if(shapes->empty())
            return true;
        auto&& key = f(shapes->front());
        return this->all_of([&](const shape& s) { return f(s) == key; });
    }

    template <class Predicate>
    bool all_of(Predicate p) const
    {
        assert(shapes != nullptr);
        return std::all_of(shapes->begin(), shapes->end(), p);
    }
};

struct not_computable
{
    argument compute(shape, std::vector<argument>) const { RTG_THROW("not computable"); }
};

struct convolution
{
    std::array<std::size_t, 2> padding  = {{0, 0}};
    std::array<std::size_t, 2> stride   = {{1, 1}};
    std::array<std::size_t, 2> dilation = {{1, 1}};
    std::string name() const { return "convolution"; }
    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs}.has(2).same_type().same_dims().only_dims(4);

        const shape& input   = inputs.at(0);
        const shape& weights = inputs.at(1);
        auto t               = input.type();
        return {t,
                {
                    input.lens()[0],
                    weights.lens()[0],
                    std::size_t(std::max<std::ptrdiff_t>(
                        1,
                        (input.lens()[2] - (1 + dilation[0] * (weights.lens()[2] - 1)) +
                         2 * padding[0]) /
                                stride[0] +
                            1)),
                    std::size_t(std::max<std::ptrdiff_t>(
                        1,
                        (input.lens()[3] - (1 + dilation[1] * (weights.lens()[3] - 1)) +
                         2 * padding[1]) /
                                stride[1] +
                            1)),
                }};
    }

    argument compute(shape, std::vector<argument>) const { RTG_THROW("not computable"); }

    friend std::ostream& operator<<(std::ostream& os, const convolution& op)
    {
        os << op.name() << "[";
        os << "padding={" << stream_range(op.padding) << "}, ";
        os << "stride={" << stream_range(op.stride) << "}, ";
        os << "dilation={" << stream_range(op.dilation) << "}";
        os << "]";
        return os;
    }
};

struct pooling
{
    std::string mode;
    std::array<std::size_t, 2> padding = {{0, 0}};
    std::array<std::size_t, 2> stride  = {{1, 1}};
    std::array<std::size_t, 2> lengths = {{1, 1}};
    std::string name() const { return "pooling"; }
    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs}.has(1).only_dims(4);

        const shape& input = inputs.at(0);
        auto t             = input.type();
        return {t,
                {
                    input.lens()[0],
                    input.lens()[1],
                    std::size_t(std::max<std::ptrdiff_t>(
                        1,
                        std::ceil((input.lens()[3] + 2 * padding[0] - lengths[0]) /
                                  static_cast<float>(stride[0])) +
                            1)),
                    std::size_t(std::max<std::ptrdiff_t>(
                        1,
                        std::ceil((input.lens()[4] + 2 * padding[1] - lengths[1]) /
                                  static_cast<float>(stride[1])) +
                            1)),
                }};
    }

    argument compute(shape, std::vector<argument>) const { RTG_THROW("not computable"); }

    friend std::ostream& operator<<(std::ostream& os, const pooling& op)
    {
        os << op.name() << "[";
        os << "padding={" << stream_range(op.padding) << "}, ";
        os << "stride={" << stream_range(op.stride) << "}, ";
        os << "lengths={" << stream_range(op.lengths) << "}";
        os << "]";
        return os;
    }
};

struct activation
{
    std::string mode;
    std::string name() const { return "activation"; }
    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs}.has(1);
        return inputs.front();
    }

    argument compute(shape, std::vector<argument>) const { RTG_THROW("not computable"); }
    friend std::ostream& operator<<(std::ostream& os, const activation& op)
    {
        os << op.name() << ":" << op.mode;
        return os;
    }
};

struct reshape
{
    std::vector<int64_t> dims;
    std::string name() const { return "reshape"; }
    shape compute_shape(std::vector<shape> inputs) const
    {
        if(inputs.empty())
            RTG_THROW("Wrong number of arguments");
        auto&& idims = inputs.front().lens();
        std::vector<std::size_t> rdims(dims.begin(), dims.end());
        for(std::size_t i = 0; i < dims.size(); i++)
        {
            if(dims[i] == 0)
                rdims[i] = idims[i];
        }
        if(dims.back() == -1)
        {
            rdims.pop_back();
            std::copy(idims.begin() + rdims.size(), idims.end(), std::back_inserter(rdims));
        }
        return {inputs.front().type(), rdims};
    }

    argument compute(shape, std::vector<argument>) const { RTG_THROW("not computable"); }

    friend std::ostream& operator<<(std::ostream& os, const reshape& op)
    {
        os << op.name() << "[";
        os << "dims={" << stream_range(op.dims) << "}, ";
        os << "]";
        return os;
    }
};

struct gemm
{
    std::string name() const { return "gemm";}
    std::size_t lda = 1; 
    std::size_t ldb = 1; 
    std::size_t ldc = 1; 
    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs}.has(2).same_type().same_dims().only_dims(2);
        const shape& A = inputs.at(0); 
        const shape& B = inputs.at(1); 
        
        auto t         = A.type();
        if (A.lens()[1] != B.lens()[0])
            RTG_THROW("Inner dimensions do not match");
        return {t, {A.lens()[0], B.lens()[1]}};
    }
  
    argument compute(shape, std::vector<argument>) const { RTG_THROW("not computable"); }
  
    friend std::ostream& operator<<(std::ostream& os, const gemm& op) 
    {
        os << op.name() << "[";
        os << "]"; 
    }
};

} // namespace rtg

#endif
