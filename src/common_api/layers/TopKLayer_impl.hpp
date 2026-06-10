#ifndef TOPKLAYER_IMPL_HPP
#define TOPKLAYER_IMPL_HPP

#include "Layer_impl.hpp"

namespace nvinfer1
{
    class TopKLayer_impl : public ITopKLayer, public apiv::VTopKLayer, public Layer_impl
    {
    public:
        TopKLayer_impl() noexcept;
        // TopKLayer_impl(ITensor& input, int k, TopKOperation op, const std::shared_ptr<migraphx::program>& program) noexcept;
        ~TopKLayer_impl() override;

        // public API
        void setOperation(TopKOperation op) noexcept override;
        TopKOperation getOperation() const noexcept override;
        void setK(int32_t k) noexcept override;
        int32_t getK() const noexcept override;
        void setReduceAxes(uint32_t reduceAxes) noexcept override;
        uint32_t getReduceAxes() const noexcept override;

        void build() noexcept override;
    };
} // namespace nvinfer1 

#endif // TOPKLAYER_IMPL_HPP
