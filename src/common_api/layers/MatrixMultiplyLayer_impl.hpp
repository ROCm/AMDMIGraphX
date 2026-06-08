#ifndef MATRIX_MULTIPLY_LAYER_IMPL_HPP
#define MATRIX_MULTIPLY_LAYER_IMPL_HPP

#include <array>

#include "Layer_impl.hpp"

namespace nvinfer1
{
    class MatrixMultiplyLayer_impl : public IMatrixMultiplyLayer, public apiv::VMatrixMultiplyLayer, virtual public Layer_impl
    {
    public:
        MatrixMultiplyLayer_impl() noexcept;
        MatrixMultiplyLayer_impl(ITensor& input0, MatrixOperation op0, ITensor& input1, MatrixOperation op1, const std::shared_ptr<migraphx::program>& program) noexcept;
        ~MatrixMultiplyLayer_impl() override;

        // public API
        void setOperation(int32_t index, MatrixOperation op) noexcept override;
        MatrixOperation getOperation(int32_t index) const noexcept override;

        void build() noexcept override;

    private:
        std::array<MatrixOperation, 2> mOps;
    };
}

#endif // MATRIX_MULTIPLY_LAYER_IMPL_HPP
