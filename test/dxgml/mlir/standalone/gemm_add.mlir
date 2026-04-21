// Standalone test: gemm -> elementwise add
//
// Computes: add(A @ B, bias)
//   A    : 4x8  f16  (input matrix)
//   B    : 8x16 f16  (weight matrix)
//   bias : 4x16 f16  (bias, same shape as gemm output)
//   out  : 4x16 f16

module attributes {gpu.container_module} {
  dxgml.module @gemm_add {
    dxgml.entry_point @gemm_add(
        %A    : !dxgml.tensor<4x8x!dxgml.float16>,
        %B    : !dxgml.tensor<8x16x!dxgml.float16>,
        %bias : !dxgml.tensor<4x16x!dxgml.float16>
    ) -> !dxgml.tensor<4x16x!dxgml.float16>
    attributes {
      torch.onnx_meta.ir_version = 8 : si64,
      torch.onnx_meta.opset_version = 17 : si64,
      torch.onnx_meta.producer_name = "pytorch",
      torch.onnx_meta.producer_version = "2.0.0"
    } {
      %null = dxgml_op.null_ptr

      // Step 1: Matrix multiply  A(4x8) @ B(8x16) -> gemm(4x16)
      %gemm = dxgml_op.gemm(%A, %B, %null)
        : (!dxgml.tensor<4x8x!dxgml.float16>,
           !dxgml.tensor<8x16x!dxgml.float16>,
           !dxgml.null)
        -> !dxgml.tensor<4x16x!dxgml.float16>

      // Step 2: Elementwise add (bias)
      %result = dxgml_op.add(%gemm, %bias)
        : (!dxgml.tensor<4x16x!dxgml.float16>,
           !dxgml.tensor<4x16x!dxgml.float16>)
        -> !dxgml.tensor<4x16x!dxgml.float16>

      dxgml.return %result : !dxgml.tensor<4x16x!dxgml.float16>
    }
  }
  gpu.module @rock_gpu_module {
  }
}
