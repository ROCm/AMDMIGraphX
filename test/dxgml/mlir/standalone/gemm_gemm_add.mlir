// Standalone test: gemm + gemm + elementwise add
//
// Computes: add(A @ W1, B @ W2)
//   Two independent linear projections of different inputs are summed.
//   This pattern appears in residual branches and dual-stream architectures.
//
//   A  : 4x8  f16  (input stream 1)
//   W1 : 8x16 f16  (weight for stream 1)
//   B  : 4x8  f16  (input stream 2)
//   W2 : 8x16 f16  (weight for stream 2)
//   out: 4x16 f16

module attributes {gpu.container_module} {
  dxgml.module @gemm_gemm_add {
    dxgml.entry_point @gemm_gemm_add(
        %A : !dxgml.tensor<4x8x!dxgml.float16>,
        %B : !dxgml.tensor<4x8x!dxgml.float16>
    ) -> !dxgml.tensor<4x16x!dxgml.float16>
    attributes {
      torch.onnx_meta.ir_version = 8 : si64,
      torch.onnx_meta.opset_version = 17 : si64,
      torch.onnx_meta.producer_name = "pytorch",
      torch.onnx_meta.producer_version = "2.0.0"
    } {
      %W1 = dxgml_op.constant(#dxgml.constant_resource<branch1.weight : !dxgml.tensor<8x16x!dxgml.float16>>)
      %W2 = dxgml_op.constant(#dxgml.constant_resource<branch2.weight : !dxgml.tensor<8x16x!dxgml.float16>>)

      // Step 1: Linear projection of stream 1  A(4x8) @ W1(8x16) -> (4x16)
      %out1 = dxgml_op.gemm(%A, %W1)
        : (!dxgml.tensor<4x8x!dxgml.float16>,
           !dxgml.tensor<8x16x!dxgml.float16>)
        -> !dxgml.tensor<4x16x!dxgml.float16>

      // Step 2: Linear projection of stream 2  B(4x8) @ W2(8x16) -> (4x16)
      %out2 = dxgml_op.gemm(%B, %W2)
        : (!dxgml.tensor<4x8x!dxgml.float16>,
           !dxgml.tensor<8x16x!dxgml.float16>)
        -> !dxgml.tensor<4x16x!dxgml.float16>

      // Step 3: Elementwise add — combine both branches
      %result = dxgml_op.add(%out1, %out2)
        : (!dxgml.tensor<4x16x!dxgml.float16>,
           !dxgml.tensor<4x16x!dxgml.float16>)
        -> !dxgml.tensor<4x16x!dxgml.float16>

      dxgml.return %result : !dxgml.tensor<4x16x!dxgml.float16>
    }
  }
  gpu.module @rock_gpu_module {
  }
}
