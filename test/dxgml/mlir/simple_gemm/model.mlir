// Simple DXML dialect example equivalent to migraphx_simple_example_f16.mlir.
//
// Both compute:  output = relu(A @ B + bias)
//   A    : 2x4 f16  (input matrix)
//   B    : 4x3 f16  (weight matrix)
//   bias : 2x3 f16  (bias, same shape as result)
//   out  : 2x3 f16
//
// To validate (parse only, no arch needed):
//   rocmlir-driver model.mlir
//
// To lower through the DXML -> MIGraphX -> TOSA -> Rock -> GPU path:
//   rocmlir-driver model.mlir --host-pipeline=dxgml ^
//       --kernel-pipeline=gpu --arch=gfx1201 -o model_gfx1201.gpu
//
// Full binary compilation:
//   rocmlir-driver model.mlir --host-pipeline=dxgml ^
//       --kernel-pipeline=full --arch=gfx1201 > model_gfx1201.bin
//
// MIGraphX equivalent (for comparison, same logical computation):
//   rocmlir-driver migraphx_simple_example_f16.mlir ^
//       --kernel-pipeline=gpu --arch=gfx1201 -o migraphx_gfx1201.gpu

module attributes {gpu.container_module} {
  dxgml.module @simple_gemm {
    dxgml.entry_point @simple_gemm_relu(
        %A    : !dxgml.tensor<2x4x!dxgml.float16>,
        %B    : !dxgml.tensor<4x3x!dxgml.float16>,
        %bias : !dxgml.tensor<2x3x!dxgml.float16>
    ) -> !dxgml.tensor<2x3x!dxgml.float16>
    attributes {
      torch.onnx_meta.ir_version = 6 : si64,
      torch.onnx_meta.opset_version = 12 : si64,
      torch.onnx_meta.producer_name = "pytorch",
      torch.onnx_meta.producer_version = "2.0.0"
    } {
      %null = dxgml_op.null_ptr

      // Step 1: Matrix multiplication  A(2x4) @ B(4x3) -> gemm(2x3)
      %gemm = dxgml_op.gemm (%A, %B, %null)
        : (!dxgml.tensor<2x4x!dxgml.float16>, !dxgml.tensor<4x3x!dxgml.float16>, !dxgml.null)
        -> !dxgml.tensor<2x3x!dxgml.float16>

      // Step 2: Add bias
      %biased = dxgml_op.add (%gemm, %bias)
        : (!dxgml.tensor<2x3x!dxgml.float16>, !dxgml.tensor<2x3x!dxgml.float16>)
        -> !dxgml.tensor<2x3x!dxgml.float16>

      // Step 3: ReLU activation
      %result = dxgml_op.relu (%biased)
        : (!dxgml.tensor<2x3x!dxgml.float16>) -> !dxgml.tensor<2x3x!dxgml.float16>

      dxgml.return %result : !dxgml.tensor<2x3x!dxgml.float16>
    }
  }
  gpu.module @rock_gpu_module {
  }
}
