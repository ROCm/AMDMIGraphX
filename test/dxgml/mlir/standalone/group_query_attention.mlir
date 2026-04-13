// Standalone test: Group Query Attention (GQA) broken down with individual dxgml ops
//
// GQA with: B=1, S=4, num_q_heads=2, num_kv_heads=1, head_dim=8, hidden=16
//   - 2 query heads share 1 KV head (GQA ratio = 2)
//   - Q projection: (S, hidden) @ (hidden, num_q_heads*head_dim) -> (S, 16)
//   - K projection: (S, hidden) @ (hidden, num_kv_heads*head_dim) -> (S, 8)
//   - V projection: (S, hidden) @ (hidden, num_kv_heads*head_dim) -> (S, 8)
//   - Reshape Q -> (S, num_q_heads, head_dim) = (4, 2, 8)
//   - Transpose Q -> (num_q_heads, S, head_dim) = (2, 4, 8)
//   - Reshape K -> (S, num_kv_heads, head_dim) = (4, 1, 8)
//   - Transpose K -> (num_kv_heads, S, head_dim) = (1, 4, 8)
//   - Reshape V -> (S, num_kv_heads, head_dim) = (4, 1, 8)
//   - Transpose V -> (num_kv_heads, S, head_dim) = (1, 4, 8)
//   - Broadcast K, V from (1, 4, 8) -> (2, 4, 8)  to match num_q_heads
//   - Transpose K for scores: (2, 4, 8) -> (2, 8, 4)
//   - Attention scores: Q(2,4,8) @ K^T(2,8,4) -> (2, 4, 4)
//   - Scale scores: multiply by 1/sqrt(head_dim) = 1/sqrt(8) ~ 0.354
//   - Softmax along last axis (axis=2)
//   - Weighted values: softmax_out(2,4,4) @ V(2,4,8) -> (2, 4, 8)
//   - Transpose: (2, 4, 8) -> (4, 2, 8)
//   - Reshape: (4, 2, 8) -> (4, 16)
//   - Output projection: (4, 16) @ (16, 16) -> (4, 16)
//
// input hidden_states: (4, 16) f16
// output:              (4, 16) f16

module attributes {gpu.container_module} {
  dxgml.module @group_query_attention {
    dxgml.entry_point @group_query_attention(
        %hidden_states : !dxgml.tensor<4x16x!dxgml.float16>
    ) -> !dxgml.tensor<4x16x!dxgml.float16>
    attributes {
      torch.onnx_meta.ir_version = 8 : si64,
      torch.onnx_meta.opset_version = 17 : si64,
      torch.onnx_meta.producer_name = "pytorch",
      torch.onnx_meta.producer_version = "2.0.0"
    } {
      // Projection weights
      %wq = dxgml_op.constant(#dxgml.constant_resource<attn.wq : !dxgml.tensor<16x16x!dxgml.float16>>)
      %wk = dxgml_op.constant(#dxgml.constant_resource<attn.wk : !dxgml.tensor<16x8x!dxgml.float16>>)
      %wv = dxgml_op.constant(#dxgml.constant_resource<attn.wv : !dxgml.tensor<16x8x!dxgml.float16>>)
      %wo = dxgml_op.constant(#dxgml.constant_resource<attn.wo : !dxgml.tensor<16x16x!dxgml.float16>>)

      // Attention scale: 1 / sqrt(head_dim=8) broadcast to scores shape (2, 4, 4)
      %scale = dxgml_op.constant(#dxgml.constant_resource<attn.scale : !dxgml.tensor<1x1x1x!dxgml.float16>>)

      // ---- Q, K, V projections ----

      // Q: (4, 16) @ (16, 16) -> (4, 16)
      %q_flat = dxgml_op.gemm(%hidden_states, %wq)
        : (!dxgml.tensor<4x16x!dxgml.float16>,
           !dxgml.tensor<16x16x!dxgml.float16>)
        -> !dxgml.tensor<4x16x!dxgml.float16>

      // K: (4, 16) @ (16, 8) -> (4, 8)
      %k_flat = dxgml_op.gemm(%hidden_states, %wk)
        : (!dxgml.tensor<4x16x!dxgml.float16>,
           !dxgml.tensor<16x8x!dxgml.float16>)
        -> !dxgml.tensor<4x8x!dxgml.float16>

      // V: (4, 16) @ (16, 8) -> (4, 8)
      %v_flat = dxgml_op.gemm(%hidden_states, %wv)
        : (!dxgml.tensor<4x16x!dxgml.float16>,
           !dxgml.tensor<16x8x!dxgml.float16>)
        -> !dxgml.tensor<4x8x!dxgml.float16>

      // ---- Reshape + Transpose to (num_heads, S, head_dim) ----

      // Q: (4, 16) -> (4, 2, 8)
      %q_3d = dxgml_op.reshape(%q_flat)
        : (!dxgml.tensor<4x16x!dxgml.float16>)
        -> !dxgml.tensor<4x2x8x!dxgml.float16>

      // Q: (4, 2, 8) -> (2, 4, 8)
      %q = dxgml_op.transpose(%q_3d) {
        permutation = #dxgml.dense_integer_elements<[1, 0, 2]> : !dxgml.tensor<3x!dxgml.int64>
      } : (!dxgml.tensor<4x2x8x!dxgml.float16>)
        -> !dxgml.tensor<2x4x8x!dxgml.float16>

      // K: (4, 8) -> (4, 1, 8)
      %k_3d = dxgml_op.reshape(%k_flat)
        : (!dxgml.tensor<4x8x!dxgml.float16>)
        -> !dxgml.tensor<4x1x8x!dxgml.float16>

      // K: (4, 1, 8) -> (1, 4, 8)
      %k_heads = dxgml_op.transpose(%k_3d) {
        permutation = #dxgml.dense_integer_elements<[1, 0, 2]> : !dxgml.tensor<3x!dxgml.int64>
      } : (!dxgml.tensor<4x1x8x!dxgml.float16>)
        -> !dxgml.tensor<1x4x8x!dxgml.float16>

      // V: (4, 8) -> (4, 1, 8)
      %v_3d = dxgml_op.reshape(%v_flat)
        : (!dxgml.tensor<4x8x!dxgml.float16>)
        -> !dxgml.tensor<4x1x8x!dxgml.float16>

      // V: (4, 1, 8) -> (1, 4, 8)
      %v_heads = dxgml_op.transpose(%v_3d) {
        permutation = #dxgml.dense_integer_elements<[1, 0, 2]> : !dxgml.tensor<3x!dxgml.int64>
      } : (!dxgml.tensor<4x1x8x!dxgml.float16>)
        -> !dxgml.tensor<1x4x8x!dxgml.float16>

      // ---- GQA broadcast: expand KV from (1,4,8) to (2,4,8) to match num_q_heads ----

      %k = dxgml_op.concat(%k_heads, %k_heads) {
        axis = #dxgml.integer<0 : !dxgml.int64>
      } : (!dxgml.tensor<1x4x8x!dxgml.float16>,
           !dxgml.tensor<1x4x8x!dxgml.float16>)
        -> !dxgml.tensor<2x4x8x!dxgml.float16>

      %v = dxgml_op.concat(%v_heads, %v_heads) {
        axis = #dxgml.integer<0 : !dxgml.int64>
      } : (!dxgml.tensor<1x4x8x!dxgml.float16>,
           !dxgml.tensor<1x4x8x!dxgml.float16>)
        -> !dxgml.tensor<2x4x8x!dxgml.float16>

      // ---- Scaled dot-product attention ----

      // Transpose K for matmul: (2, 4, 8) -> (2, 8, 4)
      %k_t = dxgml_op.transpose(%k) {
        permutation = #dxgml.dense_integer_elements<[0, 2, 1]> : !dxgml.tensor<3x!dxgml.int64>
      } : (!dxgml.tensor<2x4x8x!dxgml.float16>)
        -> !dxgml.tensor<2x8x4x!dxgml.float16>

      // Attention scores: Q(2,4,8) @ K^T(2,8,4) -> (2,4,4)
      %scores = dxgml_op.gemm(%q, %k_t)
        : (!dxgml.tensor<2x4x8x!dxgml.float16>,
           !dxgml.tensor<2x8x4x!dxgml.float16>)
        -> !dxgml.tensor<2x4x4x!dxgml.float16>

      // Scale: multiply scores by 1/sqrt(head_dim)
      %scores_scaled = dxgml_op.multiply(%scores, %scale)
        : (!dxgml.tensor<2x4x4x!dxgml.float16>,
           !dxgml.tensor<1x1x1x!dxgml.float16>)
        -> !dxgml.tensor<2x4x4x!dxgml.float16>

      // Softmax along last axis (key dimension)
      %attn_weights = dxgml_op.softmax(%scores_scaled) {
        axis = #dxgml.integer<2 : !dxgml.int64>
      } : (!dxgml.tensor<2x4x4x!dxgml.float16>)
        -> !dxgml.tensor<2x4x4x!dxgml.float16>

      // Weighted sum: attn_weights(2,4,4) @ V(2,4,8) -> (2,4,8)
      %attn_out = dxgml_op.gemm(%attn_weights, %v)
        : (!dxgml.tensor<2x4x4x!dxgml.float16>,
           !dxgml.tensor<2x4x8x!dxgml.float16>)
        -> !dxgml.tensor<2x4x8x!dxgml.float16>

      // ---- Merge heads ----

      // Transpose: (2, 4, 8) -> (4, 2, 8)
      %attn_t = dxgml_op.transpose(%attn_out) {
        permutation = #dxgml.dense_integer_elements<[1, 0, 2]> : !dxgml.tensor<3x!dxgml.int64>
      } : (!dxgml.tensor<2x4x8x!dxgml.float16>)
        -> !dxgml.tensor<4x2x8x!dxgml.float16>

      // Reshape: (4, 2, 8) -> (4, 16)
      %attn_merged = dxgml_op.reshape(%attn_t)
        : (!dxgml.tensor<4x2x8x!dxgml.float16>)
        -> !dxgml.tensor<4x16x!dxgml.float16>

      // ---- Output projection: (4, 16) @ (16, 16) -> (4, 16) ----
      %result = dxgml_op.gemm(%attn_merged, %wo)
        : (!dxgml.tensor<4x16x!dxgml.float16>,
           !dxgml.tensor<16x16x!dxgml.float16>)
        -> !dxgml.tensor<4x16x!dxgml.float16>

      dxgml.return %result : !dxgml.tensor<4x16x!dxgml.float16>
    }
  }
  gpu.module @rock_gpu_module {
  }
}
