// Standalone test: GQA Attention (phi_silica dimensions)
//
// Grouped Query Attention for autoregressive decode with explicit dxgml ops:
//   Reshape → Transpose → Transpose → Slice → Slice → Slice
//     → RotaryEmbedding → RotaryEmbedding → GroupQueryAttention
//     → Transpose → Transpose
//
// The QKV tensor is first reshaped to expose the Q/K/V split, past KV
// tensors are transposed into GQA's expected layout, Q and K are rotary-
// embedded, then GQA produces output and updated KV cache tensors which
// are transposed back to storage layout.
//
// Dimensions (phi_silica, single-token decode):
//   qkv        : (1, 1, 9216)   f16  — merged QKV projection (B=1, S=1)
//   past_k/v   : (1, 32, 128, 96) f16 — KV cache (storage: B, heads, seq, dim)
//   pos_ids    : (1, 1)         i64  — position index
//   cos/sin    : (4096, 48)     f16  — rotary cache (max_seq, head_dim/2)
//   seqlens_k  : (1,)           i32  — number of cached KV tokens
//   output     : (1, 1, 3072)   f16  — attention output (num_heads * head_dim)
//   present_k/v: (1, 32, 129, 96) f16 — updated KV cache (appended current token)
//
// GQA params: num_heads=32, kv_num_heads=32, head_dim=96,
//             scale=1/sqrt(96)=0.10206207261...

module attributes {gpu.container_module} {
  dxgml.module @gqa_attention {
    dxgml.entry_point @gqa_attention(
        %qkv     : !dxgml.tensor<1x1x9216x!dxgml.float16>,
        %past_k  : !dxgml.tensor<1x32x128x96x!dxgml.float16>,
        %past_v  : !dxgml.tensor<1x32x128x96x!dxgml.float16>,
        %pos_ids : !dxgml.tensor<1x1x!dxgml.int64>,
        %seqlens : !dxgml.tensor<1x!dxgml.int32>
    ) -> (!dxgml.tensor<1x32x129x96x!dxgml.float16>,
          !dxgml.tensor<1x32x129x96x!dxgml.float16>)
    attributes {
      torch.onnx_meta.ir_version = 8 : si64,
      torch.onnx_meta.opset_version = 17 : si64,
      torch.onnx_meta.opset_versions = {com.microsoft = 1 : si64},
      torch.onnx_meta.producer_name = "pytorch",
      torch.onnx_meta.producer_version = "2.0.0"
    } {
      // Rotary embedding caches (max_seq=4096, rotary_dim=96, half=48)
      %cos_cache = dxgml_op.constant(#dxgml.constant_resource<attn.cos_cache : !dxgml.tensor<4096x48x!dxgml.float16>>)
      %sin_cache = dxgml_op.constant(#dxgml.constant_resource<attn.sin_cache : !dxgml.tensor<4096x48x!dxgml.float16>>)

      // Null pointer for optional GQA inputs not used in this configuration
      %null = dxgml_op.null_ptr

      // ---- Step 1: Reshape QKV to expose Q/K/V split ----
      //   (1, 1, 9216) -> (1, 1, 3, 3072)  where 3072 = 32 heads * 96 head_dim
      %qkv_split = dxgml_op.reshape(%qkv)
        : (!dxgml.tensor<1x1x9216x!dxgml.float16>)
        -> !dxgml.tensor<1x1x3x3072x!dxgml.float16>

      // ---- Steps 2-3: Transpose past KV to GQA-internal layout ----
      //   Storage layout : (1, 32, 128, 96) = (B, heads, seq, dim)
      //   GQA input layout: (1, 128, 32, 96) = (B, seq, heads, dim)  T[0,2,1,3]
      %past_k_t = dxgml_op.transpose(%past_k) {
        permutation = #dxgml.dense_integer_elements<[0, 2, 1, 3]> : !dxgml.tensor<4x!dxgml.int64>
      } : (!dxgml.tensor<1x32x128x96x!dxgml.float16>)
        -> !dxgml.tensor<1x128x32x96x!dxgml.float16>

      %past_v_t = dxgml_op.transpose(%past_v) {
        permutation = #dxgml.dense_integer_elements<[0, 2, 1, 3]> : !dxgml.tensor<4x!dxgml.int64>
      } : (!dxgml.tensor<1x32x128x96x!dxgml.float16>)
        -> !dxgml.tensor<1x128x32x96x!dxgml.float16>

      // ---- Steps 4-6: Slice Q, K, V from split QKV ----
      //   (1, 1, 3, 3072) sliced on axis=2: Q=[0:1], K=[1:2], V=[2:3]
      %q_raw = dxgml_op.slice(%qkv_split) {
        axes   = #dxgml.dense_integer_elements<[2]> : !dxgml.tensor<1x!dxgml.int64>,
        starts = #dxgml.dense_integer_elements<[0]> : !dxgml.tensor<1x!dxgml.int64>,
        ends   = #dxgml.dense_integer_elements<[1]> : !dxgml.tensor<1x!dxgml.int64>
      } : (!dxgml.tensor<1x1x3x3072x!dxgml.float16>)
        -> !dxgml.tensor<1x1x1x3072x!dxgml.float16>

      %k_raw = dxgml_op.slice(%qkv_split) {
        axes   = #dxgml.dense_integer_elements<[2]> : !dxgml.tensor<1x!dxgml.int64>,
        starts = #dxgml.dense_integer_elements<[1]> : !dxgml.tensor<1x!dxgml.int64>,
        ends   = #dxgml.dense_integer_elements<[2]> : !dxgml.tensor<1x!dxgml.int64>
      } : (!dxgml.tensor<1x1x3x3072x!dxgml.float16>)
        -> !dxgml.tensor<1x1x1x3072x!dxgml.float16>

      %v_raw = dxgml_op.slice(%qkv_split) {
        axes   = #dxgml.dense_integer_elements<[2]> : !dxgml.tensor<1x!dxgml.int64>,
        starts = #dxgml.dense_integer_elements<[2]> : !dxgml.tensor<1x!dxgml.int64>,
        ends   = #dxgml.dense_integer_elements<[3]> : !dxgml.tensor<1x!dxgml.int64>
      } : (!dxgml.tensor<1x1x3x3072x!dxgml.float16>)
        -> !dxgml.tensor<1x1x1x3072x!dxgml.float16>

      // Reshape V slice to merged format for GQA: (1,1,1,3072) -> (1,1,3072)
      %v = dxgml_op.reshape(%v_raw)
        : (!dxgml.tensor<1x1x1x3072x!dxgml.float16>)
        -> !dxgml.tensor<1x1x3072x!dxgml.float16>

      // ---- Steps 7-8: Rotary position embedding on Q and K ----
      //   Input format: (B, S, num_heads * head_dim) flat heads
      //   rotary_embedding_dim = 96 (full head_dim for phi_silica)

      // Reshape Q/K slices to flat: (1,1,1,3072) -> (1,1,3072)
      %q_flat = dxgml_op.reshape(%q_raw)
        : (!dxgml.tensor<1x1x1x3072x!dxgml.float16>)
        -> !dxgml.tensor<1x1x3072x!dxgml.float16>

      %k_flat = dxgml_op.reshape(%k_raw)
        : (!dxgml.tensor<1x1x1x3072x!dxgml.float16>)
        -> !dxgml.tensor<1x1x3072x!dxgml.float16>

      // Step 7: RotaryEmbedding on Q
      %q = dxgml_op.rotary_embedding(%q_flat, %cos_cache, %sin_cache, %pos_ids) {
        interleaved          = #dxgml.integer<0 : !dxgml.int64>,
        num_heads            = #dxgml.integer<32 : !dxgml.int64>,
        rotary_embedding_dim = #dxgml.integer<96 : !dxgml.int64>
      } : (!dxgml.tensor<1x1x3072x!dxgml.float16>,
           !dxgml.tensor<4096x48x!dxgml.float16>,
           !dxgml.tensor<4096x48x!dxgml.float16>,
           !dxgml.tensor<1x1x!dxgml.int64>)
        -> !dxgml.tensor<1x1x3072x!dxgml.float16>

      // Step 8: RotaryEmbedding on K
      %k = dxgml_op.rotary_embedding(%k_flat, %cos_cache, %sin_cache, %pos_ids) {
        interleaved          = #dxgml.integer<0 : !dxgml.int64>,
        num_heads            = #dxgml.integer<32 : !dxgml.int64>,
        rotary_embedding_dim = #dxgml.integer<96 : !dxgml.int64>
      } : (!dxgml.tensor<1x1x3072x!dxgml.float16>,
           !dxgml.tensor<4096x48x!dxgml.float16>,
           !dxgml.tensor<4096x48x!dxgml.float16>,
           !dxgml.tensor<1x1x!dxgml.int64>)
        -> !dxgml.tensor<1x1x3072x!dxgml.float16>

      // ---- Step 9: Group Query Attention ----
      //   Q: (1,1,3072), K: (1,1,3072), V: (1,1,3072)
      //   past_k_t: (1,128,32,96), past_v_t: (1,128,32,96)
      //   output: (1,1,3072), present_k: (1,129,32,96), present_v: (1,129,32,96)
      //   scale = 1/sqrt(96) ≈ 0.10206207261596576
      %output, %present_key, %present_value, %output_qk_matrix =
        dxgml_op.group_query_attention["GroupQueryAttention"](
            %q, %k, %v,
            %past_k_t, %past_v_t,
            %pos_ids, %seqlens,
            %null, %null, %null, %null, %null)
        {kv_num_heads = #dxgml.integer<32 : !dxgml.int64>,
         num_heads    = #dxgml.integer<32 : !dxgml.int64>,
         scale        = #dxgml.float<0.10206207261596576 : !dxgml.float32>}
        : (!dxgml.tensor<1x1x3072x!dxgml.float16>,
           !dxgml.tensor<1x1x3072x!dxgml.float16>,
           !dxgml.tensor<1x1x3072x!dxgml.float16>,
           !dxgml.tensor<1x128x32x96x!dxgml.float16>,
           !dxgml.tensor<1x128x32x96x!dxgml.float16>,
           !dxgml.tensor<1x1x!dxgml.int64>,
           !dxgml.tensor<1x!dxgml.int32>,
           !dxgml.null, !dxgml.null, !dxgml.null, !dxgml.null, !dxgml.null)
        -> (!dxgml.tensor<1x1x3072x!dxgml.float16>,
            !dxgml.tensor<1x129x32x96x!dxgml.float16>,
            !dxgml.tensor<1x129x32x96x!dxgml.float16>,
            !dxgml.null)

      // ---- Steps 10-11: Transpose present KV back to storage layout ----
      //   GQA output layout: (1, 129, 32, 96) = (B, seq, heads, dim)
      //   Storage layout   : (1, 32, 129, 96) = (B, heads, seq, dim)  T[0,2,1,3]
      %result_k = dxgml_op.transpose(%present_key) {
        permutation = #dxgml.dense_integer_elements<[0, 2, 1, 3]> : !dxgml.tensor<4x!dxgml.int64>
      } : (!dxgml.tensor<1x129x32x96x!dxgml.float16>)
        -> !dxgml.tensor<1x32x129x96x!dxgml.float16>

      %result_v = dxgml_op.transpose(%present_value) {
        permutation = #dxgml.dense_integer_elements<[0, 2, 1, 3]> : !dxgml.tensor<4x!dxgml.int64>
      } : (!dxgml.tensor<1x129x32x96x!dxgml.float16>)
        -> !dxgml.tensor<1x32x129x96x!dxgml.float16>

      dxgml.return %result_k, %result_v
        : !dxgml.tensor<1x32x129x96x!dxgml.float16>,
          !dxgml.tensor<1x32x129x96x!dxgml.float16>
    }
  }
  gpu.module @rock_gpu_module {
  }
}
