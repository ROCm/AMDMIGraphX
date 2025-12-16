# Paged Attention Examples

This directory contains examples demonstrating how to use MIGraphX with paged attention for LLM inference.

## Examples

| Example | Description |
|---------|-------------|
| `paged_attention_decode.cpp` | Autoregressive decode with separate K/V format |
| `paged_attention_prefill.cpp` | Prefill benchmark (auto-detects paged vs standard) |
| `verify_paged_attention.cpp` | Verify paged attention correctness by comparing outputs |

## Overview

Paged attention is a memory management technique for LLM inference that stores the KV cache in fixed-size blocks (pages) rather than contiguous memory. This provides several benefits:

- **Reduced memory fragmentation** - Memory is allocated in uniform blocks
- **Better memory utilization** - No need to pre-allocate maximum sequence length
- **Efficient batching** - Different sequences can share the block pool

## How Paged Attention Works

### Traditional KV Cache (Contiguous)
```
past_key_values: [batch, num_heads, max_seq_len, head_dim]
                 └─────────────────────────────────────────┘
                         One contiguous buffer per sequence
```

### Paged KV Cache
```
KV Cache Pool: [num_blocks, num_heads, block_size, head_dim]
               ├── Block 0 ──┤
               ├── Block 1 ──┤
               ├── Block 2 ──┤
               └── ...      ─┘

Block Table (per sequence): [seq_0: [0, 3, 7], seq_1: [1, 2, 5], ...]
                            Maps logical blocks to physical blocks
```

## Key Concepts

### Block Table

There are two formats for the block table:

#### Separate K/V Format (Default)
A 2D tensor `[batch_size, max_blocks_per_seq]` that maps logical block indices to physical block indices.

```cpp
// Example: sequence with 50 tokens, block_size=16
// Needs ceil(50/16) = 4 blocks
// Block table might be: [0, 5, 12, 3]  (physical blocks)
```

#### Combined K/V Format
A 3D tensor `[batch_size, 2, max_blocks_per_seq]` where dimension 1 separates K and V block indices.

```cpp
// Shape: {batch, 2, max_blocks}
//              ^-- 0 = K block indices, 1 = V block indices
// Example for batch=1, max_blocks=4:
// [[0, 5, 12, 3],   // K blocks
//  [0, 5, 12, 3]]   // V blocks (can be same or different)
```

This combined format allows K and V to potentially use different block mappings, which is useful for advanced cache management strategies.

### Slot Mapping
A 1D tensor `[num_new_tokens]` that specifies where new tokens should be written in the flattened cache.

```cpp
// Example: writing token at position 50
// block = 50 / 16 = 3, offset = 50 % 16 = 2
// physical_block = block_table[3] = 3
// slot = 3 * 16 + 2 = 50
slot_mapping = [50]
```

## Model Parameters

After compiling with paged attention, the model expects these additional parameters:

### Separate K/V Format (default)

| Parameter | Shape | Description |
|-----------|-------|-------------|
| `block_table` | `[batch, max_blocks_per_seq]` | Maps logical to physical blocks |
| `slot_mapping` | `[num_new_tokens]` | Target slots for scatter write |

### Combined K/V Format (`use_combined_kv = true`)

| Parameter | Shape | Description |
|-----------|-------|-------------|
| `block_table` | `[batch, 2, max_blocks_per_seq]` | K (idx 0) and V (idx 1) block indices |
| `slot_mapping` | `[num_new_tokens]` | Target slots for scatter write |

## Usage

### Building
```bash
mkdir build && cd build
cmake .. -DCMAKE_PREFIX_PATH=/path/to/migraphx
make
```

### Running Decode Example
```bash
./paged_attention_decode [model.onnx]
```

### Running Prefill Example
```bash
# Uses pre-compiled .mxr file (auto-detects paged vs standard)
./paged_attention_prefill <model.mxr> <prefill_length>

# Example:
./paged_attention_prefill qwen_paged.mxr 1500
./paged_attention_prefill qwen_standard.mxr 1500
```

### Example Output
```
Model: qwen_paged.mxr
Prefill length: 1500

Model type: Paged (combined KV)

=== Results ===
Time (avg): 45.23 ms
Time (min): 44.89 ms
Time (max): 46.12 ms
Throughput: 33172.5 tokens/sec
```

### Running Verification Script
```bash
# Compare paged attention output with standard KV cache
./verify_paged_attention <paged_model.mxr> <standard_model.mxr> <prefill_len>

# Example:
./verify_paged_attention qwen_paged.mxr qwen_standard.mxr 1500
```

### Verification Output
```
=== Paged Attention Verification ===

Paged model: qwen_paged.mxr
Standard model: qwen_standard.mxr
Prefill length: 1500

Running paged model...
Running standard model...

=== Comparing Outputs ===

Tolerance tests:
  Tight  (atol=1e-5, rtol=1e-4): FAIL (123 mismatches)
  Medium (atol=1e-4, rtol=1e-3): PASS (0 mismatches)
  Loose  (atol=1e-3, rtol=1e-2): PASS (0 mismatches)

Statistics:
  Max absolute difference: 5.234e-05
  Max relative difference: 8.123e-05
  Avg absolute difference: 1.234e-06

=== Result ===
PASS: Paged attention output matches standard KV cache
```

## Code Structure

### PagedKVCacheManager (Separate K/V)

Manages block allocation and slot mapping with separate K/V tensors:

```cpp
class PagedKVCacheManager {
    // Initialize with config
    PagedKVCacheManager(const PagedAttentionConfig& cfg);
    
    // Mark prompt tokens as processed
    void prefill(size_t batch_idx, size_t prompt_len);
    
    // Get slot indices for new tokens
    std::vector<int32_t> get_slot_mapping(size_t batch_idx, size_t num_tokens);
    
    // Advance position after tokens are written
    void advance(size_t batch_idx, size_t num_tokens);
    
    // Get flattened block table for model input: {batch * max_blocks}
    std::vector<int32_t> get_block_table_flat();
};
```

### PagedKVCacheManagerCombined (Combined K/V)

Manages block allocation with combined K/V format (dimension 2 for K/V separation):

```cpp
class PagedKVCacheManagerCombined {
    // Initialize with config
    PagedKVCacheManagerCombined(const PagedAttentionConfigCombined& cfg);
    
    // Get slot mapping for prefill (multiple tokens)
    std::vector<int32_t> get_slot_mapping_prefill(size_t batch_idx, size_t num_tokens);
    
    // Advance position after tokens are written
    void advance(size_t batch_idx, size_t num_tokens);
    
    // Get combined block table: {batch, 2, max_blocks}
    // Layout: [batch][kv_idx][block] where kv_idx: 0=K, 1=V
    std::vector<int32_t> get_block_table_combined();
};
```

### Decode Loop

```cpp
// 1. Prefill: process prompt
cache_manager.prefill(batch_idx, prompt_len);

// 2. Decode loop
for(int step = 0; step < max_new_tokens; ++step) {
    // Get current slot mapping
    auto slot_mapping = cache_manager.get_slot_mapping(batch_idx, 1);
    auto block_table = cache_manager.get_block_table_flat();
    
    // Set program parameters
    prog_params.add("slot_mapping", slot_mapping);
    prog_params.add("block_table", block_table);
    
    // Run inference
    auto outputs = prog.eval(prog_params);
    
    // Get next token from output
    int next_token = sample_token(outputs);
    
    // Advance position
    cache_manager.advance(batch_idx, 1);
    
    if(next_token == eos_token) break;
}
```

## Prefill vs Decode

| Phase | seq_len | slot_mapping size | Description |
|-------|---------|-------------------|-------------|
| Prefill | N | N | Write entire prompt to cache |
| Decode | 1 | 1 | Write one token at a time |

For best performance, use separate compiled models for prefill and decode:
- **Prefill model**: `seq_len = max_prompt_len` (e.g., 2048)
- **Decode model**: `seq_len = 1`

## Performance Considerations

1. **Block size**: 16 or 32 tokens typically works well
2. **Memory**: Total blocks = `batch_size * max_seq_len / block_size`
3. **GPU memory**: KV cache pool can be pre-allocated once and reused

## See Also

- [vLLM Paper](https://arxiv.org/abs/2309.06180) - Original paged attention concept
- MIGraphX documentation

