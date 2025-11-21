# Prefix Caching Implementation Details

## Summary of Changes

This document describes the implementation changes made to support prefix caching in MIGraphX's LLM inference pipeline, specifically for `GroupQueryAttention` and KV cache management.

## Problem Statement

**Before:** MIGraphX only supported two modes:
1. **Prefill** (`sequence_length > 1`): Process all tokens, ignore past KV cache
2. **Decode** (`sequence_length == 1`): Process 1 token, append to past KV cache

This binary distinction prevented prefix caching, where:
- Some tokens are already cached (prefix)
- Multiple new tokens need to be processed
- New tokens must attend to cached prefix + themselves with causal masking

## Solution Overview

Enable a **third "hybrid" mode** that:
- Accepts `sequence_length > 1` (multiple new tokens)
- Respects non-zero `seqlens_k` values indicating cached prefix length
- Adjusts attention masking to account for absolute positions with cached prefix

## Modified Files

### 1. `/home/alturner/code/AMDMIGraphX/src/include/migraphx/op/concat_past_present.hpp`

**Location:** Lines 86-119 (update_cache function)

**Change:** Modified KV cache update logic to use `seqlens_k[batch_index]` for determining cached prefix length instead of relying solely on `sequence_length`.

**Before:**
```cpp
const bool is_prompt = sequence_length != 1;
const std::size_t past_seqlen = 
    sequence_length == 1 ? seqlens_k[batch_index] : past_buffer_sequence_length;
const std::size_t past_chunk_length = is_prompt ? 0 : past_seqlen * head_size;
```

**After:**
```cpp
// Use seqlens_k to determine actual past sequence length for this batch
const std::size_t past_seqlen = seqlens_k[batch_index];

// Only preserve past cache if past_seqlen is in valid range (0, past_buffer_seq_len)
const bool is_initial_prefill = (sequence_length > 1) && 
                               (past_seqlen == 0 || past_seqlen == past_buffer_sequence_length);
const std::size_t past_chunk_length = is_initial_prefill ? 0 : past_seqlen * head_size;
```

**Impact:** 
- ✓ Decode mode: `sequence_length=1`, `seqlens_k=N` → Preserves N cached tokens
- ✓ Initial prefill: `sequence_length>1`, `seqlens_k=0` → No preservation (fresh start)
- ✓ **Prefix-cached prefill (NEW)**: `sequence_length>1`, `0 < seqlens_k < max` → Preserves seqlens_k tokens

### 2. `/home/alturner/code/AMDMIGraphX/src/targets/gpu/kernels/include/migraphx/kernels/concat_past_present.hpp`

**Location:** Lines 58-92 (update_cache device function)

**Change:** Same logic as CPU version to maintain consistency.

**Before:**
```cpp
const bool is_prompt = sequence_length != 1;
const index_int past_seqlen = sequence_length == 1
                                ? static_cast<index_int>(seqlens_k[batch_index])
                                : past_buffer_sequence_length;
const index_int past_chunk_length = is_prompt ? 0 : past_seqlen * head_size;
```

**After:**
```cpp
// Use seqlens_k to determine actual past sequence length
const index_int past_seqlen = static_cast<index_int>(seqlens_k[batch_index]);

// Check if this is initial prefill vs prefix-cached prefill
const bool is_initial_prefill = (sequence_length > 1) && 
                               (past_seqlen == 0 || past_seqlen == past_buffer_sequence_length);
const index_int past_chunk_length = is_initial_prefill ? 0 : past_seqlen * head_size;
```

**Impact:** GPU kernels now handle prefix caching identically to CPU reference implementation.

### 3. `/home/alturner/code/AMDMIGraphX/src/onnx/parse_group_query_attention.cpp`

**Location:** Lines 202-217 (causal mask generation)

**Change:** Offset causal mask positions by `past_sl` to account for cached prefix tokens.

**Before:**
```cpp
if(sequence_length > 1)
{
    // Creates range [0, 1, 2, ...] for new tokens
    seq_range = info.add_literal(seq_range_s, seq_range_vec);
    seq_range = info.add_instruction(make_op("reshape", ...), seq_range);
    seq_range = info.add_instruction(make_op("multibroadcast", ...), seq_range);
    
    // Mask: bc_range > seq_range
    auto causal_mask = info.add_instruction(make_op("greater"), bc_range, seq_range);
    mul = info.add_instruction(make_op("where"), causal_mask, ninf, mul);
}
```

**After:**
```cpp
if(sequence_length > 1)
{
    // Creates range [0, 1, 2, ...] for new tokens
    seq_range = info.add_literal(seq_range_s, seq_range_vec);
    seq_range = info.add_instruction(make_op("reshape", ...), seq_range);
    
    // CRITICAL: Offset by past_sl for prefix caching
    auto bc_past_sl_for_range = info.add_instruction(
        make_op("multibroadcast", {{"out_lens", {sequence_length, 1}}}), past_sl);
    auto adjusted_seq_range = info.add_instruction(make_op("add"), seq_range, bc_past_sl_for_range);
    
    adjusted_seq_range = info.add_instruction(make_op("multibroadcast", ...), adjusted_seq_range);
    
    // Mask: bc_range > adjusted_seq_range (now accounts for cached prefix)
    auto causal_mask = info.add_instruction(make_op("greater"), bc_range, adjusted_seq_range);
    mul = info.add_instruction(make_op("where"), causal_mask, ninf, mul);
}
```

**Impact:** 

Without offset (old behavior):
- Token 0 (first new token) attends to positions [0] only → WRONG with prefix!

With offset (new behavior):
- If prefix length = 7, token 0 is actually at position 7
- Token 0 attends to positions [0-7] (all prefix + itself) → CORRECT!

**Visual Example:**

```
Cached prefix: [tok0, tok1, tok2]  (positions 0-2)
New tokens:    [tok3, tok4]        (positions 3-4)

OLD (broken):
  seq_range = [0, 1]
  Token3 mask: KV_pos > 0  → Only sees itself at pos 0 (WRONG!)
  Token4 mask: KV_pos > 1  → Sees pos 0-1 (WRONG!)

NEW (correct):
  seq_range = [0, 1]
  adjusted_seq_range = [0+3, 1+3] = [3, 4]
  Token3 mask: KV_pos > 3  → Sees pos 0-3 (prefix + itself) ✓
  Token4 mask: KV_pos > 4  → Sees pos 0-4 (all previous) ✓
```

### 4. `/home/alturner/code/AMDMIGraphX/src/onnx/parse_group_query_attention.cpp`

**Location:** Lines 223-239 (window attention logic)

**Change:** Use `adjusted_seq_range` instead of `seq_range` for window masking with prefix.

**Before:**
```cpp
if(local_window_size > 0)
{
    auto window_comp = info.add_instruction(
        make_op("add"), is_prompt ? seq_range : mask_comp, window_size_lit);
    // ...
}
```

**After:**
```cpp
if(local_window_size > 0)
{
    // Use adjusted_seq_range (with prefix offset) for prompts
    auto window_comp = info.add_instruction(
        make_op("add"), is_prompt ? adjusted_seq_range : mask_comp, window_size_lit);
    // ...
}
```

**Impact:** Window attention now correctly computes window boundaries relative to absolute positions, not relative positions within new tokens.

## Behavior Matrix

| Mode | `sequence_length` | `seqlens_k` | `past_chunk_length` | Attention Mask |
|------|-------------------|-------------|---------------------|----------------|
| **Initial Prefill** | > 1 | 0 | 0 | Causal within new tokens |
| **Initial Prefill** | > 1 | max_seq_len | 0 | Causal within new tokens |
| **Decode** | 1 | N | N × head_size | Attend to all N cached + self |
| **Prefix-Cached Prefill** | > 1 | N (0 < N < max) | N × head_size | Attend to N cached + causal within new |

## Key Insights

1. **Single source of truth**: `seqlens_k[batch_index]` now determines cached length in ALL modes
2. **Explicit check for initial prefill**: Prevents accidentally treating initial prefill as prefix-cached
3. **Position offset is critical**: Without offsetting `seq_range`, attention mask breaks for prefix caching
4. **Consistency across CPU/GPU**: Both implementations use identical logic

## Testing Recommendations

### Unit Tests

1. **KV Cache Preservation Test**
   ```python
   # Cache [A, B, C], process [D, E]
   # Verify cache becomes [A, B, C, D, E], not [D, E, 0, 0, 0]
   ```

2. **Attention Mask Test**
   ```python
   # Token D should attend to [A, B, C, D] but not E
   # Token E should attend to [A, B, C, D, E]
   ```

3. **Equivalence Test**
   ```python
   # Full prefill [A,B,C,D,E] should equal:
   # Step 1: Cache [A,B,C]
   # Step 2: Prefix-cached [D,E]
   ```

### Integration Tests

1. **LLM Generation Test**
   ```python
   # Same prefix, different suffixes
   # Verify outputs are correct and faster with caching
   ```

2. **Multi-Turn Chat Test**
   ```python
   # Accumulate conversation history as prefix
   # Each turn reuses growing prefix
   ```

## Backward Compatibility

✓ **Fully backward compatible**

- Existing decode-only workflows: No change (still uses `sequence_length=1`)
- Existing prefill workflows: No change (uses `seqlens_k=0` or `max_seq_len`)
- No API changes required for existing users

## Performance Expectations

Speedup formula (approximate):
```
speedup ≈ (prefix_length + new_length) / new_length
```

Examples:
- 100 prefix + 100 new = 2x speedup
- 500 prefix + 100 new = 6x speedup
- 1000 prefix + 50 new = 21x speedup

Actual speedup depends on:
- Memory bandwidth
- Compute utilization
- Attention mechanism (flash attention, etc.)

## Future Work

1. **Automatic prefix detection**: Analyze request patterns to automatically identify common prefixes
2. **LRU cache eviction**: Implement cache management for memory-constrained scenarios
3. **Fuzzy prefix matching**: Allow small differences in prefix (e.g., punctuation)
4. **Multi-user prefix sharing**: Share system prompts across users
5. **Streaming prefix updates**: Gradually build prefix cache during generation

## Related Code

Other files that interact with these changes:

- `src/fuse_attention.cpp`: Attention fusion patterns (no changes needed)
- `src/targets/gpu/fuse_mlir.cpp`: MLIR-based attention fusion (may need updates for optimal codegen)
- `src/onnx/parse_attention.cpp`: Legacy attention parsing (separate from GQA)

## References

- [ONNX GroupQueryAttention Spec](https://onnx.ai/onnx/operators/)
- [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180) (vLLM paper)
- [FlashAttention-2](https://arxiv.org/abs/2307.08691)

## Authors & Reviewers

- Implemented by: AI Assistant
- Date: November 20, 2025
- Review Status: Pending human review

## Appendix: Code Locations Summary

```
Modified Files:
├── src/
│   ├── include/migraphx/op/concat_past_present.hpp        [Lines 41-47, 86-119]
│   ├── onnx/parse_group_query_attention.cpp               [Lines 202-239]
│   └── targets/gpu/kernels/include/migraphx/kernels/
│       └── concat_past_present.hpp                        [Lines 58-92]
│
New Files:
├── docs/
│   ├── PREFIX_CACHING_GUIDE.md                            [User documentation]
│   └── PREFIX_CACHING_IMPLEMENTATION.md                   [This file]
└── test/
    └── prefix_caching_example.py                          [Demo script]
```

