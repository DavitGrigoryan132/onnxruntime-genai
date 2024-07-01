// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "../generators.h"
#include "model.h"
#include "cache_manager.h"

#include <iostream>

namespace Generators {

namespace {

constexpr int32_t DefaultBlockSize = 16;

std::pair<std::unique_ptr<OrtValue>, std::unique_ptr<OrtValue>> AllocateLayerCache(
    const CacheOptions& options, Ort::Allocator& gpu_allocator) {
  const std::vector<int64_t> shape{options.num_blocks_, options.block_size_ * options.num_kv_heads_ * options.head_size_};

  return {OrtValue::CreateTensor(gpu_allocator, shape, options.dtype_),   // Key cache
          OrtValue::CreateTensor(gpu_allocator, shape, options.dtype_)};  // Value cache
}

void ComputeNumBlocks(CacheOptions& options) {
  if (options.num_blocks_ > 0) {
    return;
  }

  size_t dtype_size = 0;
  switch (options.dtype_) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      dtype_size = 4;
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
      dtype_size = 2;
      break;
    default:
      throw std::runtime_error("Invalid cache dtype: " + std::to_string(options.dtype_));
  }

  size_t free_bytes, total_bytes;
  CudaCheck() == cudaMemGetInfo(&free_bytes, &total_bytes);

  constexpr float memory_fragmentation_factor = 0.9f;
  constexpr size_t num_caches_per_layer = 2;  // 2 for key and value caches

  // Use the free memory to compute the number of blocks needed to achieve the given gpu_utilization_factor.
  options.num_blocks_ =
      (free_bytes *
       memory_fragmentation_factor *
       options.gpu_utilization_factor_) /
      (options.block_size_ *
       options.num_kv_heads_ *
       options.head_size_ *
       options.num_layers_ *
       dtype_size *
       num_caches_per_layer);
}

}  // namespace

CacheOptions::CacheOptions(const int32_t num_layers, const std::optional<int32_t>& block_size,
                           const int32_t num_kv_heads, const int32_t head_size,
                           const ONNXTensorElementDataType dtype,
                           const std::optional<int32_t>& num_blocks,
                           const std::optional<float>& gpu_utilization_factor)
    : num_layers_(num_layers),
      block_size_(block_size.value_or(DefaultBlockSize)),
      num_kv_heads_(num_kv_heads),
      head_size_(head_size),
      dtype_(dtype) {
  if (num_blocks.has_value() && gpu_utilization_factor.has_value()) {
    throw std::runtime_error("Both num_blocks and gpu_utilization_factor cannot be set at the same time.");
  } else if (num_blocks.has_value()) {
    num_blocks_ = *num_blocks;
  } else {
    constexpr float default_gpu_utilization_factor = 0.3f;
    gpu_utilization_factor_ = gpu_utilization_factor.value_or(default_gpu_utilization_factor);
  }
}

CacheManager::CacheManager(const CacheOptions& cache_options,
                           Ort::Allocator* cpu_allocator,
                           Ort::Allocator* gpu_allocator)
    : options_(cache_options),
      cpu_allocator_(cpu_allocator),
      gpu_allocator_(gpu_allocator) {
  ComputeNumBlocks(options_);
  for (int64_t i = 0; i < options_.num_layers_; ++i) {
    cache_.emplace_back(AllocateLayerCache(options_, *gpu_allocator_));
  }
  block_allocator_ = std::make_unique<BlockAllocator>(options_.block_size_, options_.num_blocks_);
}

bool CacheManager::CanAdd(size_t num_tokens) const {
  return block_allocator_->NumFreeBlocks() > block_allocator_->NumBlocksNeeded(num_tokens);
}

void CacheManager::Add(const std::vector<size_t>& sequence_ids, size_t num_tokens) {
  if (!CanAdd(num_tokens)) {
    throw std::runtime_error("Not enough free blocks available to serve the request.");
  }

  // Allocate blocks for the first sequence id in sequence_ids.
  Add(sequence_ids.front(), num_tokens);

  // Fork the blocks for the remaining sequence ids.
  for (size_t i = 1; i < sequence_ids.size(); ++i) {
    Fork(sequence_ids.front(), sequence_ids[i]);
  }
}

void CacheManager::Add(size_t sequence_id, size_t num_tokens) {
  if (!CanAdd(num_tokens)) {
    throw std::runtime_error("Not enough free blocks available to serve the request.");
  }

  // Create an empty block table for the sequence id.
  block_table_pool_.emplace_back(BlockTable{sequence_id, {}});
  auto block_table = --block_table_pool_.end();
  block_tables_[sequence_id] = block_table;

  // Allocate blocks for the sequence id.
  block_table->blocks = block_allocator_->AllocateBlocks(num_tokens);
}

bool CacheManager::CanAppendTokens(size_t sequence_id, size_t num_tokens) const {
  const auto block_table = block_tables_.find(sequence_id);
  if (block_table == block_tables_.end()) {
    return false;
  }

  const size_t num_slots_available = block_table->second->blocks.back()->NumEmptySlots() +
                                     block_allocator_->NumFreeBlocks() * options_.block_size_;

  // TODO: Implement copy on write functionality

  return num_slots_available >= num_tokens;
}

void CacheManager::AppendTokens(size_t sequence_id, size_t num_tokens) {
  if (!CanAppendTokens(sequence_id, num_tokens)) {
    throw std::runtime_error("Not enough free slots available to serve the request.");
  }

  auto block_table = block_tables_.find(sequence_id);
  if (block_table == block_tables_.end()) {
    throw std::runtime_error("Given sequence id " + std::to_string(sequence_id) + " is not found in the cache.");
  }

  if (block_table->second->blocks.back()->RefCount() > 1) {
    // The last block has multiple references. Copy-on-write is needed.
    auto src_block = block_table->second->blocks.back();
    auto dst_block = block_allocator_->AllocateBlock(src_block->NumOccupiedSlots(), src_block->PreviosBlock());
    src_block->DecrementRefCount();
    block_table->second->blocks.back() = dst_block;
    copy_on_writes_.emplace(dst_block->Id(), src_block->Id());
  }

  if (!block_table->second->blocks.back()->IsFull()) {
    for (size_t i = 0; i < std::min(num_tokens, block_table->second->blocks.back()->NumEmptySlots()); ++i) {
      block_table->second->blocks.back()->AddSlot();
      --num_tokens;
    }
  }

  std::shared_ptr<Block> previous_block = block_table->second->blocks.back();
  for (int32_t remaining_tokens = static_cast<int32_t>(num_tokens);
       remaining_tokens > 0;
       remaining_tokens -= options_.block_size_) {
    previous_block = block_allocator_->AllocateBlock(std::min(remaining_tokens, options_.block_size_), previous_block);
    block_table->second->blocks.push_back(previous_block);
  }
}

void CacheManager::Remove(size_t sequence_id) {
  auto block_table = block_tables_.find(sequence_id);
  if (block_table == block_tables_.end()) {
    return;
  }

  block_allocator_->Free(block_table->second->blocks);

  block_table_pool_.erase(block_table->second);
  block_tables_.erase(block_table);
}

void CacheManager::Fork(size_t src_sequence_id, size_t dst_sequence_id) {
  auto src_block_table = block_tables_.find(src_sequence_id);
  if (src_block_table == block_tables_.end()) {
    throw std::runtime_error("Given source sequence id " + std::to_string(src_sequence_id) + " is not found in the cache.");
  }

  if (block_tables_.find(dst_sequence_id) != block_tables_.end()) {
    throw std::runtime_error("Given destination sequence id " + std::to_string(dst_sequence_id) + " already exists in the cache.");
  }

  block_table_pool_.emplace_back(BlockTable{dst_sequence_id, {}});
  auto dst_block_table = --block_table_pool_.end();

  dst_block_table->blocks = block_allocator_->Fork(src_block_table->second->blocks);
}

std::pair<OrtValue*, OrtValue*> CacheManager::Cache(size_t layer_id) {
  return {cache_[layer_id].first.get(), cache_[layer_id].second.get()};
}

std::unique_ptr<OrtValue> CacheManager::BlockTables(const std::vector<size_t>& sequence_ids) const {
  size_t max_blocks = 0;
  for (const auto& sequence_id : sequence_ids) {
    auto block_table = block_tables_.find(sequence_id);
    if (block_table == block_tables_.end()) {
      throw std::runtime_error("Given sequence id " + std::to_string(sequence_id) + " is not found in the cache.");
    }
    max_blocks = std::max(max_blocks, block_table->second->blocks.size());
  }

  std::vector<int64_t> shape = {static_cast<int64_t>(sequence_ids.size()), static_cast<int64_t>(max_blocks)};
  auto block_tables_value = OrtValue::CreateTensor(*cpu_allocator_, shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32);
  auto* block_tables_data = block_tables_value->GetTensorMutableData<int32_t>();

  constexpr int32_t block_tables_pad_value = -1;

  for (size_t i = 0; i < sequence_ids.size(); ++i) {
    auto block_table = block_tables_.find(sequence_ids[i]);
    for (size_t j = 0; j < block_table->second->blocks.size(); ++j) {
      block_tables_data[i * max_blocks + j] = block_table->second->blocks[j]->Id();
    }
    for (size_t j = block_table->second->blocks.size(); j < max_blocks; ++j) {
      block_tables_data[i * max_blocks + j] = block_tables_pad_value;
    }
  }

  return block_tables_value;
}

std::unique_ptr<OrtValue> CacheManager::SlotMapping(const std::vector<size_t>& sequence_ids) const {
  size_t num_tokens = 0U;
  for (const auto& sequence_id : sequence_ids) {
    auto block_table = block_tables_.find(sequence_id);
    if (block_table == block_tables_.end()) {
      throw std::runtime_error("Given sequence id " + std::to_string(sequence_id) + " is not found in the cache.");
    }
    for (const auto& block : block_table->second->blocks) {
      num_tokens += block->NumOccupiedSlots();
    }
  }
  std::vector<int64_t> shape = {static_cast<int64_t>(num_tokens)};
  auto slot_mapping_value = OrtValue::CreateTensor(*cpu_allocator_, shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32);
  auto* slot_mapping_data = slot_mapping_value->GetTensorMutableData<int32_t>();
  for (size_t i = 0; i < sequence_ids.size(); ++i) {
    auto block_table = block_tables_.find(sequence_ids[i]);
    for (size_t i = 0, block_id = 0; block_id < block_table->second->blocks.size(); ++block_id) {
      const auto slot_ids = block_table->second->blocks[block_id]->SlotIds();
      for (size_t j = 0; j < slot_ids.size(); ++j, ++i) {
        slot_mapping_data[i] = slot_ids[j];
      }
    }
  }
  return slot_mapping_value;
}

}  // namespace Generators