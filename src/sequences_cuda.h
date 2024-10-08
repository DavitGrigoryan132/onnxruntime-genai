#pragma once
namespace Generators {

// This class keeps track of sequences generated.
struct Sequences_Cuda {
  Sequences_Cuda(int batch_size, int beam_size, int max_length, cudaStream_t stream);

  // Returns a sequence of word IDs for a given beam index ( beam_index < batch_beam_size).
  RoamingArray<int32_t> GetSequence(size_t batch_beam_index);
  gpu_span<int32_t> GetSequences() { return sequences_; }
  gpu_span<int32_t> GetNextSequences() { return sequences_next_; }

  void AppendNextTokenToSequences(std::span<const int32_t> next_tokens);
  void AppendUserTokensToSequences(gpu_span<int32_t> user_tokens, int num_beams);

  void GetLastTokens(gpu_span<int32_t>& last_tokens);
  void RewindTo(size_t index);

  // Returns current sequence length.
  int GetSequenceLength() const;
  void AfterDeviceAppendedNextToken();

 private:
  cuda_unique_ptr<int32_t> sequences_buffer_;
  cudaStream_t stream_;

  // Two buffers of shape (batch_size, num_beams, max_seq_length) to store sequences.
  // At each time, there is only one buffer is active. The other one will be active in next token.
  // Each AppendNextTokenToSequences call will trigger a rotation of active buffer.
  gpu_span<int32_t> sequences_;
  gpu_span<int32_t> sequences_next_;  // This only exists for beam search, to allow for the easy reordering of sequences

  int batch_beam_size_;
  int max_length_;
  int current_length_;
};

}  // namespace Generators