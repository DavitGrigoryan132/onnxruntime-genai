#include "sequences.h"
#include <random>
#include "beam_search_scorer.h"
#pragma once

namespace Generators {

struct Search : LeakChecked<Search> {
  Search(const GeneratorParams& params) : params_{params.shared_from_this()} {}
  virtual ~Search() = default;

  virtual RoamingArray<int32_t> GetNextTokens() = 0;
  virtual RoamingArray<int32_t> GetNextIndices() = 0;
  virtual RoamingArray<int32_t> GetSequenceLengths() = 0;
  virtual int GetSequenceLength() const = 0;
  virtual RoamingArray<int32_t> GetSequence(size_t index) = 0;

  virtual void SetLogits(RoamingArray<float> logits) = 0;
  virtual bool IsDone() const = 0;

  virtual void SelectTop() = 0;
  virtual void SampleTopP(float /*p*/, float /*temperature*/) { assert(false); }
  virtual void SampleTopK(int /*k*/, float /*temperature*/) { assert(false); }
  virtual void SampleTopKTopP(int /*k*/, float /*p*/, float /*temperature*/) { assert(false); }

  // Scoring features
  virtual void ApplyMinLength(int min_length) = 0;
  virtual void ApplyRepetitionPenalty(float penalty) = 0;

  // Set user input tokens
  virtual void SetUserTokens(RoamingArray<int32_t> next_tokens) { assert(false); };
  // To be used for rewind
  virtual void RewindTo(size_t index) { assert(false); };
  // To be used for rewind
  virtual void DropLastTokens(size_t num_tokens) { assert(false); };

  std::shared_ptr<const GeneratorParams> params_;
};

struct Search_Cpu : Search {
  Search_Cpu(const GeneratorParams& params);

  int GetSequenceLength() const override;
  RoamingArray<int32_t> GetSequenceLengths() override { return sequence_lengths_; }
  RoamingArray<int32_t> GetSequence(size_t index) override { return sequences_.GetSequence(index); }

  bool IsDone() const override { return done_; }
  void SetLogits(RoamingArray<float> logits) override;

  void ApplyMinLength(int min_length) override;
  void ApplyRepetitionPenalty(float penalty) override;

  std::span<float> GetScores(int batch_beam_index) const;
  Sequences& GetSequences() { return sequences_; }

  cpu_span<int32_t> sequence_lengths_;  // shape (beam_size*batch_size)
  std::unique_ptr<int32_t[]> sequence_lengths_buffer_;

  cpu_span<int32_t> next_tokens_;  // shape (beam_size*batch_size)

  cpu_span<float> next_token_scores_;  // shape (beam_size*batch_size, vocab_size)

  Sequences sequences_;
  bool done_{};
};

struct GreedySearch_Cpu : Search_Cpu {
  GreedySearch_Cpu(const GeneratorParams& params);

  RoamingArray<int32_t> GetNextTokens() override;
  RoamingArray<int32_t> GetNextIndices() override { return cpu_span<int32_t>{}; }

  void SelectTop() override;
  void SampleTopK(int k, float temperature) override;
  void SampleTopP(float p, float temperature) override;
  void SampleTopKTopP(int /*k*/, float /*p*/, float /*temperature*/) override;

  // Used by continuous decoding search.
  void SetUserTokens(RoamingArray<int32_t> next_tokens) override;
  void RewindTo(size_t index) override;
  void DropLastTokens(size_t num_tokens) override;

 protected:
  void SetNextToken(size_t batch_id, int32_t token, bool check_eos = true);
  void AppendNextTokensToSequences();

 private:
  bool PadIfAlreadyEOS(size_t batch_id);

  std::unique_ptr<int32_t[]> next_tokens_buffer_;
  std::unique_ptr<int32_t[]> temp_topk_buffer_;

  std::span<bool> eos_seen_;  // shape (batch_size)
  std::unique_ptr<bool[]> eos_seen_buffer_;
  int not_done_count_{params_->search.batch_size};  // When zero, every batch entry is done (starts at batch_size_)

  std::mt19937 gen_;
};

struct BeamSearch_Cpu : Search_Cpu {
  BeamSearch_Cpu(const GeneratorParams& params);
  ~BeamSearch_Cpu();

  RoamingArray<int32_t> GetNextTokens() override;
  RoamingArray<int32_t> GetNextIndices() override;
  // In Beam Search there are batch_size * num_beams sequences. Index is batch_id * num_beams + beam_id... Easier to use the other version.
  RoamingArray<int32_t> GetSequence(size_t index) override;
  RoamingArray<int32_t> GetSequence(size_t batch_id, size_t beam_id);

  bool IsDone() const override;

  void SelectTop() override;

 private:
  void AppendNextTokensToSequences();
  void Finalize(size_t num_return_sequences);

  bool finalized_{};  // To avoid calling Finalize multiple times

  std::unique_ptr<BeamSearchScorer> beam_scorer_;
};

}  // namespace Generators