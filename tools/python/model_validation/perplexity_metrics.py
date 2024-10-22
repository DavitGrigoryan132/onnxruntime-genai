from datasets import load_dataset
import numpy as np
import onnxruntime_genai as og
import torch

def get_wikitext2():
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    return testdata

def perplexity_eval(model_dir):
    model = og.Model(f'{model_dir}')
    tokenizer = og.Tokenizer(model)

    dataset = get_wikitext2()

    for batch in dataset:
        text = batch["text"]

        input_ids = tokenizer.encode_batch([text])

        params = og.GeneratorParams(model)
        params.set_model_input("input_ids", input_ids)
        generator = og.Generator(model, params)

        logits = generator.compute_logits()

        targets = np.roll(input_ids, -1, axis=1)

        probs = torch.softmax(torch.tensor(logits), dim=1).numpy()

        batch_size, seq_length = targets.shape
        target_probs = probs[np.arange(batch_size)[:, None], np.arange(seq_length), targets]

        log_probs = np.log(target_probs)

        total_log_probs += np.sum(log_probs)

        total_token_count += targets.size

    avg_log_prob = total_log_probs / total_token_count

    perplexity = np.exp(-avg_log_prob)

    print(f"The perplexity of {model_dir} is {perplexity}")
    return perplexity
