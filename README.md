# Understanding Tokenizers

This project contains rudimentary implementation of two tokenizers, namely `BPE` and `SentencePiece`.
Purpose of this project is to implement basic components of these tokenizers without any external libraries.

## Possible key takeways:
Using this project, you can:
- Understand building blocks of tokenizers
- Play around with basic components of BPE tokenizers
- Check out how SentencePiece tokenizer can be trained

## BPE Tokenizer

A popular tokenizer that can be found in `tiktoken` library, most famously used by OpenAI in GPT-series models. The `bpe_tokenizer.ipynb` follows [Karpathy's Tokenizer lecture](https://www.youtube.com/watch?v=zduSFxRajkE).

## SentencePiece Tokenizer

Another poppular tokenizer that is used by famoulsy llama and Mistral models, and can be found in `SentencePiece` Library. Here `sentencepeace.py` provides a clean and, simple implementation of it. 