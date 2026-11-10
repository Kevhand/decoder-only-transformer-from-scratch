# Bits per byte by kind of text (lower is better)

| Model | FineWeb-Edu (training domain) | Wikipedia (WikiText-103) | Books (PG-19) | General web (C4) |
|---|---|---|---|---|
| Ours (110M) | 1.0004 | 1.1257 | 1.4244 | 1.1832 |
| GPT-2 small (124M) | 1.0093 | 1.0155 | 1.4098 | 1.0968 |
| Ours (336M) | 0.8966 | 0.9980 | 1.2961 | 1.0741 |
| GPT-2 medium (355M) | 0.9262 | 0.9212 | 1.2489 | 1.0138 |
| GPT-2 large (774M) | 0.8801 | 0.8706 | 1.2159 | 0.9681 |

## Text used

- **FineWeb-Edu (training domain)**: 1,036 documents, 5.0 MB. HuggingFaceFW/fineweb-edu sample-10BT, after the documents used for training.
- **Wikipedia (WikiText-103)**: 62 documents, 1.2 MB. Salesforce/wikitext, wikitext-103-raw-v1 test split, detokenized as in lm-evaluation-harness.
- **Books (PG-19)**: 55 documents, 5.1 MB. PG-19 test split; from each book, a slice starting 20% in.
- **General web (C4)**: 2,434 documents, 5.0 MB. allenai/c4, English validation shard 0.

Each model uses its own tokenizer, so loss per token is not comparable across models; bits per byte is (total loss in bits / bytes of UTF-8 text).
