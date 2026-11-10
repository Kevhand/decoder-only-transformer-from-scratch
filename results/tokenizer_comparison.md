# Tokenizer comparison

## Compression (higher bytes/token is better)

| Tokenizer | Vocab | fineweb_heldout bytes/token | code bytes/token | hindi bytes/token | numbers bytes/token |
|---|---|---|---|---|---|
| Ours (32k) | 32,768 | 4.58 | 2.07 | 2.14 | 2.68 |
| GPT-2 (50k) | 50,257 | 4.61 | 2.25 | 1.68 | 2.86 |
| LLaMA-2 (32k) | 32,000 | 3.99 | 3.60 | 2.42 | 1.63 |
| GPT-4 cl100k (100k) | 100,277 | 4.69 | 4.58 | 2.62 | 2.40 |

## Round-trip exactness

| Tokenizer | fineweb_heldout round-trip | code round-trip | hindi round-trip | numbers round-trip |
|---|---|---|---|---|
| Ours (32k) | 100.0% | 100.0% | 100.0% | 100.0% |
| GPT-2 (50k) | 100.0% | 100.0% | 100.0% | 100.0% |
| LLaMA-2 (32k) | 100.0% | 100.0% | 100.0% | 100.0% |
| GPT-4 cl100k (100k) | 100.0% | 100.0% | 100.0% | 100.0% |

## Vocabulary use and speed

| Tokenizer | Vocab used (FineWeb) | Encode MB/s (1 process) |
|---|---|---|
| Ours (32k) | 90.7% | 0.89 |
| GPT-2 (50k) | 74.1% | 14.13 |
| LLaMA-2 (32k) | 57.5% | 2.55 |
| GPT-4 cl100k (100k) | 40.9% | 5.53 |

## Example splits

```
"Jane Austen’s novels weren't only about romance; they're about independence."
  Ours (32k)              18 tokens  ['J', 'ane', ' Aust', 'en', '’', 's', ' novels', ' were', "n't", ' only', ' about', ' romance', ';', ' they', "'re", ' about', ' independence', '.']
  GPT-2 (50k)             18 tokens  ['Jane', ' Aust', 'en', '<e2 80>', '<99>', 's', ' novels', ' weren', "'t", ' only', ' about', ' romance', ';', ' they', "'re", ' about', ' independence', '.']
  LLaMA-2 (32k)           22 tokens  ['▁Jane', '▁Aust', 'en', '’', 's', '▁nov', 'els', '▁wer', 'en', "'", 't', '▁only', '▁about', '▁rom', 'ance', ';', '▁they', "'", 're', '▁about', '▁independence', '.']
  GPT-4 cl100k (100k)     16 tokens  ['Jane', ' Aust', 'en', '’s', ' novels', ' weren', "'t", ' only', ' about', ' romance', ';', ' they', "'re", ' about', ' independence', '.']

'In 2026, revenue rose 12.5% to $3,450,000 across 1,024 stores.'
  Ours (32k)              24 tokens  ['In', ' 20', '26', ',', ' revenue', ' rose', ' 12', '.', '5', '%', ' to', ' $', '3', ',', '450', ',', '000', ' across', ' 1', ',', '0', '24', ' stores', '.']
  GPT-2 (50k)             23 tokens  ['In', ' 20', '26', ',', ' revenue', ' rose', ' 12', '.', '5', '%', ' to', ' $', '3', ',', '450', ',', '000', ' across', ' 1', ',', '024', ' stores', '.']
  LLaMA-2 (32k)           36 tokens  ['▁In', '▁', '2', '0', '2', '6', ',', '▁re', 'venue', '▁rose', '▁', '1', '2', '.', '5', '%', '▁to', '▁$', '3', ',', '4', '5', '0', ',', '0', '0', '0', '▁across', '▁', '1', ',', '0', '2', '4', '▁stores', '.']
  GPT-4 cl100k (100k)     26 tokens  ['In', ' ', '202', '6', ',', ' revenue', ' rose', ' ', '12', '.', '5', '%', ' to', ' $', '3', ',', '450', ',', '000', ' across', ' ', '1', ',', '024', ' stores', '.']

'मशीन लर्निंग कंप्यूटर को डेटा से सीखना सिखाती है।'
  Ours (32k)              62 tokens  ['<e0 a4>', '<ae>', '<e0 a4>', '<b6>', '<e0 a5>', '<80>', 'न', '<20 e0 a4>', '<b2>', 'र', '्', 'न', 'ि', 'ं', '<e0 a4>', '<97>', ' क', 'ं', '<e0 a4>', '<aa>', '्', 'य', '<e0 a5>', '<82>', '<e0 a4>', '<9f>', 'र', ' क', '<e0 a5>', '<8b>', '<20 e0 a4>', '<a1>', 'े', '<e0 a4>', '<9f>', 'ा', '<20 e0 a4>', '<b8>', 'े', '<20 e0 a4>', '<b8>', '<e0 a5>', '<80>', '<e0 a4>', '<96>', 'न', 'ा', '<20 e0 a4>', '<b8>', 'ि', '<e0 a4>', '<96>', 'ा', 'त', '<e0 a5>', '<80>', '<20 e0 a4>', '<b9>', '<e0 a5>', '<88>', '<e0 a5>', '<a4>']
  GPT-2 (50k)             79 tokens  ['<e0 a4>', '<ae>', '<e0 a4>', '<b6>', '<e0 a5>', '<80>', '<e0 a4>', '<a8>', '<20 e0 a4>', '<b2>', '<e0 a4>', '<b0>', '<e0 a5>', '<8d>', '<e0 a4>', '<a8>', '<e0 a4>', '<bf>', '<e0 a4>', '<82>', '<e0 a4>', '<97>', '<20 e0 a4>', '<95>', '<e0 a4>', '<82>', '<e0 a4>', '<aa>', '<e0 a5>', '<8d>', '<e0 a4>', '<af>', '<e0 a5>', '<82>', '<e0 a4>', '<9f>', '<e0 a4>', '<b0>', '<20 e0 a4>', '<95>', '<e0 a5>', '<8b>', '<20 e0 a4>', '<a1>', '<e0 a5>', '<87>', '<e0 a4>', '<9f>', 'ा', '<20 e0 a4>', '<b8>', '<e0 a5>', '<87>', '<20 e0 a4>', '<b8>', '<e0 a5>', '<80>', '<e0 a4>', '<96>', '<e0 a4>', '<a8>', 'ा', '<20 e0 a4>', '<b8>', '<e0 a4>', '<bf>', '<e0 a4>', '<96>', 'ा', '<e0 a4>', '<a4>', '<e0 a5>', '<80>', '<20 e0 a4>', '<b9>', '<e0 a5>', '<88>', '<e0 a5>', '<a4>']
  LLaMA-2 (32k)           50 tokens  ['▁', 'म', 'श', 'ी', 'न', '▁', 'ल', 'र', '्', 'न', 'ि', 'ं', 'ग', '▁', 'क', 'ं', 'प', '्', 'य', 'ू', 'ट', 'र', '▁', 'क', 'ो', '▁', 'ड', 'े', 'ट', 'ा', '▁', 'स', 'े', '▁', 'स', 'ी', 'ख', 'न', 'ा', '▁', 'स', 'ि', 'ख', 'ा', 'त', 'ी', '▁', 'ह', 'ै', '।']
  GPT-4 cl100k (100k)     50 tokens  ['म', '<e0 a4>', '<b6>', 'ी', 'न', '<20 e0 a4>', '<b2>', 'र', '<e0 a5 8d e0 a4>', '<a8>', '<e0 a4 bf e0 a4>', '<82>', '<e0 a4>', '<97>', ' क', 'ं', 'प', '<e0 a5 8d e0 a4>', '<af>', '<e0 a5>', '<82>', '<e0 a4>', '<9f>', 'र', ' क', 'ो', '<20 e0 a4>', '<a1>', '<e0 a5 87 e0 a4>', '<9f>', 'ा', ' स', 'े', ' स', 'ी', '<e0 a4>', '<96>', 'न', 'ा', ' स', '<e0 a4 bf e0 a4>', '<96>', '<e0 a4 be e0 a4>', '<a4>', 'ी', ' ह', '<e0 a5>', '<88>', '<e0 a5>', '<a4>']

'def train(model, data):\n    for batch in data:\n        loss = model(batch)'
  Ours (32k)              31 tokens  ['def', ' train', '(', 'model', ',', ' data', '):', '\n', ' ', ' ', ' ', ' for', ' batch', ' in', ' data', ':', '\n', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' loss', ' =', ' model', '(', 'b', 'atch', ')']
  GPT-2 (50k)             30 tokens  ['def', ' train', '(', 'model', ',', ' data', '):', '\n', ' ', ' ', ' ', ' for', ' batch', ' in', ' data', ':', '\n', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' loss', ' =', ' model', '(', 'batch', ')']
  LLaMA-2 (32k)           22 tokens  ['▁def', '▁train', '(', 'model', ',', '▁data', '):', '<0x0A>', '▁▁▁', '▁for', '▁batch', '▁in', '▁data', ':', '<0x0A>', '▁▁▁▁▁▁▁', '▁loss', '▁=', '▁model', '(', 'batch', ')']
  GPT-4 cl100k (100k)     18 tokens  ['def', ' train', '(model', ',', ' data', '):\n', '   ', ' for', ' batch', ' in', ' data', ':\n', '       ', ' loss', ' =', ' model', '(batch', ')']
```
