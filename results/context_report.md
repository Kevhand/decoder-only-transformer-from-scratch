# Using context: our model vs GPT-2

## LAMBADA (predict the last word from the passage)

| Model | Accuracy | Perplexity of the target word |
|---|---|---|
| Ours (110M) | 22.38% | 158.9 |
| GPT-2 small (124M) | 30.93% | 38.9 |
| GPT-2 medium (355M) | 41.41% | 17.8 |
| GPT-2 large (774M) | 47.31% | 13.0 |

## SQuAD reading comprehension (2 worked examples)

| Model | With passage: EM | With passage: F1 | Without passage: EM | Without passage: F1 | F1 gain from passage |
|---|---|---|---|---|---|
| Ours (110M) | 0.1% | 8.3% | 0.2% | 4.6% | +3.6 pts |
| GPT-2 small (124M) | 9.3% | 19.8% | 0.9% | 4.8% | +15.0 pts |
| GPT-2 medium (355M) | 24.0% | 36.4% | 2.3% | 8.0% | +28.4 pts |
| GPT-2 large (774M) | 32.3% | 45.7% | 2.2% | 8.2% | +37.5 pts |

Sample answers (with passage):

- **Ours (110M)**: Q: Which NFL team represented the AFC at Super Bowl 50? → *The AFC was the first team to win the title of Super Bowl 50.* (gold: Denver Broncos); Q: What day was the Super Bowl played on? → *The Super Bowl was played on the same day as the first Super Bowl, and the first Super Bowl* (gold: February 7, 2016); Q: What month, day and year did Super Bowl 50 take place?  → *The Super Bowl was held in the San Francisco Bay Area in the early 1900s. The first Super* (gold: February 7, 2016)
- **GPT-2 small (124M)**: Q: Which NFL team represented the AFC at Super Bowl 50? → *the New England Patriots.* (gold: Denver Broncos); Q: What day was the Super Bowl played on? → *the day of the Super Bowl* (gold: February 7, 2016); Q: What month, day and year did Super Bowl 50 take place?  → *January* (gold: February 7, 2016)
- **GPT-2 medium (355M)**: Q: Which NFL team represented the AFC at Super Bowl 50? → *Denver Broncos* (gold: Denver Broncos); Q: What day was the Super Bowl played on? → *Sunday, February 7, 2016* (gold: February 7, 2016); Q: What month, day and year did Super Bowl 50 take place?  → *January* (gold: February 7, 2016)
- **GPT-2 large (774M)**: Q: Which NFL team represented the AFC at Super Bowl 50? → *The Denver Broncos* (gold: Denver Broncos); Q: What day was the Super Bowl played on? → *February 7, 2016* (gold: February 7, 2016); Q: What month, day and year did Super Bowl 50 take place?  → *February 7, 2016* (gold: February 7, 2016)

## Closed book vs open book vs counterfactual passage

Share of the 25 fact questions where the model prefers the true answer (first two columns), or prefers the passage's false answer (last column). 50% is chance.

| Model | Knows it (no passage) | Reads it (true passage) | Follows a false passage |
|---|---|---|---|
| Ours (110M) | 80% | 100% | 92% |
| GPT-2 small (124M) | 68% | 96% | 96% |
| GPT-2 medium (355M) | 80% | 100% | 100% |
| GPT-2 large (774M) | 84% | 100% | 100% |
