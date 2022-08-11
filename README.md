# LMKBC_2022
prompt learning to query triples from pre-trained language models.

Frist of all, to find the prompts which elicit the pre-trained model better, we design prompts consider both a priori knowledge and synonym description from wikidata  as potential prompts. We applied the potential prompts to training data and keep several the  top-performing prompts based on F1-score (F1-score higher than 0.01 or Top 3). 

sh run_each_prompt.sh

Then, search the best threshold for each realtion.

run pipeline.py to generate the predicted results. 

python pipeline.py -i data/test.jsonl -o data/prediction.jsonl
