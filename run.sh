# # run pipeline on train data
# python pipeline.py -i data/train.jsonl -o data/train.pred.jsonl -m "bert-large-cased"
# python evaluate.py -g data/train.jsonl -p data/train.pred.jsonl

# # run pipeline one dev data
# python pipeline.py -i data/dev.jsonl -o data/dev.pred.jsonl -m "bert-large-cased"
# python evaluate.py -g data/dev.jsonl -p data/dev.pred.jsonl


# generate predicte results for test data
python pipeline.py -i data/test.jsonl -o data/prediction.jsonl -m "bert-large-cased"
