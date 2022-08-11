# validate the performance of each prompt each model on train data
# mkdir bert_large
# mkdir train_t0.1
# python run_each_prompt.py -i data/train.jsonl -o train_t0.1/ -m "bert-large-cased" -t 0.1
# for file in `ls train_t0.1`; 
# do 
#     echo ${file#*_} >> train_t0.1/${file%_*}.txt
#     python each_evaluate.py -g data/train.jsonl -p train_t0.1/$file >> train_t0.1/${file%_*}.txt
#     echo >> train_t0.1/${file%_*}.txt
# done
# mv train_t0.1 bert_large

### 
mkdir train_t0.5
python run_each_prompt.py -i data/train.jsonl -o train_t0.5/ -m "bert-large-cased" -t 0.5
for file in `ls train_t0.5`; 
do 
    echo ${file#*_} >> train_t0.5/${file%_*}.txt
    python each_evaluate.py -g data/train.jsonl -p train_t0.5/$file >> train_t0.5/${file%_*}.txt
    echo >> train_t0.5/${file%_*}.txt
done
mv train_t0.5 bert_large



# for file in `ls bert_large/train_results`; 
# do 
# #     echo $file >> train_performance.txt
#     if [[ ${file:0:2} -eq "Chemical" ]];then
#         echo >> $file
#         python each_evaluate.py -g data/train.jsonl -p bert_large/train_results/$file >> train_performance.txt
# #     echo >> train_performance.txt
#     fi
    
# done

# for file in `ls dev_results`; 
# do 
#     echo '--------' $file >> dev_performance.txt
#     python evaluate.py -g data/dev.jsonl -p dev_results/$file >> dev_performance.txt
#     echo >> dev_performance.txt
# done

# mv train_* bert_large
# mv dev_* bert_large

# # next model
# mkdir scibert
# mkdir train_results
# mkdir dev_results
# python test_each_prompt.py -i data/train.jsonl -o train_results/ -m "allenai/scibert_scivocab_uncased"
# python test_each_prompt.py -i data/dev.jsonl -o dev_results/ -m "allenai/scibert_scivocab_uncased"

# for file in `ls train_results`; 
# do 
#     echo '--------' $file >> train_performance.txt
#     python evaluate.py -g data/train.jsonl -p train_results/$file >> train_performance.txt
#     echo >> train_performance.txt
# done

# for file in `ls dev_results`; 
# do 
#     echo '--------' $file >> dev_performance.txt
#     python evaluate.py -g data/dev.jsonl -p dev_results/$file >> dev_performance.txt
#     echo >> dev_performance.txt
# done

# mv train_* scibert
# mv dev_* scibert

# # next model
# mkdir chemicalbert
# mkdir train_results
# mkdir dev_results
# python test_each_prompt.py -i data/train.jsonl -o train_results/ -m "recobo/chemical-bert-uncased"
# python test_each_prompt.py -i data/dev.jsonl -o dev_results/ -m "recobo/chemical-bert-uncased"

# for file in `ls train_results`; 
# do 
#     echo '--------' $file >> train_performance.txt
#     python evaluate.py -g data/train.jsonl -p train_results/$file >> train_performance.txt
#     echo >> train_performance.txt
# done

# for file in `ls dev_results`; 
# do 
#     echo '--------' $file >> dev_performance.txt
#     python evaluate.py -g data/dev.jsonl -p dev_results/$file >> dev_performance.txt
#     echo >> dev_performance.txt
# done

# mv train_* chemicalbert
# mv dev_* chemicalbert