import argparse
import json
import logging

from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForMaskedLM, pipeline

import numpy as np
import pandas as pd
from itertools import chain

import utils
from utils import PromptSet
from utils import create_prompt
from file_io import read_lm_kbc_jsonl, read_lm_kbc_jsonl_to_df

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.corpus import stopwords
stopwords = stopwords.words('english')
REMOVE_STR = ['!',',','.','?','-s','-ly','</s>','s', "first", "up","1970s","1980s","1990s","him","her", \
              "himself","herself","it","n", "x", "-","x", "X," "e", "0", \
              "by", "up","nothing","scratch","unknown", "by","c","company","done", "over","One","home",\
              "Mt","office","Still","used","choice","[UNK]", ":", "", "A", "a", "data", "Li", "Mt","here", \
              "B", "C", "T","Y","here","them","next","response","reply","support","home","Trump","them",\
             "spoken", "I", "every","C\u00f3rdoba", "Z\u00fcrich", "Li\u00e8ge", "Le\u00f3n","W\u00fcrttemberg"]

for w in REMOVE_STR:
    stopwords.append(w)

REPLACE_SET ={'music': 'producer',
             'acting': 'actor',
             'teacher': 'professor',
             "water": "hydrogen" }


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -  %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def filter_func(output, threshold, raltion_type):
    filtered_list = []
    for seq in output:
        object_entity = seq["token_str"]
        prob = seq["score"]
        if (prob > threshold) and ((object_entity not in stopwords) and (object_entity[0:2] != "##")): # 保留了一小部分数据
            if (raltion_type == 'PersonProfession') and (object_entity in REPLACE_SET.keys()):
                object_entity = REPLACE_SET[object_entity]
            if (raltion_type == 'ChemicalCompoundElement') and (object_entity in REPLACE_SET.keys()):
                object_entity = REPLACE_SET[object_entity]
            filtered_list.append({object_entity: prob})
    return filtered_list

def one_relation_one_prompt_with_probability(input_rows, raltion_type: str, pipe, mask_token, prompt_pos=0, threshold=0.05):
    # one relation
    input_rows = list(filter(lambda row : row['Relation'] == raltion_type, input_rows))
    
    # Create prompts
    logger.info("Creating prompts for {}......".format(raltion_type))
    
    prompts = PromptSet([create_prompt(
                            subject_entity=row["SubjectEntity"],
                            relation=row["Relation"],
                            mask_token=mask_token,
                            prompt_pos=prompt_pos) for row in input_rows])
    # Run the model
    logger.info(f"Running the model...")
    outputs = []
    for out in tqdm(pipe(prompts, batch_size=8), total=len(prompts)):
        outputs.append(out)
    results = []
    for row, prompt, output in zip(input_rows, prompts, outputs):        
        result = {
            "SubjectEntity": row["SubjectEntity"],
            "Relation": row["Relation"],
            "ObjectEntities": filter_func(output, threshold=threshold, raltion_type=raltion_type),}
        results.append(result)

    return results

def cmpkey(entity, sorted_base='count_prob'):
    if sorted_base == 'count_prob':
        return (entity[1]["count"], entity[1]["possibility"])
    if sorted_base == 'prob_only':
        return (entity[1]["possibility"])

def select_candidats(results_list):
    temp_result = {}
    for item in results_list:
        key = (item["SubjectEntity"], item["Relation"])
        if not key in temp_result:
            temp_result[key] = {}
        entity_set = temp_result[key]
        for obj in item["ObjectEntities"]:
            for entity, possibility in obj.items():
                if not entity in entity_set:
                    entity_set[entity] = {"possibility": possibility, "count": 1}
                else:
                    entity_set[entity]["possibility"] = max(entity_set[entity]["possibility"], possibility)
                    entity_set[entity]["count"] += 1

    for key in temp_result:
        temp_result[key] = sorted(temp_result[key].items(), key=cmpkey, reverse=True)

    result = []

    for key, value in temp_result.items():
        result.append({
            "SubjectEntity": key[0],
            "Relation": key[1],
            "ObjectEntities": [i[0] for i in value[:5]]
        })
    return result



def run(args):
    # Load the model
    model_type = args.model
    logger.info(f"Loading the model \"{model_type}\"...")

    tokenizer = AutoTokenizer.from_pretrained(model_type)
    model = AutoModelForMaskedLM.from_pretrained(model_type)

    pipe = pipeline(
        task="fill-mask",
        model=model,
        tokenizer=tokenizer,
        top_k=args.top_k,
        device=args.gpu
    )

    mask_token = tokenizer.mask_token

    # Load the input file
    logger.info(f"Loading the input file \"{args.input}\"...")
    input_rows = read_lm_kbc_jsonl(args.input)
    logger.info(f"Loaded {len(input_rows):,} rows.")
    
    df = pd.DataFrame(input_rows)
    relations = df.Relation.value_counts().keys()
#     relation_threshold_dic = {'ChemicalCompoundElement': 0.07,
#                                 'CompanyParentOrganization': 0.60,
#                                 'CountryBordersWithCountry': 0.10,
#                                 'CountryOfficialLanguage': 0.20,
#                                 'PersonCauseOfDeath': 0.95,
#                                 'PersonEmployer': 0.01,
#                                 'PersonInstrument': 0.28,
#                                 'PersonLanguage': 0.30,
#                                 'PersonPlaceOfDeath': 0.45,
#                               'PersonProfession': 0.010, #0.045
#                               'RiverBasinsCountry': 0.150,
#                               'StateSharesBorderState': 0.050}    
    relation_threshold_dic = {'PersonEmployer': 0.06,
                             'PersonProfession': 0.01,
                             'PersonInstrument': 0.29,
                             'StateSharesBorderState': 0.04,
                             'ChemicalCompoundElement': 0.04,
                             'CountryOfficialLanguage': 0.36,
                             'PersonCauseOfDeath': 0.92,
                             'RiverBasinsCountry': 0.15,
                             'PersonLanguage': 0.29,
                             'CompanyParentOrganization': 0.54,
                             'CountryBordersWithCountry': 0.1,
                             'PersonPlaceOfDeath': 0.35}      

    # Create prompts
    logger.info(f"Creating prompts...")
    prompts = PromptSet([create_prompt(
                                        subject_entity=row["SubjectEntity"],
                                        relation=row["Relation"],
                                        mask_token=mask_token) 
                         for row in input_rows])
    
    final_results = []
    for relation in relations:
        # relation = 'PersonProfession'    
        prompts_num = create_prompt('prompts_num',relation=relation,mask_token=None)
        threshold = relation_threshold_dic[relation] if relation in relation_threshold_dic.keys() else 0.1
        results_list = []
        for pos in range(prompts_num): # each prompt
            results = one_relation_one_prompt_with_probability(input_rows,
                                                               raltion_type=relation, 
                                                               pipe=pipe,
                                                               mask_token=mask_token,
                                                               prompt_pos=pos, 
                                                               threshold = threshold)
            results_list.append(np.array(results))  
        results_list = list(np.array(results_list).ravel())
        relation_result = select_candidats(results_list)
        final_results.append(relation_result)
    final_results = list(chain.from_iterable(final_results))    

    fp = args.output
    with open(fp, "w") as f:
        for pred in final_results:
            f.write(json.dumps(pred) + "\n")
            

def main():
    parser = argparse.ArgumentParser(
        description="Probe a Language Model and "
                    "Run the Baseline Method on Prompt Outputs"
    )

    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="bert-large-cased",
        help="HuggingFace model name (default: bert-large-cased)",
    )
    
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        default="./data/train.jsonl",
        required=True,
        help="Input test file (required)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="./data/train.pred.jsonl",
        required=True,
        help="Output file (required)",
    )

    parser.add_argument(
        "-k",
        "--top_k",
        type=int,
        default=100,
        help="Top k prompt outputs (default: 100)",
    )

    parser.add_argument(
        "-g",
        "--gpu",
        type=int,
        default=0, # -1
        help="GPU ID, (default: -1, i.e., using CPU)"
    )

    args = parser.parse_args()

    run(args)


if __name__ == '__main__':
    main()
