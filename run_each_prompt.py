import argparse
import json
import logging

import numpy as np
import pandas as pd

from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForMaskedLM, pipeline

from file_io import read_lm_kbc_jsonl

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -  %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


class PromptSet(Dataset):
    def __init__(self, prompts):
        self.prompts = prompts

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, index) -> T_co:
        return self.prompts[index]


def create_prompt(subject_entity: str, relation: str, mask_token: str, prompt_pos=0) -> str:
    """
    Depending on the relation, we fix the prompt
    """

    prompt = mask_token

    if relation == "CountryBordersWithCountry":
        prompt1 = f"{subject_entity} shares border with {mask_token}."
        prompt2 = f"{subject_entity} next to {mask_token}."
        prompt3 = f"{subject_entity} borders {mask_token}."
        prompt4 = f"{subject_entity} bordered by {mask_token}."
        prompt5 = f"{subject_entity} adjacent to {mask_token}."
        prompt6 = f"{subject_entity} border {mask_token}."
        prompt_list = [prompt1, prompt2, prompt3, prompt4, prompt5, prompt6]
        
    elif relation == "CountryOfficialLanguage":
        prompt1 = f"The official language of {subject_entity} is {mask_token}."
        prompt2 = f"The language spoken in {subject_entity} is {mask_token}."
        prompt3 = f"The language spoken in {subject_entity} is {mask_token}."
        prompt4 = f"The language official in {subject_entity} is {mask_token}."
        prompt5 = f"{subject_entity} speaking {mask_token}."
        prompt_list = [prompt1, prompt2, prompt3, prompt4, prompt5]
        
    elif relation == "StateSharesBorderState":
        prompt1 = subject_entity + " next to {}.".format(mask_token)
        prompt2 = subject_entity + " borders {}.".format(mask_token)
        prompt3 = subject_entity + " bordered by {}.".format(mask_token)
        prompt4 = subject_entity + " adjacent to {}.".format(mask_token)
        prompt5 = subject_entity + " border {}".format(mask_token)
        prompt_list = [prompt1, prompt2, prompt3, prompt4, prompt5]
        
    elif relation == "RiverBasinsCountry":
        # ['water basin', ' catchment area', ' watershed', ' river basin', ' river basin district']
        prompt1 = f"{subject_entity} river basins in {mask_token}."
        prompt2 = f"{mask_token} is river basin district of {subject_entity}."
        prompt3 = f"{mask_token} is catchment area of {subject_entity}."
        prompt4 = f"the watershed in {mask_token} is {subject_entity}."
        
        prompt_list = [prompt1, prompt2, prompt3, prompt4]
        
    elif relation == "ChemicalCompoundElement": # [检查出来一些0.00] [2,3,4,5,6,7,8,9, 14, 16, 17, 19]
        prompt1 = f"{subject_entity} consists of {mask_token}, " f"which is an element."
        prompt2 = f"{subject_entity} is set of {mask_token}."
        prompt3 = f"{subject_entity} is part if {mask_token}."
        prompt4 = f"{subject_entity} includes part of {mask_token}."
        prompt5 = f"{subject_entity} holonym of {mask_token}."
        prompt6 = f"{subject_entity} have part of {mask_token}."
        prompt7 = f"{subject_entity} has ingredient of {mask_token}."
        prompt8 = f"{subject_entity} has component of {mask_token}."
        prompt9 = f"{subject_entity} has as part of {mask_token}."
        prompt10 = f"{subject_entity} created out of {mask_token}."
        prompt11 = f"{subject_entity} created from {mask_token}."
        prompt12 = f"{subject_entity} contain {mask_token}."
        prompt13 = f"{subject_entity} composed of {mask_token}."
        prompt14 = f"{subject_entity} assembled out of {mask_token}."
        prompt15 = f"{subject_entity} assembled from {mask_token}."
        prompt16 = f"{subject_entity} amalgamation of {mask_token}."
        prompt17 = f"{subject_entity} formed out of {mask_token}."
        prompt18 = f"{subject_entity} formed from {mask_token}."
        prompt19 = f"{subject_entity} has member of {mask_token}."
        prompt20 = f"{subject_entity} comprised of {mask_token}."

        prompt_list = [prompt1, prompt2, prompt3, prompt4, prompt5, prompt6, prompt7, prompt8, prompt9, prompt10,\
                      prompt11, prompt12, prompt13, prompt14, prompt15, prompt16, prompt17, prompt18, prompt19, prompt20]
        
    elif relation == "PersonLanguage":  # [4, 6,9]
        prompt1 = f"{subject_entity} speaks in {mask_token}."
        prompt2 = f"{subject_entity} wrote language is{mask_token}."
        prompt3 = f"{subject_entity} wrote in {mask_token}."
        prompt4 = f"{subject_entity} writes language {mask_token}."
        prompt5 = f"{subject_entity} writes in {mask_token}."
        prompt6 = f"{subject_entity} uses {mask_token}."
        prompt7 = f"{subject_entity} speaks {mask_token}."
        prompt8 = f"{subject_entity} written {mask_token}."
        prompt9 = f"{subject_entity} signs {mask_token}."
        prompt10 = f"{subject_entity} second language is {mask_token}."    #[0.621]  
        
        prompt_list = [prompt1, prompt2, prompt3, prompt4, prompt5, prompt6, prompt7, prompt8, prompt9, prompt10]

    elif relation == "PersonProfession": # 'job, work, profession, craft, employment, career'
        prompt1 = f"{subject_entity} is a {mask_token} by profession."
        prompt2 = f"{subject_entity} worked as a {mask_token}."
        prompt3 = f"Occupation of {subject_entity} is a {mask_token}."
        prompt4 = f"{subject_entity} has the job of {mask_token}."
        prompt5 = f"{subject_entity} has a career in {mask_token}."
        prompt6 = f"{subject_entity} employed as a {mask_token}."       
        prompt7 = f"The job of {subject_entity} is {mask_token}."
        prompt8 = f"{subject_entity} is a {}.".format(mask_token) + ", which is an occupation requiring special education."
        prompt9 = f"{subject_entity} received a specialized professional training and became a {mask_token}."       
        
        prompt_list = [prompt1, prompt2, prompt3, prompt4, prompt5, prompt6, prompt7, prompt8, prompt9]
        
    elif relation == "PersonInstrument":
        prompt1 = f"{subject_entity} plays {mask_token}, which is an instrument."
        prompt2 = f"{subject_entity} plays {mask_token}."
        prompt3 = f"{subject_entity} plays instrument of {mask_token}."
        prompt4 = f"{subject_entity} taken up the {mask_token} at the age of "
        
        
        prompt_list = [prompt1, prompt2, prompt3]
        
    elif relation == "PersonEmployer":
        prompt1 = f"{subject_entity} joined {mask_token}, " f"which is a firm that employs workers."
        prompt2 = f"{subject_entity} works for {mask_token}."
        prompt3 = f"{subject_entity} is employed by {mask_token}."
        prompt4 = f"{subject_entity} worked at {mask_token}."
        prompt5 = f"{subject_entity} started his career in the {mask_token}."
        prompt6 = f"{subject_entity} is the CEO of {mask_token}."
        prompt7 = f"{subject_entity} was appointed as CEO of {mask_token}."
        prompt8 = f"{subject_entity} became a professor at {mask_token}."
        prompt9 = f"{subject_entity} has served as a visiting professor at {mask_token}."
        prompt10 = f"{subject_entity} set up {mask_token}."
        prompt11 = f"{subject_entity} is the founder and current chairman and CEO of the {mask_token}."
        prompt12 = f"{subject_entity} is the chairman of {mask_token}."
        prompt13 = f"{subject_entity} co-founded and was CEO of {mask_token}."
        prompt14 = f"{subject_entity} is an employer at {mask_token}," f"which is a firm that employs workers."

        prompt_list = [prompt1, prompt2, prompt3, prompt4, prompt5, prompt6, prompt7, prompt8, prompt9, prompt10, \
                       prompt11, prompt12,prompt13, prompt14]

    elif relation == "PersonPlaceOfDeath":
        prompt1 = f"{subject_entity} died at {mask_token}."
        prompt2 = f"{subject_entity} died in {mask_token}."
        prompt3 = f"The death place of {subject_entity} at {mask_token}."
        prompt4 = f"The death location of {subject_entity} at {mask_token}."
        
        prompt_list = [prompt1, prompt2, prompt3, prompt4]

    elif relation == "PersonCauseOfDeath":
        prompt1 = f"{subject_entity} died due to {mask_token}."
        prompt2 = f"{subject_entity} died of {mask_token}."
        prompt3 = f"{subject_entity} died from {mask_token}."
        prompt4 = f"{subject_entity} death cause is {mask_token}."
        prompt5 = f"murder method of {subject_entity} is {mask_token}."
    
        prompt_list = [prompt1, prompt2, prompt3, prompt4, prompt5]
        
    elif relation == "CompanyParentOrganization": # 删了4，1变成原来的4
        prompt1 = f"The parent organisation of {subject_entity} is {mask_token}."
        prompt2 = f"{subject_entity} owed by {mask_token}."
        prompt3 = f"{subject_entity} is part of {mask_token}."
        prompt4 = f"The parent company of {subject_entity} is {mask_token}."
        prompt5 = f"{mask_token} holding {subject_entity}."
        
        prompt_list = [prompt1, prompt2, prompt3, prompt4, prompt5]
        
    if subject_entity == 'prompts_num':
        return len(prompt_list)

    return prompt_list[prompt_pos]

def one_relation_one_prompt(input_rows_all, raltion_type: str, mask_token: str, pipe, prompt_pos, threshold):
    # one relation
    input_rows = list(filter(lambda row : row['Relation'] == raltion_type, input_rows_all))
    
    # Create prompts
    logger.info("Creating prompts for {}......".format(raltion_type))
    
#     prompts = PromptSet([create_prompt(
#                             subject_entity=row["SubjectEntity"],
#                             relation=row["Relation"],
#                             mask_token=mask_token,
#                             prompt_pos=prompt_pos)['prompt_list'] for row in input_rows])
    
    prompt_list = []
    for row in input_rows:
        prompt = create_prompt(
                            subject_entity=row["SubjectEntity"],
                            relation=row["Relation"],
                            mask_token=mask_token,
                            prompt_pos=prompt_pos)
        prompt_list.append(prompt)
    prompts = PromptSet(prompt_list)
        

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
            "Prompt":prompt,
            "ObjectEntities": [
                seq["token_str"]
                for seq in output if seq["score"] > threshold],
        }
        results.append(result)
    return results

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
    input_rows_all = read_lm_kbc_jsonl(args.input)
    logger.info(f"Loaded {len(input_rows_all):,} rows.")
    

    df = pd.DataFrame(input_rows_all)
    relations = df.Relation.value_counts().keys()
    relations_dic = dict()
    for relation in relations:
        print(relation)
        # 一个relation 对应的prompts个数
        prompts_num = create_prompt('prompts_num',relation=relation, mask_token=mask_token)
        # run all prompts, save each result as a jsonl file.
        for pos in range(prompts_num):
            results = one_relation_one_prompt(input_rows_all,
                                              raltion_type=relation,
                                              pipe=pipe,
                                              mask_token=mask_token, 
                                              prompt_pos=pos, 
                                              threshold = args.threshold)

            # Save the results
            output = args.output + relation + '_' + str(pos + 1) + '.jsonl'
            logger.info(f"Saving the results to \"{output}\"...")
            with open(output, "w") as f:
                for result in results:
                    f.write(json.dumps(result) + "\n")
    

def main():
    parser = argparse.ArgumentParser(
        description="Probe a Language Model and "
                    "Run the Baseline Method on Prompt Outputs"
    )

    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="bert-large-cased", # 1. "bert-large-cased" 2. "allenai/scibert_scivocab_uncased" 3. "dmis-lab/biobert-large-cased-v1.1" 4 recobo/chemical-bert-uncased
        help="HuggingFace model name (default: bert-large-cased)",
    )

    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="Input test file (required)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
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
        "-t",
        "--threshold",
        type=float,
        required=True,
        default=0.1,
        help="Probability threshold (default: 0.5)",
    )
    parser.add_argument(
        "-g",
        "--gpu",
        type=int,
        default=0,
        help="GPU ID, (default: -1, i.e., using CPU)"
    )

    args = parser.parse_args()

    run(args)


if __name__ == '__main__':
    main()