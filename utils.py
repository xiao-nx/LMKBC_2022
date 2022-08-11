from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co

class PromptSet(Dataset):
    def __init__(self, prompts):
        self.prompts = prompts

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, index) -> T_co:
        return self.prompts[index]


# selecte prompts according to the performance on train dataset
# after test the performance of each prompt on ber-large, we got several bettter prompts
def create_prompt(subject_entity: str, relation: str, mask_token: str, prompt_pos=0) -> str:
    """
    Depending on the relation, we fix the prompt
    """

    prompt = mask_token

    if relation == "CountryBordersWithCountry":
        prompt1 = f"{subject_entity} shares border with {mask_token}."
        #prompt2 = f"{subject_entity} next to {mask_token}."
        #prompt3 = f"{subject_entity} borders {mask_token}."
        prompt4 = f"{subject_entity} bordered by {mask_token}."
        prompt5 = f"{subject_entity} adjacents to {mask_token}."
        #prompt6 = f"{subject_entity} border {mask_token}."
        prompt_list = [prompt1, prompt4, prompt5]
        #prompt_list = [prompt1, prompt2, prompt3, prompt4, prompt5]
        
    elif relation == "CountryOfficialLanguage":
        prompt1 = f"The official language of {subject_entity} is {mask_token}."
        prompt2 = f"The language spoken in {subject_entity} is {mask_token}." # 2 3 重复
        #prompt3 = f"The language spoken in {subject_entity} is {mask_token}."
        prompt4 = f"The language official in {subject_entity} is {mask_token}."
        #prompt5 = f"{subject_entity} speaking {mask_token}."
        prompt_list = [prompt1, prompt2, prompt4]

    # for test
    elif relation == "StateSharesBorderState":
        prompt1 = f"{subject_entity} state next to {mask_token} state."
        prompt2 = f"{subject_entity} state borders {mask_token} state."
        prompt3 = f"{subject_entity} state bordered by {mask_token} state."
        prompt4 = f"{subject_entity} state adjacent to {mask_token} state."
        prompt_list = [prompt1, prompt2, prompt3, prompt4]
        
    elif relation == "RiverBasinsCountry":
        # ['water basin', ' catchment area', ' watershed', ' river basin', ' river basin district']
        prompt1 = f"{subject_entity} river basins in {mask_token}."
        #prompt2 = f"{mask_token} is river basin district of {subject_entity}."
        #prompt3 = f"{mask_token} is catchment area of {subject_entity}."
        prompt4 = f"the watershed in {mask_token} is {subject_entity}."
        
        prompt_list = [prompt1, prompt4]
        
    elif relation == "ChemicalCompoundElement": # [检查出来一些0.00] [2,3,4,5,6,7,8,9, 14, 16, 17, 19]
        prompt1 = f"{subject_entity} consists of {mask_token}, " f"which is an element."
        #prompt2 = f"{subject_entity} is set of {mask_token}."
        #prompt3 = f"{subject_entity} is part of {mask_token}."
        prompt4 = f"{subject_entity} includes part of {mask_token}."
        #prompt5 = f"{subject_entity} holonym of {mask_token}."
        #prompt6 = f"{subject_entity} have part of {mask_token}."
        #prompt7 = f"{subject_entity} has ingredient of {mask_token}."
        #prompt8 = f"{subject_entity} has component of {mask_token}."
        #prompt9 = f"{subject_entity} has as part of {mask_token}."
        prompt10 = f"{subject_entity} created out of {mask_token}."
        prompt11 = f"{subject_entity} created from {mask_token}."
        prompt12 = f"{subject_entity} contains {mask_token}."
        prompt13 = f"{subject_entity} composed of {mask_token}."
        #prompt14 = f"{subject_entity} assembled out of {mask_token}."
        prompt15 = f"{subject_entity} assembled from {mask_token}."
        #prompt16 = f"{subject_entity} amalgamation of {mask_token}."
        #prompt17 = f"{subject_entity} formed out of {mask_token}."
        prompt18 = f"{subject_entity} formed from {mask_token}."
        #prompt19 = f"{subject_entity} has member of {mask_token}."
        prompt20 = f"{subject_entity} comprised of {mask_token}."
    
        prompt_list = [prompt1]
        #prompt_list = [prompt1, prompt12, prompt13]
        #prompt_list = [prompt1, prompt4, prompt10, prompt11, prompt12, prompt13, prompt15, prompt18, prompt20]
        
    elif relation == "PersonLanguage":
        prompt1 = f"{subject_entity} speaks in {mask_token}."
        #prompt2 = f"{subject_entity} wrote language is{mask_token}." xxx
        #prompt3 = f"{subject_entity} wrote in {mask_token}." xxx
        prompt4 = f"{subject_entity} writes language {mask_token}."
        #prompt5 = f"{subject_entity} writes in {mask_token}."
        #prompt6 = f"{subject_entity} uses {mask_token}."
        #prompt7 = f"{subject_entity} speaks {mask_token}."
        #prompt8 = f"{subject_entity} written {mask_token}."
        #prompt9 = f"{subject_entity} signs {mask_token}."
        prompt10 = f"{subject_entity} second language is {mask_token}."    #[0.621] 
        
        prompt_list = [prompt1, prompt4, prompt10]
        #prompt_list = [prompt1, prompt2, prompt3, prompt4, prompt6, prompt7, prompt8, prompt10]

    elif relation == "PersonProfession": # 'job, work, profession, craft, employment, career'
        prompt1 = f"{subject_entity} is a {mask_token} by profession."
        prompt2 = f"{subject_entity} worked as a {mask_token}."
        #prompt3 = f"Occupation of {subject_entity} is a {mask_token}."
        prompt4 = f"{subject_entity} has the job of {mask_token}."
        #prompt5 = f"{subject_entity} has a career in {mask_token}."
        prompt6 = f"{subject_entity} employed as a {mask_token}."       
        #prompt7 = f"The job of {subject_entity} is {mask_token}."
        prompt8 = subject_entity + " is a {} ".format(mask_token) + ", which is an occupation requiring special education."
        prompt9 = f"{subject_entity} received a specialized professional training and became a {mask_token}."       
        
        prompt_list = [prompt1, prompt2, prompt4, prompt6, prompt8, prompt9]
        
    elif relation == "PersonInstrument":
        prompt1 = f"The musician {subject_entity} plays {mask_token}, which is an instrument."
        prompt2 = f"The musician {subject_entity} plays {mask_token}."
        prompt3 = f"The musician {subject_entity} plays instrument of {mask_token}."

        prompt_list = [prompt1, prompt2, prompt3]
        
    elif relation == "PersonEmployer":
        #prompt1 = f"{subject_entity} joined {mask_token}, which is a firm that employs workers."
        prompt1 = f"{subject_entity} joined {mask_token} company."
        #prompt2 = f"{subject_entity} works for {mask_token}."
        prompt3 = f"{subject_entity} is employed by {mask_token}."
        #prompt4 = f"{subject_entity} worked at {mask_token}."
        #prompt5 = f"{subject_entity} started his career in the {mask_token}."
        #prompt6 = f"{subject_entity} is the CEO of {mask_token}."
        #prompt7 = f"{subject_entity} was appointed as CEO of {mask_token}."
        #prompt8 = f"{subject_entity} became a professor at {mask_token}."
        #prompt9 = f"{subject_entity} has served as a visiting professor at {mask_token}."
        #prompt10 = f"{subject_entity} set up {mask_token}."
        #prompt11 = f"{subject_entity} is the founder and current chairman and CEO of the {mask_token}."
        prompt12 = f"{subject_entity} is the chairman of {mask_token}."
        #prompt13 = f"{subject_entity} co-founded and was CEO of {mask_token}."
        #prompt14 = f"{subject_entity} is an employer at {mask_token}, which is a firm that employs workers."
        
        prompt_list = [prompt1, prompt3, prompt12]
        #prompt_list = [prompt1, prompt3, prompt4, prompt5, prompt6, prompt7, prompt8, prompt12,prompt13, prompt14]

    elif relation == "PersonPlaceOfDeath":
        prompt1 = f"{subject_entity} died at {mask_token}."
        prompt2 = f"{subject_entity} died in {mask_token}."
        prompt3 = f"The death place of {subject_entity} at {mask_token}."
        prompt4 = f"The death location of {subject_entity} at {mask_token}."
        
        prompt_list = [prompt1, prompt2, prompt3, prompt4]

    elif relation == "PersonCauseOfDeath":
        prompt1 = f"{subject_entity} died due to {mask_token}."
        prompt2 = f"{subject_entity} died of {mask_token}."
        prompt3 = f"{subject_entity} died from {mask_token} disease."
        #prompt4 = f"{subject_entity} death cause is {mask_token}."
        #prompt5 = f"The murder method of {subject_entity} is {mask_token}."
    
        prompt_list = [prompt1, prompt2, prompt3]
        
    elif relation == "CompanyParentOrganization":
        prompt1 = f"The parent organization of {subject_entity} is {mask_token} company."
        prompt2 = f"{subject_entity} owed by {mask_token}."
        prompt3 = f"{subject_entity} is part of {mask_token}."
        prompt4 = f"The parent company of {subject_entity} is {mask_token} company."
        prompt5 = f"{mask_token} holding {subject_entity}."
        
        prompt_list = [prompt1,prompt3, prompt4]
        #prompt_list = [prompt1, prompt2, prompt3, prompt4, prompt5]
        
    if subject_entity == 'prompts_num':
        return len(prompt_list)

    return prompt_list[prompt_pos]


