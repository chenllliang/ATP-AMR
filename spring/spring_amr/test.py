from spring_amr.utils_temp import instaniate_model_tokenizer_cl
from spring_amr.dataset import AMRDataset

model,tokenizer = instaniate_model_tokenizer_cl("facebook/bart-large",None,0.3,0)


import pdb
pdb.set_trace()


dataset = AMRDataset(["/home/cl/AMR_Active/AMR_Dataset/abstract_meaning_representation_amr_2.0/data/amrs/split/test/*.txt"],tokenizer)

