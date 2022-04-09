from transformers import AutoConfig
from transformers.models.bart import BartForConditionalGeneration




from transformers.models.bart import BartTokenizer


# alter tokenizer for AMR Parsing

dp_edges = [i.strip() for i in open("/home/cl/AMR_Multitask_Inter/Dataset/DP/seq2seq/edges.txt","r").readlines()]
print("add dependency parsing relations to vocab: ",dp_edges)

class AMRization_Tokenizer(BartTokenizer):
    pass




def init_AMRization_tokenizer():
    return AMRization_Tokenizer()

def init_AMRization_model(name,checkpoint,dropout,attention_dropout,tokenizer,from_pretrained=True):
    if name is None:
        name = 'facebook/bart-large'

    if name == 'facebook/bart-base':
        tokenizer_name = 'facebook/bart-large'
    else:
        tokenizer_name = name

    config = AutoConfig.from_pretrained(name)
    config.output_past = False
    config.no_repeat_ngram_size = 0
    config.prefix = " "
    config.output_attentions = True
    config.dropout = dropout
    config.attention_dropout = attention_dropout

    if from_pretrained:
        model = BartForConditionalGeneration.from_pretrained(name, config=config)
    else:
        model = BartForConditionalGeneration(config)
    
    # model.resize_token_embeddings(55555)
    #The number of new tokens in the embedding matrix. 
    #Increasing the size will add newly initialized vectors at the end.
    #Reducing the size will remove vectors from the end. 



    return model

if __name__=="__main__":
    model = init_AMRization_model("facebook/bart-large",None,0.3,0,None)
    #original model 50265 tokens
    print()