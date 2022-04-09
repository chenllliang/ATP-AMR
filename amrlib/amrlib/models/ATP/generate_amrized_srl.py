import penman

class SRL:
    def __init__(self) -> None:
        self.snt = None
        self.semantic_roles = None
    
    def __str__(self) -> str:
        return str(self.snt)+"\n"+str(self.semantic_roles)
    
    def get_word_sense(self):
        pass
    
    def easy_amrization(self):
        pass

def get_semantic_roles_from_BIO(tokens,bio):
    roles=[]

    cur_role=''
    cur_tok=[]
    for i,j in enumerate(bio):
        
        if "B-" in j and cur_role=='':
            cur_role = j.split("-",1)[1]
            cur_tok.append(tokens[i]+"@"+str(i))

            
        
        elif "B-" in j and cur_role!='':
            roles.append({cur_role:" ".join(cur_tok)})

            cur_role=''
            cur_tok=[]

            cur_role = j.split("-",1)[1]
            cur_tok.append(tokens[i]+"@"+str(i))
        

        elif "I-" in j:
            cur_tok.append(tokens[i]+"@"+str(i))

        elif "O" in j and cur_role!='':
            roles.append({cur_role:" ".join(cur_tok)})
            cur_role=''
            cur_tok=[]

    
    return roles
        



def read_from_conll12(path):
    with open(path,"r") as f:
        samples = f.readlines()
        srls = []

        cur_sent = ''
        cur_srl = None

        for i in samples:
            if "|||" in i:
                snt,args = i.split("|||")
                words = snt.strip().split(" ")[1:]
                roles = args.strip().split(" ")
                assert len(words)==len(roles)

                args = get_semantic_roles_from_BIO(words,roles)
                
                if cur_sent=='':
                    cur_srl = SRL()
                    cur_srl.snt = words
                    cur_srl.semantic_roles = [args]
                    cur_sent = words
                else:
                    if words == cur_sent:
                        cur_srl.semantic_roles.append(args)
                    else:
                        srls.append(cur_srl)
                        cur_srl = SRL()
                        cur_srl.snt = words
                        cur_srl.semantic_roles = [args]
                        cur_sent = words
        return srls






srls = read_from_conll12("/home/cl/Semantic-Role-Labeling/data/conll2012.train.txt")

print(len(srls))

print(srls[0])



