import penman
from penman.graph import Graph
from penman.codec import PENMANCodec
import re

class SRL:
    def __init__(self) -> None:
        self.snt = None
        self.semantic_roles = None
    
    def __str__(self) -> str:
        return str(self.snt)+"\n"+str(self.semantic_roles)
    
    def get_word_sense(self):
        pass

    def isolated_srl_triples_amrization(self,splitby="and",remove_alignment=True,remove_empty_relation=True):
        # use "and" and "op" to connect the srls to a graph, no other process 
        # the top concept can be "and" , "multi-sentence" , or "None"
        penman_graph = Graph()
        triples = []
        
        instance_id = ["a"+str(i) for i in range(1000)]

        index=0

    

        if splitby=="and":
            triples.append((instance_id[index],':instance','and'))
            index += 1
            for i,j in enumerate(self.semantic_roles):
                predicate =[ t['V'] for t in j if "V" in t.keys() ]
                try:
                    triples.append((instance_id[index],':instance',predicate[0]))
                except Exception as e:
                    continue
                triples.append((instance_id[0],':op'+str(i+1),instance_id[index]))
                predicate_index = 0+index
                index += 1
                args = sorted(j,key=lambda x:list(x.keys())[0])
                for i in args:
                    if list(i.keys())[0]!="V":
                        triples.append((instance_id[index],':instance',list(i.values())[0]))
                        triples.append((instance_id[predicate_index],list(i.keys())[0],instance_id[index]))
                        index += 1
        
        if splitby=="multi-sentence":
            triples.append((instance_id[index],':instance','multi-sentence'))
            index += 1
            for i,j in enumerate(self.semantic_roles):
                predicate =[ t['V'] for t in j if "V" in t.keys() ]
                triples.append((instance_id[index],':instance',predicate[0]))
                triples.append((instance_id[0],':snt'+str(i+1),instance_id[index]))
                predicate_index = 0+index
                index += 1
                args = sorted(j,key=lambda x:list(x.keys())[0])
                for i in args:
                    if list(i.keys())[0]!="V":
                        triples.append((instance_id[index],':instance',list(i.values())[0]))
                        triples.append((instance_id[predicate_index],list(i.keys())[0],instance_id[index]))
                        index += 1
        
        if splitby=="None":
            instance_id[0]=' '
            triples.append((instance_id[index],':instance',''))
            index += 1
            for i,j in enumerate(self.semantic_roles):
                predicate =[ t['V'] for t in j if "V" in t.keys() ]
                triples.append((instance_id[index],':instance',predicate[0]))
                triples.append((instance_id[0],'',instance_id[index]))
                predicate_index = 0+index
                index += 1
                args = sorted(j,key=lambda x:list(x.keys())[0])
                for i in args:
                    if list(i.keys())[0]!="V":
                        triples.append((instance_id[index],':instance',list(i.values())[0]))
                        triples.append((instance_id[predicate_index],list(i.keys())[0],instance_id[index]))
                        index += 1



        penman_graph.triples = triples


        penman_graph.metadata={'snt':" ".join(self.snt)}
        codec = PENMANCodec()
        ret = codec.encode(penman_graph,compact=True)

        if remove_alignment==True:
            ret = re.sub("@@\d*","",ret)
        if remove_empty_relation==True:
            ret = re.sub(": ","",ret)

        return ret





def get_semantic_roles_from_BIO(tokens,bio):
    roles=[]
    cur_role=''
    cur_tok=[]
    for i,j in enumerate(bio):
        if "B-" in j and cur_role=='':
            cur_role = j.split("-",1)[1]
            cur_tok.append(tokens[i]+"@@"+str(i))
        elif "B-" in j and cur_role!='':
            roles.append({cur_role:" ".join(cur_tok)})
            cur_role=''
            cur_tok=[]
            cur_role = j.split("-",1)[1]
            cur_tok.append(tokens[i]+"@@"+str(i))
        elif "I-" in j:
            cur_tok.append(tokens[i]+"@@"+str(i))
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



srls = read_from_conll12("/mnt/unilm/v-lianchen/ATP/amrization/SRL/Semantic-Role-Labeling/data/conll2012.train.txt")




for i in srls:
    print(i.isolated_srl_triples_amrization())
    print()
    




