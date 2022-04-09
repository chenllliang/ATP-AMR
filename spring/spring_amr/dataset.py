import imp
import logging
import pdb
import random
import torch
from cached_property import cached_property
from torch.utils.data import Dataset
from spring_amr.IO import read_raw_amr_data
from spring_amr.tokenization_bart import MultiTaskAmrTokenizer
from pathlib import Path
from glob import glob
import pdb

def reverse_direction(x, y, pad_token_id=1):
    input_ids = torch.cat([y['decoder_input_ids'], y['lm_labels'][:, -1:]], 1)
    attention_mask = torch.ones_like(input_ids)
    attention_mask[input_ids == pad_token_id] = 0
    decoder_input_ids = x['input_ids'][:,:-1]
    lm_labels = x['input_ids'][:,1:]
    x = {'input_ids': input_ids, 'attention_mask': attention_mask}
    y = {'decoder_input_ids': decoder_input_ids, 'lm_labels': lm_labels}
    return x, y

class SentenceDataset(Dataset):
    def __init__(self,path,tokenizer):
        with open(path,"r") as f:
            sents = f.readlines()
        self.sentences = [i.strip() for i in sents]
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sample = {}
        sample['id'] = idx
        sample['sentences'] = self.sentences[idx]
        return sample
    
    def size(self, sample):
        return len(sample['sentences'].split(" "))

    def collate_fn(self, samples, device=torch.device('cpu')):
        x = [s['sentences'] for s in samples]
        x, extra = self.tokenizer.batch_encode_sentences(x, device=device)
        extra['ids'] = [s['id'] for s in samples]
        return x,  extra
    

class SentenceLoader:
    def __init__(self, dataset, batch_size=800 ,device=torch.device('cpu'), shuffle=False, sort=False):
        assert not (shuffle and sort)
        self.batch_size = batch_size
        self.tokenizer = dataset.tokenizer
        self.dataset = dataset
        self.device = device
        self.shuffle = shuffle
        self.sort = sort

    def __len__(self):
        it = self.sampler()
        it = [[self.dataset[s] for s in b] for b in it]
        return len(it)
    
    def __iter__(self):
        it = self.sampler()
        it = ([[self.dataset[s] for s in b] for b in it])
        it = (self.dataset.collate_fn(b, device=self.device) for b in it)
        return it

    @cached_property
    def sort_ids(self):
        if isinstance(self.dataset,MultiTaskDataset):
            lengths = [len(s[0].split()) for s in self.dataset.sentences]
        else:
            lengths = [len(s.split()) for s in self.dataset.sentences]
        ids, _ = zip(*sorted(enumerate(lengths), reverse=True))
        ids = list(ids)
        return ids

    def sampler(self):
        ids = list(range(len(self.dataset)))[::-1]
        
        if self.shuffle:
            random.shuffle(ids)
        if self.sort:
            ids = self.sort_ids.copy()

        batch_longest = 0
        batch_nexamps = 0
        batch_ntokens = 0
        batch_ids = []

        def discharge():
            nonlocal batch_longest
            nonlocal batch_nexamps
            nonlocal batch_ntokens
            ret = batch_ids.copy()
            batch_longest *= 0
            batch_nexamps *= 0
            batch_ntokens *= 0
            batch_ids[:] = []
            return ret

        while ids:
            idx = ids.pop()
            size = self.dataset.size(self.dataset[idx])
            cand_batch_ntokens = max(size, batch_longest) * (batch_nexamps + 1)
            if cand_batch_ntokens > self.batch_size and batch_ids:
                yield discharge()
            batch_longest = max(batch_longest, size)
            batch_nexamps += 1
            batch_ntokens = batch_longest * batch_nexamps
            batch_ids.append(idx)

            if len(batch_ids) == 1 and batch_ntokens > self.batch_size:
                yield discharge()

        if batch_ids:
            yield discharge()


class SummDataset(Dataset):
    pass



class AMRDataset_Index(Dataset):
    
    def __init__(
        self,
        paths,
        tokenizer,
        indexs,
        device=torch.device('cpu'),
        use_recategorization=False,
        remove_longer_than=None,
        remove_wiki=False,
        dereify=True,
    ):
        self.paths = paths
        self.tokenizer = tokenizer
        self.device = device
        total_graphs = read_raw_amr_data(paths, use_recategorization, remove_wiki=remove_wiki, dereify=dereify)
        assert max(indexs)<len(total_graphs)

        graphs = [j for i,j in enumerate(total_graphs) if i in indexs]
        

        self.graphs = []
        self.sentences = []
        self.linearized = []
        self.linearized_extra = []
        self.remove_longer_than = remove_longer_than
        for g in graphs:
            l, e = self.tokenizer.linearize(g)
            
            try:
                self.tokenizer.batch_encode_sentences([g.metadata['snt']])
            except:
                logging.warning('Invalid sentence!')
                continue

            if remove_longer_than and len(l) > remove_longer_than:
                continue
            if len(l) > 1024:
                logging.warning('Sequence longer than 1024 included. BART does not support it!')

            self.sentences.append(g.metadata['snt'])
            self.graphs.append(g)
            self.linearized.append(l)
            self.linearized_extra.append(e)
        


    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        sample = {}
        sample['id'] = idx
        sample['sentences'] = self.sentences[idx]
        if self.linearized is not None:
            sample['linearized_graphs_ids'] = self.linearized[idx]
            sample.update(self.linearized_extra[idx])            
        return sample
    
    def size(self, sample):
        return len(sample['linearized_graphs_ids'])
    
    def collate_fn(self, samples, device=torch.device('cpu')):
        x = [s['sentences'] for s in samples]
        x, extra = self.tokenizer.batch_encode_sentences(x, device=device)
        if 'linearized_graphs_ids' in samples[0]:
            y = [s['linearized_graphs_ids'] for s in samples]
            y, extra_y = self.tokenizer.batch_encode_graphs_from_linearized(y, samples, device=device)
            extra.update(extra_y)
        else:
            y = None
        extra['ids'] = [s['id'] for s in samples]
        return x, y, extra

class AMRDataset(Dataset):
    
    def __init__(
        self,
        paths,
        tokenizer,
        device=torch.device('cpu'),
        use_recategorization=False,
        remove_longer_than=1024,
        remove_wiki=False,
        dereify=True,
    ):
        self.paths = paths
        self.tokenizer = tokenizer
        self.device = device
        graphs = read_raw_amr_data(paths, use_recategorization, remove_wiki=remove_wiki, dereify=dereify)
        self.graphs = []
        self.sentences = []
        self.linearized = []
        self.linearized_extra = []
        self.remove_longer_than = remove_longer_than
        print("max input length = ",remove_longer_than)

        sentence_tokens_len = []

        for g in graphs:
            l, e = self.tokenizer.linearize(g)
            
            try:
                #self.tokenizer.batch_encode_sentences([g.metadata['snt']])
                sentence_tokens_len.append(self.tokenizer.batch_encode_sentences([g.metadata['snt']])[0]['input_ids'].shape[1])
            except:
                logging.warning('Invalid sentence!')
                continue

            if remove_longer_than and len(l) > remove_longer_than:
                continue
            if len(l) > 1024:
                print('Sequence longer than 1024 included. BART does not support it!')

            self.sentences.append(g.metadata['snt'])
            self.graphs.append(g)
            self.linearized.append(l)
            self.linearized_extra.append(e)

        
        # #HACK 
        # import json

        # with open("linerized.txt","w") as f:
        #     json.dump([ i['linearized_graphs'] for i in self.linearized_extra],f)

        



    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        sample = {}
        sample['id'] = idx
        sample['sentences'] = self.sentences[idx]
        if self.linearized is not None:
            sample['linearized_graphs_ids'] = self.linearized[idx]
            sample.update(self.linearized_extra[idx])            
        return sample
    
    def size(self, sample):
        return len(sample['linearized_graphs_ids'])
    
    def collate_fn(self, samples, device=torch.device('cpu')):
        x = [s['sentences'] for s in samples]
        x, extra = self.tokenizer.batch_encode_sentences(x, device=device)
        if 'linearized_graphs_ids' in samples[0]:
            y = [s['linearized_graphs_ids'] for s in samples]
            y, extra_y = self.tokenizer.batch_encode_graphs_from_linearized(y, samples, device=device)
            extra.update(extra_y)
        else:
            y = None
        extra['ids'] = [s['id'] for s in samples]
        return x, y, extra
    
class AMRDatasetTokenBatcherAndLoader:
    
    def __init__(self, dataset, batch_size=800 ,device=torch.device('cpu'), shuffle=False, sort=False):
        assert not (shuffle and sort)
        self.batch_size = batch_size
        self.tokenizer = dataset.tokenizer
        self.dataset = dataset
        self.device = device
        self.shuffle = shuffle
        self.sort = sort

    def __len__(self):
        it = self.sampler()
        it = [[self.dataset[s] for s in b] for b in it]
        return len(it)
    
    def __iter__(self):
        it = self.sampler()
        it = ([[self.dataset[s] for s in b] for b in it])
        it = (self.dataset.collate_fn(b, device=self.device) for b in it)
        return it

    @cached_property
    def sort_ids(self):
        if isinstance(self.dataset,MultiTaskDataset):
            lengths = [len(s[0].split()) for s in self.dataset.sentences]
        else:
            lengths = [len(s.split()) for s in self.dataset.sentences]
        ids, _ = zip(*sorted(enumerate(lengths), reverse=True))
        ids = list(ids)
        return ids

    def sampler(self):
        ids = list(range(len(self.dataset)))[::-1]
        
        if self.shuffle:
            random.shuffle(ids)
        if self.sort:
            ids = self.sort_ids.copy()

        batch_longest = 0
        batch_nexamps = 0
        batch_ntokens = 0
        batch_ids = []

        def discharge():
            nonlocal batch_longest
            nonlocal batch_nexamps
            nonlocal batch_ntokens
            ret = batch_ids.copy()
            batch_longest *= 0
            batch_nexamps *= 0
            batch_ntokens *= 0
            batch_ids[:] = []
            return ret

        while ids:
            idx = ids.pop()
            size = self.dataset.size(self.dataset[idx]) # graph's length, longer than sentence
            cand_batch_ntokens = max(size, batch_longest) * (batch_nexamps + 1)
            if cand_batch_ntokens > self.batch_size and batch_ids:
                yield discharge()
            batch_longest = max(batch_longest, size)
            batch_nexamps += 1
            batch_ntokens = batch_longest * batch_nexamps
            batch_ids.append(idx)

            if len(batch_ids) == 1 and batch_ntokens > self.batch_size:
                yield discharge()

        if batch_ids:
            yield discharge()


def read_srl_tributes(path):
    f = open(path, "r")
    samples = f.readlines()
    tributes = []

    for i in samples:
        sample = {}
        snt, tri = i.strip().split("|||")
        sample["snt"] = snt.strip()
        sample["tributes"] = eval(tri.strip())
        tributes.append(sample)

    return tributes

def read_dp_samples(src,gold):
    srcs = open(src,"r").readlines()
    golds = open(gold,"r").readlines()

    assert len(srcs) == len(golds)

    samples =[]

    for i,j in zip(srcs,golds):
        samples.append({"snt":i.strip(),"dp":j.strip()})
    
    return samples


class MultiTaskDataset(Dataset):
    def __init__(
        self,
        path_srcs:dict,
        path_gold:dict,
        tokenizer:MultiTaskAmrTokenizer,
        remove_longer_than=None,
        device=torch.device('cpu'),
        use_recategorization=False,
        remove_wiki=False,
        dereify=True,
    ):

        assert len(path_gold)==len(path_srcs)

        self.dp_samples = []
        self.srl_samples = []
        self.amr_samples = []

        self.graphs = []
        self.sentences = []
        self.linearized = []
        self.linearized_extra = []
        self.remove_longer_than = remove_longer_than

        self.tokenizer = tokenizer
        self.device = device

        self.lens = []

        if "amr" in path_srcs.keys():
            paths = []
            print(path_srcs['amr'])
            if isinstance(path_srcs['amr'], str) or isinstance(path_srcs['amr'], Path):
                glob_pattn = [path_srcs['amr']]
                for gpattn in glob_pattn:
                    paths += [Path(p) for p in glob(gpattn)]
            else:
                paths = [Path(p) for p in path_srcs['amr']]
            
            graphs = read_raw_amr_data(paths, use_recategorization, remove_wiki=remove_wiki, dereify=dereify)

            for g in graphs:
                l, e = self.tokenizer.linearize(g)
                try:
                    self.tokenizer.batch_encode_sentences([g.metadata['snt']])
                except:
                    logging.warning('Invalid sentence!')
                    continue

                if remove_longer_than and len(l) > remove_longer_than:
                    continue

                self.sentences.append((g.metadata['snt'],"<amr>"))
                self.graphs.append(g)
                self.linearized.append(l)
                self.lens.append(len(l))
        

        if "dp" in path_srcs.keys():
            samples=read_dp_samples(path_srcs['dp'],path_gold['dp'])
            for g in samples:
                sent = g['snt']
                dp = g['dp']
            
                ids, e = tokenizer.tokenize_dp(dp)

                if self.remove_longer_than and len(ids) > self.remove_longer_than:
                    continue
                if len(sent) > self.remove_longer_than:
                    continue

                self.sentences.append((sent,"<dp>"))
                self.lens.append(len(ids))
                self.linearized.append(ids)

        
        if "srl" in path_srcs.keys():
            samples=read_dp_samples(path_srcs['srl'],path_gold['srl'])
            for g in samples:
                sent = g['snt']
                dp = g['dp']
            
                ids, e = tokenizer.tokenize_dp(dp)

                if self.remove_longer_than and len(ids) > self.remove_longer_than:
                    continue
                if len(sent) > self.remove_longer_than:
                    continue

                self.sentences.append((sent,"<srl>"))
                self.lens.append(len(ids))
                self.linearized.append(ids)
        
        print('[Direction]: Multitask')
        print("[Data NUM]:{}".format(len(self.sentences)))
        print("[Mean Target len]: {}".format(sum(self.lens) / len(self.lens)))
        print("[Max Target len]: {}".format(max(self.lens)))
    
    def __len__(self):
        return len(self.sentences)

    def size(self, sample):
        return len(sample['linearized_graphs_ids'])

    def __getitem__(self, idx):
        sample = {}
        sample['id'] = idx
        sample['sentences'] = self.sentences[idx]
        if self.linearized is not None:
            sample['linearized_graphs_ids'] = self.linearized[idx]
        return sample

    def collate_fn(self, samples, device=torch.device('cpu')):
        x = [s['sentences'] for s in samples]
        sents = []
        for i in x:
            sents.append(" ".join([i[1],i[0]]))

        x, extra = self.tokenizer.batch_encode_sentences(sents, device=device)
        if "linearized_graphs_ids" in samples[0]:
            y = [s['linearized_graphs_ids'] for s in samples]
            y, extra_y = self.tokenizer.batch_encode_graphs_from_linearized(y, extras=None, device=device)
            extra.update(extra_y)
        else:
            y = None
        extra['ids'] = [s['id'] for s in samples]
        return x, y, extra






    
    

class DpPretrainDataset(Dataset):
    def __init__(
        self,
        path_src,
        path_gold,
        tokenizer,
        remove_longer_than=None,
        device=torch.device('cpu'),
    ):
        self.tokenizer = tokenizer
        self.device = device
        self.samples = read_dp_samples(path_src,path_gold)


        self.sentences = []
        self.tgt = []
        self.lens = []
        self.graphs = []
        self.remove_longer_than = remove_longer_than

        for g in self.samples:
            sent = g['snt']
            dp = g['dp']
            
            
            ids, e = tokenizer.tokenize_dp(dp)

            if self.remove_longer_than and len(ids) > self.remove_longer_than:
                continue
            if len(sent) > self.remove_longer_than:
                continue

            self.sentences.append(sent)
            self.lens.append(len(ids))
            self.tgt.append(ids)
            self.graphs.append(e)
        
        # pdb.set_trace()
        
        import json
        with open("linerized_srl_fixed.txt","w") as f:
            json.dump(self.graphs,f)


        
        print('[Direction]: Dependency Parsing')
        print("[Data NUM]:{}".format(len(self.sentences)))
        print("[Mean Target len]: {}".format(sum(self.lens) / len(self.lens)))
        print("[Max Target len]: {}".format(max(self.lens)))
    
    def __len__(self):
        return len(self.sentences)

    def size(self, sample):
        return len(sample['linearized_graphs_ids'])

    def __getitem__(self, idx):
        sample = {}
        sample['id'] = idx
        sample['sentences'] = self.sentences[idx]
        if self.tgt is not None:
            sample['linearized_graphs_ids'] = self.tgt[idx]
        return sample

    def collate_fn(self, samples, device=torch.device('cpu')):
        x = [s['sentences'] for s in samples]
        x, extra = self.tokenizer.batch_encode_sentences(x, device=device)
        if "linearized_graphs_ids" in samples[0]:
            y = [s['linearized_graphs_ids'] for s in samples]
            y, extra_y = self.tokenizer.batch_encode_graphs_from_linearized(y, extras=None, device=device)
            extra.update(extra_y)
        else:
            y = None
        extra['ids'] = [s['id'] for s in samples]
        return x, y, extra




class SrlPretrainDataset(Dataset):
    def __init__(
            self,
            paths,
            tokenizer,
            device=torch.device('cpu'),
    ):
        self.paths = paths
        self.tokenizer = tokenizer
        self.device = device
        self.tributes = read_srl_tributes(self.paths)

        self.gold_tuples = []
        self.sentences = []
        self.tgt = []
        self.lens = []

        for g in self.tributes:
            sent = g['snt']
            tuples = g['tributes']
            self.sentences.append(sent)
            ids, e = tokenizer.encode_srl_tuples(tuples)
            self.gold_tuples.append(list(filter(lambda x:x[1]!=':verb', tuples)))

            self.lens.append(len(ids))
            self.tgt.append(ids)
        print('[Direction]: SRL')
        print("[Data NUM]:{}".format(len(self.sentences)))
        print("[Mean Target len]: {}".format(sum(self.lens) / len(self.lens)))

    def __len__(self):
        return len(self.sentences)

    def size(self, sample):
        return len(sample['linearized_graphs_ids'])

    def __getitem__(self, idx):
        sample = {}
        sample['id'] = idx
        sample['sentences'] = self.sentences[idx]
        if self.tgt is not None:
            sample['linearized_graphs_ids'] = self.tgt[idx]
        return sample

    def collate_fn(self, samples, device=torch.device('cpu')):
        x = [s['sentences'] for s in samples]
        x, extra = self.tokenizer.batch_encode_sentences(x, device=device)
        if "linearized_graphs_ids" in samples[0]:
            y = [s['linearized_graphs_ids'] for s in samples]
            y, extra_y = self.tokenizer.batch_encode_graphs_from_linearized(y, extras=None, device=device)
            extra.update(extra_y)
        else:
            y = None
        extra['ids'] = [s['id'] for s in samples]
        return x, y, extra


if __name__=="__main__":
    pass