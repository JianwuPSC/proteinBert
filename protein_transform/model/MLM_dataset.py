from torch.utils.data import Dataset
import tqdm
import torch
import random

protein = {'A':0,'R':1,'N':2,'D':3,'C':4,'Q':5,'E':6,'G':7,'H':8,'I':9,'L':10,'K':11,'M':12,
          'F':13,'P':14,'S':15,'T':16,'W':17,'Y':18,'V':19,'SOS':20,'EOS':21,'mask':22,'unindex':23,'padd':24}

def random_word(sentence):
        
    tokens = []
    for i, token in enumerate(list(sentence)):
        tokens.append(token)
    output_label = []

    for i, token in enumerate(list(tokens)):
        prob = random.random()
        if prob < 0.15:
            prob /= 0.15

            # 80% randomly change token to mask token
            if prob < 0.8:
                tokens[i] = protein['mask']

            # 10% randomly change token to random token
            elif prob < 0.9:
                tokens[i] = random.randrange(len(protein))

            # 10% randomly change token to current token
            else:
                tokens[i] = protein.get(token, protein['unindex'])

            output_label.append(protein.get(token, protein['unindex']))

        else:
            tokens[i] = protein.get(token, protein['unindex'])
            output_label.append(protein['unindex'])

    return tokens, output_label



class BERTDataset(Dataset):
    
    """built dataset """
    
    # input file
    def __init__(self, corpus_path, vocab, encoding="utf-8", corpus_lines=None, on_memory=True):
        
        """corpus_path: input file (two L: sequence; type)
           vocab: default protein type (20 aa + SOS + EOS + '*' (unindex) + '-' (padding))"""
        
        self.vocab = vocab
        self.on_memory = on_memory
        self.corpus_lines = corpus_lines
        self.corpus_path = corpus_path
        self.encoding = encoding

        with open(corpus_path, "r", encoding=encoding) as f:

            self.lines = [line[:-1].split("\t")
                          for line in tqdm.tqdm(f, desc="Loading Dataset", total=corpus_lines)]
            self.corpus_lines = len(self.lines)
     
    # file lines             
    def __len__(self):
        return self.corpus_lines
    
    # random words    
    
    # return dict to tensor
    def __getitem__(self, item):
        
        sentence = self.lines[item][0]
        segment = self.lines[item][1]
        t1_random, t1_lab=random_word(sentence)

        # [CLS] tag = SOS tag, [SEP] tag = EOS tag
        t1 = [protein['SOS']] + t1_random + [protein['EOS']]
        t1_label = [protein['SOS']] + t1_lab + [protein['EOS']]
        segment_label = [int(segment)] + [int(segment) for _ in range(len(t1_random))] + [int(segment)]

        # [-] padding
        max_len = max([len(str(text)) for text in self.lines[:]])
        padding = [protein['padd'] for _ in range(max_len - len(t1))]
        seg_padding = [int(segment) for _ in range(max_len - len(t1))]
        t1.extend(padding), t1_label.extend(padding), segment_label.extend(seg_padding)

        # dict make
        output = {"bert_input": t1,
                  "bert_label": t1_label,
                  "segment_label": segment_label,
                  }

        # return dict
        return {key: torch.tensor(value) for key, value in output.items()}
