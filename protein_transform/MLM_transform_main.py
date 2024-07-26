import argparse
import os
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model.MLM_model_transform import BERT
from model.MLM_dataset import BERTDataset, protein
from model.MLM_embedding import BERTEmbedding
from train.pretrain import BERTTrainer

import os 
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
#device = "cpu"

def get_args_parser():
    parser = argparse.ArgumentParser('Transformer pre-training', add_help=True)

    parser.add_argument("--train_dataset", required=True, type=str, help="train dataset for train bert")
    parser.add_argument("--vocab_size", required=True, type=int, help="Number of vocabsize")
    parser.add_argument("--seg_size", required=True, type=int, help="Number of species")
    parser.add_argument("--test_dataset", type=str, default=None, help="test set for evaluate train set")
    #parser.add_argument("--vocab_path", required=True, type=str, help="built vocab model path with bert-vocab")
    parser.add_argument("--output_path", required=True, type=str, help="ex output/bert.model")
    parser.add_argument("--test_output", type=str, default=None, help="test output")
    parser.add_argument("--hidden", type=int, default=128, help="hidden size of transformer model")
    parser.add_argument("--layers", type=int, default=4, help="number of layers")
    parser.add_argument("--attn_heads", type=int, default=8, help="number of attention heads")
   # parser.add_argument("-s", "--seq_len", type=int, default=20, help="maximum sequence len")

    parser.add_argument("--batch_size", type=int, default=8, help="number of batch_size")
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("--num_workers", type=int, default=5, help="dataloader worker size")

    parser.add_argument("--with_cuda", type=bool, default=True, help="training with CUDA: true, or false")
    parser.add_argument("--log_freq", type=int, default=10, help="printing loss every n iter: setting n")
    parser.add_argument("--corpus_lines", type=int, default=None, help="total number of lines in corpus")
    parser.add_argument("--cuda_devices", type=int, nargs='+', default=0, help="CUDA device ids")
    parser.add_argument("--on_memory", type=bool, default=True, help="Loading on memory: true or false")

    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate of adam")
    parser.add_argument("--adam_weight_decay", type=float, default=0.01, help="weight_decay of adam")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="adam first beta value")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="adam first beta value")

    return parser


vocab = protein

def main (args, mode = 'train'):

    if mode == 'train':

        print(f"{mode} model =============================================================")    

        vocab = protein
        print(f"Vocab: {len(vocab)}")
    
        print("Loading Train Dataset", args.train_dataset)
        bertdataset = BERTDataset(args.train_dataset, protein)
        train_size = int(0.8 * len(bertdataset))
        val_size = len(bertdataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(bertdataset, [train_size, val_size])

        print("Creating Dataloader")
        train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers,shuffle=True)
        test_data_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers)

        print("Building BERT model (include embedding)")
        max_len = max([len(str(text)) for text in bertdataset.lines[:]])
        bert = BERT(args.vocab_size, max_len=max_len, seg_size=args.seg_size, hidden=args.hidden, n_layers=args.layers, attn_heads=args.attn_heads)

        print("Creating BERT Trainer")
        trainer = BERTTrainer(bert, args.vocab_size, train_dataloader=train_data_loader, test_dataloader=test_data_loader,
                          lr=args.lr, betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay,
                          with_cuda=args.with_cuda, cuda_devices=args.cuda_devices, log_freq=args.log_freq)

        print("Training Start")
        for epoch in range(args.epochs):
            trainer.train(epoch)
            trainer.save(epoch, args.output_path)
            if epoch % 10 ==0:
                print(f"Evaluate model ====================================================")
                trainer.test(epoch)


    else:  #test

       testdataset = BERTDataset(args.test_dataset, vocab)
       data = testdataset.__getitem__(0)

       file_path = 'output/bert_trained.model'
       output_path = args.output_path + ".ep%d" % epoch
       model = torch.load(output_path)
      
       with torch.no_grad():
           mask_lm_output = model.forward(data["bert_input"])

       criterion = nn.NLLLoss(ignore_index=0)
       mask_loss = criterion(mask_lm_output.transpose(1, 2), data["bert_label"])
       mask_pred = mask_lm_output.argmax(dim=-1)
       equal_tensor = mask_pred.eq(data["bert_label"])
       equal_tensor_padd = mask_pred[equal_tensor] != protein['padd']
       equal_tensor_padd_index = mask_pred[equal_tensor][equal_tensor_padd] != protein['unindex']

       logits_aa = equal_tensor_padd_index.sum().item()
       total_element = math.floor(((data["bert_input"] != protein['padd']).sum().item() + 1e-6) * 0.15)
       print(f'Evaluate =========================  logits_aa:{logits_aa} total_aa:{total_element}')
       #correct = (mask_pred.eq(data["bert_label"]).sum().item() - (mask_pred == protein['padd']).sum().item() - (mask_pred == protein['unindex']).sum().item())
       
       acc = correct / total_element
       with open(args.test_output, 'w') as f:
           print('sample input {samples.squeeze(0)}', file=f)
           print('sample label {target.squeeze(0)}', file=f)
           print('sample predi {pred_argmax.squeeze(0)}', file=f)
           print('sample accur {correct} {total_element} {acc}', file=f)

       print(f"Test finish {arg.test_output}")
       times.sleep(0.5)



if __name__ =='__main__':

    args = get_args_parser()
    args = args.parse_args()

    if args.output_path:
        Path(args.output_path).mkdir(parents=True, exist_ok=True)

    mode = 'train' # train/evalu

    if mode == 'train':
        main(args, mode=mode)
    else:
        print('test ==============================================================')
        main(args, mode=mode)
