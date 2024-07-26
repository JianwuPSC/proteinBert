import argparse
import os
import torch
from pathlib import Path
from tensorboardX import SummaryWriter

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler

from model.MLM_dataset import BERTDataset, protein
from model.MLM_model_LSTM import Model

import sys
sys.path.append(r'/home/wuj/data/protein_design/Bert/protein_LSTM/train')
import train_one_epoch
import evaluate


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


with open('util/parameter.txt', 'r') as file:
    content = file.read()	

parameter = {}
for line in content.split('\n'):
    if line:
        key, value = line.split(':')
        parameter[key] = value


def get_args_parser():
    parser = argparse.ArgumentParser('LSTM-attention pre-training', add_help=True)
    parser.add_argument('--batch_size', default=72, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * #gpus)')
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--accum_iter', default=400, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')
    
    parser.add_argument('--corpus_path',default='', 
                        help='dataset pathway (sequence  type)')
    parser.add_argument('--output_dir', default='./output_dir_pretrained',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir_pretrained',
                        help='path where to tensorboard log')
    parser.add_argument('--test_path',default='',
                        help='dataset pathway (sequence  type)')
    parser.add_argument('--test_output',default='',
                        help='dataset pathway (sequence  type)')

    # Model parameters
    parser.add_argument('--embedding_dim', default=128, type=int,
                        help='embedding_dim size (default=128)')
    parser.add_argument('--hidden_size', default=512, type=int,
                        help='hidden_size size (default=512)')
    
    # Optimizer parameters
    parser.add_argument('--weight_decay', default=0.05, type=float,
                        help='weight decay (default 0.05)')
    parser.add_argument('--lr', default=0.001, type=float, metavar='LR',
                        help='learning rate (absolute lr default=0.001)')
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int,metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=5, type=int)
    parser.add_argument('--pin_mem',action='store_true',
                        help='Pin CPU memory in Dataloader for more efficient (sometimes) transfer to GPU')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)
    
    return parser


def main (args, mode = 'train', test_image_path = ''):
    print(f"{mode} model=============================================================")

    if mode == 'train':
        
        # 构建数据批次
        bertdataset = BERTDataset(args.corpus_path, protein)
        train_size = int(0.8 * len(bertdataset))
        val_size = len(bertdataset) - train_size

        train_dataset, val_dataset = torch.utils.data.random_split(bertdataset, [train_size, val_size])

        #sampler_train = torch.utils.data.RandomSampler(bertdataset)
        #sampler_val = torch.utils.data.SequentialSampler(data_val)
        
        #data_loader_train = torch.utils.data.DataLoader(
        #     data_train, sampler=sampler_train, 
        #     batch_size=args.batch_size,
        #     num_workers=args.num_workers,
        #     pin_memory=args.pin_mem,
        #     drop_last=True)

        data_loader_train = torch.utils.data.DataLoader(
                train_dataset, batch_size=args.batch_size, 
                shuffle=True, num_workers=args.num_workers,
                pin_memory=args.pin_mem,drop_last=True)

        #data_loader_val = torch.utils.data.DataLoader(
        #     data_val, sampler=sampler_val, 
        #     batch_size=args.batch_size,
        #     num_workers=args.num_workers,
        #     pin_memory=args.pin_mem,
        #     drop_last=False)

        data_loader_val = torch.utils.data.DataLoader(
                val_dataset, batch_size=args.batch_size,
                shuffle=True, num_workers=args.num_workers,
                pin_memory=args.pin_mem,drop_last=True)

        # data embedding
 
        #构建模型
        model = Model(args.embedding_dim, args.hidden_size, int(parameter['vocab_size']), int(parameter['start_id']), int(parameter['end_id']), device)

        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('number of params (M):%.2f' % (n_parameters / 1.e6))
        
        #criterion = torch.nn.CrossEntropyLoss()
        criterion = torch.nn.NLLLoss(ignore_index=0)

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        os.makedirs(args.log_dir, exist_ok = True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
        loss_scaler = NativeScaler()
        
        misc.load_model(args=args, model_without_ddp=model, optimizer=optimizer, loss_scaler=loss_scaler)
        
        for epoch in range(args.start_epoch, args.epochs):
            
            print(f"Epoch {epoch}")
            print(f"Length of data_load_train is {len(data_loader_train)}")

            if epoch % 5 ==0:

                print("Evaluating ...")
                model.eval()

                test_stats = evaluate.evaluate(data_loader_val, model, device)
                print(f"Accuracy of the network on the {len(data_loader_val)} evaluate data:{test_stats['acc']:.3f}%")
                
                if log_writer is not None:
                    log_writer.add_scalar('perf/test_acc', test_stats['acc'], epoch)
                    log_writer.add_scalar('perf/test_loss', test_stats['loss'], epoch)


                model.train()
                
            print("Training ...")
            train_stats = train_one_epoch.train_one_epoch(
                model, criterion, data_loader_train,
                optimizer, device, epoch+1,
                loss_scaler,None,
                log_writer=log_writer,
                args=args)
                
            if args.output_dir:
                print("Save checkpoint ...")
                misc.save_model(
                    args=args, model=model, model_without_ddp=model,
                    optimizer=optimizer, loss_scaler=loss_scaler,
                    epoch=epoch)

    else:
        
        model = torch.load([args.output_dir/('checkpoint-%s.pth' % args.epochs)])

        # load other model
	#model = timm.create_model('resnet18', pretrained=True, 
        #                          num_classes=36, drop_rate=0.1,
        #                          drop_path_rate=0.1)

        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('number of trainable params (M): %.2f' % (n_parameters / 1.e6))
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        os.makedirs(args.log_dir, exist_ok=True)
        loss_scaler = NativeScaler()
        
        misc.load_model(args=args, model_without_ddp=model, optimizer=optimizer, loss_scaler=loss_scaler)
        model.eval()
        
        test_dataset = BERTDataset(args.corpus_path, protein)
        test_sample = test_dataset.__getitem__(0)
        sample = test_sample['bert_input'].unsqueeze(0).to(device, non_blocking=True)
        target = test_sample['bert_label'].unsqueeze(0).to(device, non_blocking=True)

        with torch.no_grad():
            predicted_ids = model.inference(sample)	    

        predicted_soft = torch.log(torch.softmax(predicted_ids, dim=-1))
        pred_argmax = predicted_soft.argmax(dim=-1).squeeze(0)

        equal_tensor = pred_argmax.eq(target)
        equal_tensor_padd = pred_argmax[equal_tensor] != protein['padd']
        equal_tensor_padd_index = pred_argmax[equal_tensor][equal_tensor_padd] != protein['unindex']

        pred_aa = equal_tensor_padd_index.sum().item()
        total_aa = math.floor(((sample != protein['padd']).sum().item() + 1e-6) * 0.15) 
        
        acc = pred_aa / total_aa

        with open(args.test_output, 'w') as f:
            print('sample input {samples.squeeze(0)}', file=f)
            print('sample label {target.squeeze(0)}', file=f)
            print('sample predi {pred_argmax.squeeze(0)}', file=f)
            print('sample accur {acc}', file=f)
	
        print(f"Test finish {arg.test_output}")
        times.sleep(0.5)




if __name__ =='__main__':
    args = get_args_parser()
    args = args.parse_args()
    
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    if args.log_dir:
        Path(args.log_dir).mkdir(parents=True, exist_ok=True)
      
    mode = 'train' # train/test
    
    if mode == 'train':
        main(args, mode=mode)
    else:
        #images = glob.glob('dataset_protein/test/*fa') # 做测试
        
        #for image in images:
        print('test ==============================================================')
        main(args, mode=mode)
