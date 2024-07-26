import math
import torch
from collections import Iterable, Iterator
import sys

sys.path.append(r'/home/wuj/data/protein_design/Bert/protein_LSTM')
import MLM_LSTM_main


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float=0,
                    log_writer=None, args=None):

    model = model.to(device)
    model.train(True)
    
    print_freq = 2
    
    accum_iter = args.accum_iter    
    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.logdir))
        
    for data_iter_step, sample in enumerate(data_loader):
        
        samples = sample['bert_input'].to(device, non_blocking=True)
        targets = sample['bert_label'].to(device, non_blocking=True)
        
        probs, logits = model(samples,targets)
        logits = logits.to(device, non_blocking=True)

        #optimizer = torch.optim.AdamW(model.parameters())
        # warmup_lr = args.lr*(min(1.0, epoch/2.))
        warmup_lr = args.lr
        optimizer.param_groups[0]["lr"] = warmup_lr

        loss = criterion(torch.log(torch.softmax(logits, dim=-1)).transpose(1, 2), targets)
        loss /=accum_iter
        
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=((data_iter_step + 1) % accum_iter) ==0)
        
        loss_value = loss.item()
        
        if ((data_iter_step + 1) % accum_iter) == 0:
            optimizer.zero_grad()
        
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)
        
        
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value, epoch_1000x)
            log_writer.add_scalar('lr', warmup_lr, epoch_1000x)
            print(f"Epoch: {epoch}, Step: {data_iter_step}, Loss: {loss}, Lr: {warmup_lr}")

        #loss.backward()
        #optimizer.step()
