import torch
import torch.nn as nn
import sys
import math
sys.path.append(r'/home/wuj/data/protein_design/Bert/protein_LSTM')
import MLM_LSTM_main
import util.misc as misc
from model.MLM_dataset import protein

@torch.no_grad()

def evaluate(data_loader, model, device):

    criterion = nn.NLLLoss(ignore_index=0)
#    criterion = torch.nn.CrossEntropyLoss()
    
    metric_logger = misc.MetricLogger(delimiter=" ")
    header = 'Test:'
    
    # switch to evalution model
    model = model.to(device)
    model.eval()
    
    for batch in metric_logger.log_every(data_loader, 10, header):
        sample = batch['bert_input']
        target = batch['bert_label']
        samples = sample.to(device, non_blocking=True)
        targets = target.to(device, non_blocking=True)
        
        # compute output
        with torch.no_grad():
            probs, logits = model(samples,targets)
            logits = logits.to(device, non_blocking=True)
            loss = criterion(torch.log(torch.softmax(logits, dim=-1)).transpose(1, 2), targets)

        loss_logits = torch.log(torch.softmax(logits), dim=-1)
        logits_argmax = loss_logits.argmax(dim=-1)

        equal_tensor = logits_argmax.eq(targets)
        equal_tensor_padd = logits_argmax[equal_tensor] != protein['padd']
        equal_tensor_padd_index = logits_argmax[equal_tensor][equal_tensor_padd] != protein['unindex']

        logits_aa = equal_tensor_padd_index.sum().item()
        total_aa = math.floor(((samples != protein['padd']).sum().item() + 1e-6) * 0.15)
        print(f'Evaluate =========================  logits_aa:{logits_aa} total_aa:{total_aa}')
	
        if logits_aa >0 :
            logits_aa = logits_aa
        else:
            logits_aa = 0
	
        acc = (logits_aa / total_aa)*100
        
        batch_size = sample.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc'].update(acc , n=batch_size)

    # gather the states from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc, losses=metric_logger.loss))
    
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
