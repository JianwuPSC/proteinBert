import torch.nn as nn
import math
import torch
import torch.nn.functional as F

class Seq2seqEncoder(nn.Module):

    """基于LSTM的encoder"""    
    def __init__(self, embedding_dim, hidden_size, vocab_size, device):
        super(Seq2seqEncoder, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.device = device

        self.embedding_table = torch.nn.Embedding(self.vocab_size, self.embedding_dim, device=self.device)
        self.lstm_layer = nn.LSTM(input_size = self.embedding_dim,
                                  hidden_size = self.hidden_size,
                                  batch_first = True,device=self.device)
        
    def forward(self, input_ids):
        self.input_ids = input_ids
        input_sequence = self.embedding_table(input_ids)
        output_states, (final_h, final_c) = self.lstm_layer(input_sequence)    
        return output_states, final_h


class Seq2seqEncoder_posi(nn.Module):

    """基于LSTM的encoder"""
    def __init__(self, embedding_dim, hidden_size, vocab_size, max_len, seg_size, dropout=0.1):
        super(Seq2seqEncoder_posi, self).__init__()

        self.embedding_table = BERTEmbedding(vocab_size, max_len, seg_size, embed_size, dropout=0.1)
        self.lstm_layer = nn.LSTM(input_size = embedding_dim,
                                  hidden_size = hidden_size,
                                  batch_first = True)

    def forward(self, input_ids, seg_ids):
        input_sequence = self.embedding_table(input_ids, seg_ids)
        output_states, (final_h, final_c) = self.lstm_layer(input_sequence)
        return output_states, final_h



class Seq2seqAttentionMechanism(nn.Module):
    
    """实现dot-product attention"""
    def __init__(self):
        super(Seq2seqAttentionMechanism,self).__init__()
        
    def forward(self, decoder_state_t, encoder_states):
        bs, source_length, hidden_size = encoder_states.shape #3D tensor
        decoder_state_t = decoder_state_t.unsqueeze(1)
        decoder_state_t = torch.tile(decoder_state_t, dims=(1,source_length,1)) # 3D tensor
        score = torch.sum(decoder_state_t * encoder_states, dim = -1) # [bs, source_length]
        atten_prob = F.softmax(score, dim = -1) # [bs, source_length]
        context = torch.sum(atten_prob.unsqueeze(-1) * encoder_states,1)
        return atten_prob,context


class Seq2seqDecoder(nn.Module):
    
    def __init__(self, embedding_dim, hidden_size, vocab_size, start_id, end_id, device):
        super(Seq2seqDecoder,self).__init__()
        
        self.lstm_cell = torch.nn.LSTMCell(embedding_dim, hidden_size, device=device)
        self.proj_layer = nn.Linear(hidden_size*2, vocab_size, device=device)
        self.attention_mechanism = Seq2seqAttentionMechanism()
        self.vocab_size = vocab_size
        self.embedding_table = torch.nn.Embedding(vocab_size, embedding_dim, device=device)
        self.start_id = start_id
        self.end_id = end_id
        
    def forward(self, shifted_target_ids, encoder_states):
        
        shifted_target = self.embedding_table(shifted_target_ids) #[bs, length, embedding_dim]
        bs, target_length, embedding_dim = shifted_target.shape
        bs, source_length, hidden_size = encoder_states.shape
        
        logits = torch.zeros(bs, target_length, self.vocab_size)
        probs = torch.zeros(bs, target_length, source_length)
        
        for t in range(target_length):

            decoder_input_t = shifted_target[:, t, :]
            
            if t == 0:
                h_t, c_t = self.lstm_cell(decoder_input_t)
            else :
                h_t, c_t = self.lstm_cell(decoder_input_t, (h_t, c_t))
            
            atten_prob, context = self.attention_mechanism(h_t, encoder_states)
            
            decoder_output = torch.cat((context, h_t), -1)
            logits[:, t, :] = self.proj_layer(decoder_output)
            probs[:, t, :] = atten_prob

        return probs, logits
    
    
    def inference(self, encoder_states):
        
        target_id = self.start_id
        h_t = None
        result = []
        
        while True:
            decoder_input_t = self.embedding_tab(target_id)
            
            if h_t is None:
                h_t, c_t =  self.lstm_cell(decoder_input_t)
            else:
                h_t, c_t = self.lstm_cell(decoder_input_t, (h_c, c_t))
            
            attn_prob, context = self.attention_mechanism(h_t, encoder_states)
            decoder_output =torch.cat(context, h_t)
            
            logits = self.proj_layer(decoder_output)
            
            target_id = torch.argmax(logits, -1)
            result.append(target_id)
            
            if torch.any(target_id == self.end_id): #解码终止条件
                print("stop decodeing !!")
                break
            
        predicted_ids = torch.stack(result, dim = 0)
            
        return predicted_ids



class Model(nn.Module):
    
     def __init__(self, embedding_dim, hidden_size, vocab_size, start_id, end_id, device):
         super(Model, self).__init__()

         self.encoder = Seq2seqEncoder(embedding_dim, hidden_size, vocab_size, device)
         self.decoder = Seq2seqDecoder(embedding_dim, hidden_size, vocab_size, 
                                       start_id, end_id, device)
         
     def forward(self, input_sequence_ids, shifted_target_ids):
         """traning"""    
         encoder_states, final_h = self.encoder(input_sequence_ids)
         probs, logits = self.decoder(shifted_target_ids, encoder_states)
             
         return probs, logits
         
         
     def inference(self, input_sequence_ids):             
         """inferencing"""
         encoder_states, final_h = self.encoder(input_sequence_ids)
         predicted_ids = self.decoder.inference(encoder_states)
             
         return predicted_ids
