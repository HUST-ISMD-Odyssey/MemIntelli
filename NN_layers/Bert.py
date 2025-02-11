# -*- coding:utf-8 -*-
# @File  : ResNet.py
# @Author: Zhou
# @Date  : 2024/4/1

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url
from typing import Dict, Any
from NN_layers import Conv2dMem, LinearMem, SliceMethod

cfgs: Dict[str, Dict[str, Any]] = {
    'bert_tiny': {'hidden_size': 128, 'num_layers': 2, 'num_heads': 2, 'intermediate_size': 512},
    'bert_mini': {'hidden_size': 256, 'num_layers': 4, 'num_heads': 4, 'intermediate_size': 1024},
    'bert_small': {'hidden_size': 512, 'num_layers': 4, 'num_heads': 8, 'intermediate_size': 2048},
    'bert_medium': {'hidden_size': 512, 'num_layers': 8, 'num_heads': 8, 'intermediate_size': 2048},
    'bert_base': {'hidden_size': 768, 'num_layers': 12, 'num_heads': 12, 'intermediate_size': 3072},
    'bert-base-uncased': {'hidden_size': 768, 'num_layers': 12, 'num_heads': 12, 'intermediate_size': 3072},
}

class BertSelfAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads):
        super().__init__()
        assert hidden_size % num_attention_heads == 0
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, hidden_states, attention_mask=None):
        batch_size, seq_length, _ = hidden_states.size()
        query_layer = self.query(hidden_states).view(batch_size, seq_length, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        key_layer = self.key(hidden_states).view(batch_size, seq_length, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        value_layer = self.value(hidden_states).view(batch_size, seq_length, self.num_attention_heads, self.attention_head_size).transpose(1, 2)

        # 计算attention分数
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2)) / (self.attention_head_size ** 0.5)
        
        # 扩展并添加attention_mask，使其形状与attention_scores相匹配
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # 扩展到 (batch_size, 1, 1, seq_length)
            attention_scores = attention_scores + attention_mask  # 广播操作

        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer).transpose(1, 2).contiguous().view(batch_size, seq_length, self.all_head_size)
        return context_layer


class BertSelfOutput(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(0.1)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class BertAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads):
        super().__init__()
        self.self = BertSelfAttention(hidden_size, num_attention_heads)
        self.output = BertSelfOutput(hidden_size)

    def forward(self, input_tensor, attention_mask=None):
        self_output = self.self(input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output

class BertIntermediate(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, intermediate_size)
        self.intermediate_act_fn = nn.GELU()

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states

class BertOutput(nn.Module):
    def __init__(self, intermediate_size, hidden_size):
        super().__init__()
        self.dense = nn.Linear(intermediate_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(0.1)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class BertLayer(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, intermediate_size):
        super().__init__()
        self.attention = BertAttention(hidden_size, num_attention_heads)
        self.intermediate = BertIntermediate(hidden_size, intermediate_size)
        self.output = BertOutput(intermediate_size, hidden_size)

    def forward(self, hidden_states, attention_mask=None):
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class BertEmbeddings(nn.Module):
    def __init__(self, hidden_size, max_position_embeddings=512, vocab_size=30522):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.token_type_embeddings = nn.Embedding(2, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(0.1)
        self.register_buffer("position_ids", torch.arange(max_position_embeddings).expand((1, -1)))

    def forward(self, input_ids, token_type_ids=None):
        seq_length = input_ids.size(1)
        position_ids = self.position_ids[:, :seq_length]
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class BertEncoder(nn.Module):
    def __init__(self, num_hidden_layers, hidden_size, num_attention_heads, intermediate_size):
        super().__init__()
        self.layer = nn.ModuleList([BertLayer(hidden_size, num_attention_heads, intermediate_size) for _ in range(num_hidden_layers)])

    def forward(self, hidden_states, attention_mask=None):
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
        return hidden_states

class BertPooler(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class CustomBertModel(nn.Module):
    def __init__(self, vocab_size=30522, hidden_size=768, num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072):
        super().__init__()
        self.embeddings = BertEmbeddings(hidden_size=hidden_size)
        self.encoder = BertEncoder(num_hidden_layers, hidden_size, num_attention_heads, intermediate_size)
        self.pooler = BertPooler(hidden_size)

    def forward(self, input_ids, attention_mask=None):
        embedding_output = self.embeddings(input_ids)
        encoder_output = self.encoder(embedding_output, attention_mask)
        pooled_output = self.pooler(encoder_output)
        return encoder_output, pooled_output

class BertForSequenceClassification(nn.Module):
    def __init__(self, num_labels=2, vocab_size=30522, hidden_size=768, num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072):
        super().__init__()
        self.bert = CustomBertModel(vocab_size=vocab_size, hidden_size=hidden_size, num_hidden_layers=num_hidden_layers, num_attention_heads=num_attention_heads, intermediate_size=intermediate_size)
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.num_labels = num_labels

    def forward(self, input_ids, attention_mask=None, labels=None):
        encoder_output, pooled_output = self.bert(input_ids, attention_mask)
        logits = self.classifier(pooled_output)
        
        if labels is not None:
            if self.num_labels == 1:
                # 回归任务
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1).float())
            else:
                # 分类任务
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss, logits
        return logits

def get_model_for_task_glue(task_name,hidden_size=128,num_hidden_layers=2,num_attention_heads=2,intermediate_size=512):
    if task_name in ["cola", "sst2", "rte", "wnli"]:
        model = BertForSequenceClassification(num_labels=2,hidden_size=hidden_size,num_hidden_layers=num_hidden_layers,num_attention_heads=num_attention_heads,intermediate_size=intermediate_size)
    elif task_name in ["mrpc", "qqp", "qnli", "mnli", "mnli_matched", "mnli_mismatched"]:
        model = BertForSequenceClassification(num_labels=3,hidden_size=hidden_size,num_hidden_layers=num_hidden_layers,num_attention_heads=num_attention_heads,intermediate_size=intermediate_size)
    elif task_name == "stsb":
        model = BertForSequenceClassification(num_labels=1,hidden_size=hidden_size,num_hidden_layers=num_hidden_layers,num_attention_heads=num_attention_heads,intermediate_size=intermediate_size)
    else:
        raise ValueError(f"Task {task_name} is not supported.")
    return model
#---------------------------------------------------BertForMaskedLM
class BertPredictionHeadTransform(nn.Module):
    def __init__(self, hidden_size, layer_norm_eps=1e-12):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.transform_act_fn = nn.GELU()
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states

class BertLMPredictionHead(nn.Module):
    def __init__(self, hidden_size, vocab_size):
        super().__init__()
        self.transform = BertPredictionHeadTransform(hidden_size)
        # The decoder weight can be shared with input embeddings
        self.decoder = nn.Linear(hidden_size, vocab_size)
        self.bias = nn.Parameter(torch.zeros(vocab_size))

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states

class BertPreTrainingHeads(nn.Module):
    def __init__(self, hidden_size, vocab_size):
        super().__init__()
        self.predictions = BertLMPredictionHead(hidden_size, vocab_size)
        self.seq_relationship = nn.Linear(hidden_size, 2)

    def forward(self, sequence_output, pooled_output):
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score

class BertForMaskedLM(nn.Module):
    def __init__(self, vocab_size=30522, hidden_size=768, num_hidden_layers=12, 
                 num_attention_heads=12, intermediate_size=3072):
        super().__init__()
        self.bert = CustomBertModel(vocab_size=vocab_size, hidden_size=hidden_size,
                                  num_hidden_layers=num_hidden_layers,
                                  num_attention_heads=num_attention_heads,
                                  intermediate_size=intermediate_size)
        self.cls = BertPreTrainingHeads(hidden_size, vocab_size)
        
        # Initialize and tie weights between embedding and decoder if needed
        self.tie_weights()
        
    def tie_weights(self):
        """Tie the weights between the input embeddings and the decoder."""
        self.cls.predictions.decoder.weight = self.bert.embeddings.word_embeddings.weight
        
    def forward(self, input_ids, attention_mask=None, labels=None):
        encoder_output, pooled_output = self.bert(input_ids, attention_mask)
        prediction_scores, seq_relationship_score = self.cls(encoder_output, pooled_output)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(prediction_scores.view(-1, prediction_scores.size(-1)), 
                                    labels.view(-1))
            loss = masked_lm_loss
            
        return loss, prediction_scores

def bert_zoo(pretrained=False, model_name='bert_base',task_type="sst2", **kwargs):
    if model_name not in cfgs:
        raise ValueError(f"Model name {model_name} not found in configs. Available models are: {list(cfgs.keys())}")
    cfg = cfgs[model_name]

    if task_type==1:
        model = BertForMaskedLM(hidden_size=cfg['hidden_size'],num_hidden_layers=cfg['num_layers'],num_attention_heads=cfg['num_heads'],intermediate_size=cfg['intermediate_size'])
    else:
        model = get_model_for_task_glue(task_type,hidden_size=cfg['hidden_size'],num_hidden_layers=cfg['num_layers'],num_attention_heads=cfg['num_heads'],intermediate_size=cfg['intermediate_size'])
    
    if pretrained:
        state_dict = torch.load(f"D:/work/deep_learning/pretrained_model/Bert/{model_name}.bin")
        # 创建一个新的字典，用于重命名参数
        try:
            model.load_state_dict(state_dict, strict=True)
            print("Model loaded successfully with strict=True.")
        except RuntimeError as e:
            print(f"Error loading model with strict=True: {e}")
            print("Retrying with strict=False...")
            model.load_state_dict(state_dict, strict=False)
            print("Model loaded successfully with strict=False.")
            
    return model

#---------------------------------------------------BertForMaskedLMmem
class BertSelfAttentionMem(nn.Module):
    def __init__(self, engine, hidden_size, num_attention_heads, input_slice=None, weight_slice=None, device="cuda:0", bw_e=None, input_en=None, dbfp_en=None):
        super().__init__()
        assert hidden_size % num_attention_heads == 0
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = LinearMem(engine, hidden_size, self.all_head_size, input_sli_med=input_slice, weight_sli_med=weight_slice, device=device, bw_e=bw_e, input_en=input_en, dbfp_en=dbfp_en)
        self.key = LinearMem(engine, hidden_size, self.all_head_size, input_sli_med=input_slice, weight_sli_med=weight_slice, device=device, bw_e=bw_e, input_en=input_en, dbfp_en=dbfp_en)
        self.value = LinearMem(engine, hidden_size, self.all_head_size, input_sli_med=input_slice, weight_sli_med=weight_slice, device=device, bw_e=bw_e, input_en=input_en, dbfp_en=dbfp_en)
        self.dropout = nn.Dropout(0.1)

    def forward(self, hidden_states, attention_mask=None):
        batch_size, seq_length, _ = hidden_states.size()
        query_layer = self.query(hidden_states).view(batch_size, seq_length, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        key_layer = self.key(hidden_states).view(batch_size, seq_length, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        value_layer = self.value(hidden_states).view(batch_size, seq_length, self.num_attention_heads, self.attention_head_size).transpose(1, 2)

        # 计算attention分数
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2)) / (self.attention_head_size ** 0.5)
        
        # 扩展并添加attention_mask，使其形状与attention_scores相匹配
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # 扩展到 (batch_size, 1, 1, seq_length)
            attention_scores = attention_scores + attention_mask  # 广播操作

        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer).transpose(1, 2).contiguous().view(batch_size, seq_length, self.all_head_size)
        return context_layer


class BertSelfOutputMem(nn.Module):
    def __init__(self, engine, hidden_size, input_slice=None, weight_slice=None, device="cuda:0", bw_e=None, input_en=None, dbfp_en=None):
        super().__init__()
        self.dense = LinearMem(engine, hidden_size, hidden_size, input_sli_med=input_slice, weight_sli_med=weight_slice, device=device, bw_e=bw_e, input_en=input_en, dbfp_en=dbfp_en)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(0.1)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class BertAttentionMem(nn.Module):
    def __init__(self, engine, hidden_size, num_attention_heads, input_slice=None, weight_slice=None, device="cuda:0", bw_e=None, input_en=None, dbfp_en=None):
        super().__init__()
        self.self = BertSelfAttentionMem(engine, hidden_size, num_attention_heads, input_slice, weight_slice, device, bw_e, input_en, dbfp_en)
        self.output = BertSelfOutputMem(engine, hidden_size, input_slice, weight_slice, device, bw_e, input_en, dbfp_en)

    def forward(self, input_tensor, attention_mask=None):
        self_output = self.self(input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output

class BertIntermediateMem(nn.Module):
    def __init__(self, engine, hidden_size, intermediate_size, input_slice=None, weight_slice=None, device="cuda:0", bw_e=None, input_en=None, dbfp_en=None):
        super().__init__()
        self.dense = LinearMem(engine, hidden_size, intermediate_size, input_sli_med=input_slice, weight_sli_med=weight_slice, device=device, bw_e=bw_e, input_en=input_en, dbfp_en=dbfp_en)
        self.intermediate_act_fn = nn.GELU()

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states

class BertOutputMem(nn.Module):
    def __init__(self, engine, intermediate_size, hidden_size, input_slice=None, weight_slice=None, device="cuda:0", bw_e=None, input_en=None, dbfp_en=None):
        super().__init__()
        self.dense = LinearMem(engine, intermediate_size, hidden_size, input_sli_med=input_slice, weight_sli_med=weight_slice, device=device, bw_e=bw_e, input_en=input_en, dbfp_en=dbfp_en)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(0.1)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class BertLayerMem(nn.Module):
    def __init__(self, engine, hidden_size, num_attention_heads, intermediate_size, input_slice=None, weight_slice=None, device="cuda:0", bw_e=None, input_en=None, dbfp_en=None):
        super().__init__()
        self.attention = BertAttentionMem(engine, hidden_size, num_attention_heads, input_slice, weight_slice, device, bw_e, input_en, dbfp_en)
        self.intermediate = BertIntermediateMem(engine, hidden_size, intermediate_size, input_slice, weight_slice, device, bw_e, input_en, dbfp_en)
        self.output = BertOutputMem(engine, intermediate_size, hidden_size, input_slice, weight_slice, device, bw_e, input_en, dbfp_en)

    def forward(self, hidden_states, attention_mask=None):
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class BertEmbeddings(nn.Module):
    def __init__(self, hidden_size, max_position_embeddings=512, vocab_size=30522):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.token_type_embeddings = nn.Embedding(2, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(0.1)
        self.register_buffer("position_ids", torch.arange(max_position_embeddings).expand((1, -1)))

    def forward(self, input_ids, token_type_ids=None):
        seq_length = input_ids.size(1)
        position_ids = self.position_ids[:, :seq_length]
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class BertEncoderMem(nn.Module):
    def __init__(self, engine, num_hidden_layers, hidden_size, num_attention_heads, intermediate_size, input_slice=None, weight_slice=None, device="cuda:0", bw_e=None, input_en=None, dbfp_en=None):
        super().__init__()
        self.layer = nn.ModuleList([BertLayerMem(engine, hidden_size, num_attention_heads, intermediate_size, input_slice, weight_slice, device, bw_e, input_en, dbfp_en) for _ in range(num_hidden_layers)])

    def forward(self, hidden_states, attention_mask=None):
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
        return hidden_states

class BertPoolerMem(nn.Module):
    def __init__(self, engine, hidden_size, input_slice=None, weight_slice=None, device="cuda:0", bw_e=None, input_en=None, dbfp_en=None):
        super().__init__()
        self.dense = LinearMem(engine, hidden_size, hidden_size, input_sli_med=input_slice, weight_sli_med=weight_slice, device=device, bw_e=bw_e, input_en=input_en, dbfp_en=dbfp_en)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class CustomBertModelMem(nn.Module):
    def __init__(self, engine, vocab_size=30522, hidden_size=768, num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072, input_slice=None, weight_slice=None, device="cuda:0", bw_e=None, input_en=None, dbfp_en=None):
        super().__init__()
        self.embeddings = BertEmbeddings(hidden_size=hidden_size)
        self.encoder = BertEncoderMem(engine,num_hidden_layers, hidden_size, num_attention_heads, intermediate_size, input_slice, weight_slice, device, bw_e, input_en, dbfp_en)
        self.pooler = BertPoolerMem(engine, hidden_size, input_slice, weight_slice, device, bw_e, input_en, dbfp_en)

    def forward(self, input_ids, attention_mask=None):
        embedding_output = self.embeddings(input_ids)
        encoder_output = self.encoder(embedding_output, attention_mask)
        pooled_output = self.pooler(encoder_output)
        return encoder_output, pooled_output

class BertForSequenceClassificationMem(nn.Module):
    def __init__(self, engine, num_labels=2, vocab_size=30522, hidden_size=768, num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072, input_slice=None, weight_slice=None, device="cuda:0", bw_e=None, input_en=None, dbfp_en=None):
        super().__init__()
        self.bert = CustomBertModelMem(engine, vocab_size=vocab_size, hidden_size=hidden_size, num_hidden_layers=num_hidden_layers, num_attention_heads=num_attention_heads, intermediate_size=intermediate_size, input_slice=input_slice, weight_slice=weight_slice, device=device, bw_e=bw_e, input_en=input_en, dbfp_en=dbfp_en)
        self.classifier = LinearMem(engine, hidden_size, num_labels, input_sli_med=input_slice, weight_sli_med=weight_slice, device=device, bw_e=bw_e, input_en=input_en, dbfp_en=dbfp_en)
        self.num_labels = num_labels

    def forward(self, input_ids, attention_mask=None, labels=None):
        encoder_output, pooled_output = self.bert(input_ids, attention_mask)
        logits = self.classifier(pooled_output)
        
        if labels is not None:
            if self.num_labels == 1:
                # 回归任务
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1).float())
            else:
                # 分类任务
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss, logits
        return logits
    
    def update_weight(self):
        for m in self.modules():
            if isinstance(m, LinearMem) or isinstance(m, Conv2dMem):
                m.update_weight()

def get_model_for_task_glueMem(engine,task_name,vocab_size=30522,hidden_size=128,num_hidden_layers=2,num_attention_heads=2,intermediate_size=512,input_slice=None, weight_slice=None, device="cuda:0", bw_e=None, input_en=None, dbfp_en=None):
    if task_name in ["cola", "sst2", "rte", "wnli"]:
        model = BertForSequenceClassificationMem(engine=engine,num_labels=2,vocab_size=vocab_size,hidden_size=hidden_size,num_hidden_layers=num_hidden_layers,num_attention_heads=num_attention_heads,intermediate_size=intermediate_size,input_slice=input_slice, weight_slice=weight_slice, device=device, bw_e=bw_e, input_en=input_en, dbfp_en=dbfp_en)
    elif task_name in ["mrpc", "qqp", "qnli", "mnli", "mnli_matched", "mnli_mismatched"]:
        model = BertForSequenceClassificationMem(engine=engine,num_labels=3,vocab_size=vocab_size,hidden_size=hidden_size,num_hidden_layers=num_hidden_layers,num_attention_heads=num_attention_heads,intermediate_size=intermediate_size,input_slice=input_slice, weight_slice=weight_slice, device=device, bw_e=bw_e, input_en=input_en, dbfp_en=dbfp_en)
    elif task_name == "stsb":
        model = BertForSequenceClassificationMem(engine=engine,num_labels=1,vocab_size=vocab_size,hidden_size=hidden_size,num_hidden_layers=num_hidden_layers,num_attention_heads=num_attention_heads,intermediate_size=intermediate_size,input_slice=input_slice, weight_slice=weight_slice, device=device, bw_e=bw_e, input_en=input_en, dbfp_en=dbfp_en)
    else:
        raise ValueError(f"Task {task_name} is not supported.")
    return model

#---------------------------------------------------BertForMaskedLM
class BertPredictionHeadTransformMem(nn.Module):
    def __init__(self, engine, hidden_size, layer_norm_eps=1e-12, input_slice=None, weight_slice=None, device=None, bw_e=None, input_en=False, dbfp_en=False):
        super().__init__()
        self.dense = LinearMem(engine, hidden_size, hidden_size, input_sli_med=input_slice, weight_sli_med=weight_slice, device=device, bw_e=bw_e, input_en=input_en, dbfp_en=dbfp_en)
        self.transform_act_fn = nn.GELU()
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states

class BertLMPredictionHeadMem(nn.Module):
    def __init__(self, engine,hidden_size, vocab_size, input_slice=None, weight_slice=None, device=None, bw_e=None, input_en=False, dbfp_en=False):
        super().__init__()
        self.transform = BertPredictionHeadTransformMem(engine=engine,hidden_size= hidden_size, input_slice=input_slice, weight_slice=weight_slice, device=device, bw_e=bw_e, input_en=input_en, dbfp_en=dbfp_en)
        # The decoder weight can be shared with input embeddings
        self.decoder = LinearMem(engine, hidden_size, vocab_size, input_sli_med=input_slice, weight_sli_med=weight_slice, device=device, bw_e=bw_e, input_en=input_en, dbfp_en=dbfp_en)
        self.bias = nn.Parameter(torch.zeros(vocab_size))

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states

class BertPreTrainingHeadsMem(nn.Module):
    def __init__(self, engine,hidden_size, vocab_size, input_slice=None, weight_slice=None, device=None, bw_e=None, input_en=False, dbfp_en=False):
        super().__init__()
        self.predictions = BertLMPredictionHeadMem(engine=engine, hidden_size=hidden_size, vocab_size=vocab_size, input_slice=input_slice, weight_slice=weight_slice, device=device, bw_e=bw_e, input_en=input_en, dbfp_en=dbfp_en)
        self.seq_relationship = LinearMem(engine, hidden_size, 2, input_sli_med=input_slice, weight_sli_med=weight_slice, device=device, bw_e=bw_e, input_en=input_en, dbfp_en=dbfp_en)

    def forward(self, sequence_output, pooled_output):
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score

class BertForMaskedLMMem(nn.Module):
    def __init__(self,engine, vocab_size=30522, hidden_size=768, num_hidden_layers=12, 
                 num_attention_heads=12, intermediate_size=3072, input_slice=None, weight_slice=None, device=None, bw_e=None, input_en=False, dbfp_en=False):
        super().__init__()
        self.bert = CustomBertModelMem(engine=engine,vocab_size=vocab_size, hidden_size=hidden_size,
                                  num_hidden_layers=num_hidden_layers,
                                  num_attention_heads=num_attention_heads,
                                  intermediate_size=intermediate_size, input_slice=input_slice, weight_slice=weight_slice, device=device, bw_e=bw_e, input_en=input_en, dbfp_en=dbfp_en)
        self.cls = BertPreTrainingHeadsMem(engine=engine,hidden_size=hidden_size, vocab_size=vocab_size, input_slice=input_slice, weight_slice=weight_slice, device=device, bw_e=bw_e, input_en=input_en, dbfp_en=dbfp_en)
        
        # Initialize and tie weights between embedding and decoder if needed
        self.tie_weights()
        
    def tie_weights(self):
        """Tie the weights between the input embeddings and the decoder."""
        self.cls.predictions.decoder.weight = self.bert.embeddings.word_embeddings.weight
        
    def forward(self, input_ids, attention_mask=None, labels=None):
        encoder_output, pooled_output = self.bert(input_ids, attention_mask)
        prediction_scores, seq_relationship_score = self.cls(encoder_output, pooled_output)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(prediction_scores.view(-1, prediction_scores.size(-1)), 
                                    labels.view(-1))
            loss = masked_lm_loss
            
        return loss, prediction_scores
    
    def update_weight(self):
        for m in self.modules():
            if isinstance(m, LinearMem) or isinstance(m, Conv2dMem):
                m.update_weight()

def bert_zooMem(engine, input_slice, weight_slice, device, bw_e=None, input_en=False, dbfp_en=False,pretrained=False, model_name='bert_base',task_type="sst2", **kwargs):
    if model_name not in cfgs:
        raise ValueError(f"Model name {model_name} not found in configs. Available models are: {list(cfgs.keys())}")
    cfg = cfgs[model_name]

    if task_type==1:
        model = BertForMaskedLMMem(engine=engine,hidden_size=cfg['hidden_size'],num_hidden_layers=cfg['num_layers'],num_attention_heads=cfg['num_heads'],intermediate_size=cfg['intermediate_size'],input_slice=input_slice, weight_slice=weight_slice, device=device, bw_e=bw_e, input_en=input_en, dbfp_en=dbfp_en)
    else:
        model = get_model_for_task_glueMem(engine,task_type,hidden_size=cfg['hidden_size'],num_hidden_layers=cfg['num_layers'],num_attention_heads=cfg['num_heads'],intermediate_size=cfg['intermediate_size'],input_slice=input_slice, weight_slice=weight_slice, device=device, bw_e=bw_e, input_en=input_en, dbfp_en=dbfp_en)
        
    if pretrained:
        state_dict = torch.load(f"D:/work/deep_learning/pretrained_model/Bert/{model_name}.bin")
        # 创建一个新的字典，用于重命名参数
        try:
            model.load_state_dict(state_dict, strict=True)
            print("Model loaded successfully with strict=True.")
        except RuntimeError as e:
            print(f"Error loading model with strict=True: {e}")
            print("Retrying with strict=False...")
            model.load_state_dict(state_dict, strict=False)
            print("Model loaded successfully with strict=False.")
            
    return model
