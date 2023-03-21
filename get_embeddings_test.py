import torch
import transformers
from transformers import LlamaForCausalLM, LlamaTokenizer

import pickle

def get_model():

    transformer_name = 'decapoda-research/llama-7b-hf'
    tokenizer = LlamaTokenizer.from_pretrained(transformer_name, use_fast=True, pad_token='<pad>')
    model = LlamaForCausalLM.from_pretrained(transformer_name)
    return model, tokenizer

model, tokenizer = get_model()

sentences = ['Anna loves fast food.',
        'Анна любит фаст-фуд.',
        'Anna ama la comida rápida.',
        'Where is Mona Lisa located?',
        'Где находится Мона Лиза?',
        '¿Dónde se encuentra la Mona Lisa?']

def feed_sentence(sentences):
    tokenizer_result = tokenizer(sentences, return_attention_mask=True, return_tensors='pt', padding=True)
    input_ids = tokenizer_result.input_ids
    attention_mask = tokenizer_result.attention_mask

    model_result = model(input_ids, attention_mask=attention_mask, return_dict=True)

    token_embeddings = model_result.last_hidden_state

    pooled = torch.max((token_embeddings * attention_mask.unsqueeze(-1)), axis=1)
    mean_pooled = token_embeddings.sum(axis=1) / attention_mask.sum(axis=-1).unsqueeze(-1)

    return pooled,mean_pooled

pooled, mean_pooled = [], []
for s in sentences:
    max_p, mean_p = feed_sentence([s])
    pooled.append(max_p)
    mean_pooled.append(mean_p)

with open(f'emb_maxpool.pkl', 'wb') as f:
    pickle.dump(pooled, f)

with open(f'emb_meanpool.pkl', 'wb') as f:
    pickle.dump(mean_pooled, f)
