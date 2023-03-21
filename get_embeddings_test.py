import torch
import transformers
from transformers import AutoTokenizer, AutoModel

import pickle

transformer_name = 'llama'

tokenizer = AutoTokenizer.from_pretrained(transformer_name, use_fast=True)

model = AutoModel.from_pretrained(transformer_name)

sentences = ['Deep learning is difficult yet very rewarding.',
             'Deep learning is not easy.',
             'But is rewarding if done right.']
tokenizer_result = tokenizer(sentences, max_length=32, padding=True, return_attention_mask=True, return_tensors='pt')

input_ids = tokenizer_result.input_ids
attention_mask = tokenizer_result.attention_mask

model_result = model(input_ids, attention_mask=attention_mask, return_dict=True)

token_embeddings = model_result.last_hidden_state

pooled = torch.max((token_embeddings * attention_mask.unsqueeze(-1)), axis=1)
mean_pooled = token_embeddings.sum(axis=1) / attention_mask.sum(axis=-1).unsqueeze(-1)

with open(f'emb_maxpool.pkl', 'wb') as f:
    pickle.dump(pooled, f)

with open(f'emb_meanpool.pkl', 'wb') as f:
    pickle.dump(mean_pooled, f)
