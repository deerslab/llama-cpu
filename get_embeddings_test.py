import torch
import transformers
from transformers import LlamaForCausalLM, LlamaTokenizer
import pdb

import pickle

def get_model():

    transformer_name = 'decapoda-research/llama-7b-hf'
    tokenizer = LlamaTokenizer.from_pretrained(transformer_name, use_fast=True, pad_token='<pad>')
    model = LlamaForCausalLM.from_pretrained(transformer_name)
    return model, tokenizer

model, tokenizer = get_model()

sentences = ["McDonald's is a famous fast food chain.",
                'Anna loves fast food.',
                'she likes hamburgers!',
                'she hates hamburgers!',
                'Where is Mona Lisa located?',
                'the famous davinci painting is in paris']

def feed_sentence(sentences):
    tokenizer_result = tokenizer(sentences, return_attention_mask=True, return_tensors='pt', padding=True)
    input_ids = tokenizer_result.input_ids
    attention_mask = tokenizer_result.attention_mask

    model_result = model(input_ids, attention_mask=attention_mask, return_dict=True, output_hidden_states=True)
    pdb.set_trace()

    token_embeddings = model_result.hidden_states[-1]

    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


mean_pooled = []
for s in sentences:
    mean_p = feed_sentence([s])
    mean_pooled.append(mean_p)

with open(f'emb_meanpool.pkl', 'wb') as f:
    pickle.dump(mean_pooled, f)
