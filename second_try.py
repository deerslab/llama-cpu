from transformers import pipeline
import pickle

clf = pipeline(
    task = 'feature-extraction',
    model = 'deerslab/llama-7b-embeddings')#'shalomma/llama-7b-embeddings')

text = ['Anna loves fast food.',
        'Анна любит фаст-фуд.',
        'Anna ama la comida rápida.',
        'Where is Mona Lisa located?',
        'Где находится Мона Лиза?',
        '¿Dónde se encuentra la Mona Lisa?']

result = clf(text)

with open(f'emb_lang.pkl', 'wb') as f:
    pickle.dump(result, f)

#print(result)
