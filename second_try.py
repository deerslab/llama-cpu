from transformers import pipeline
import pickle

clf = pipeline(
    task = 'feature-extraction',
    model = 'deerslab/llama-7b-embeddings')#'shalomma/llama-7b-embeddings')

text = ['McDonalds is a famous fast food chain.',
        'Anna loves fast food.',
        'she likes hamburgers!',
        'she hates hamburgers!',
        'Where is Mona Lisa located?',
        'the famous davinci painting is in paris']

result = clf(text)

with open(f'emb_lang.pkl', 'wb') as f:
    pickle.dump(result, f)

#print(result)
