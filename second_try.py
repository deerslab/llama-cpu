from transformers import pipeline

clf = pipeline(
    task = 'feature-extraction',
    model = 'deerslab/llama-7b-embeddings')#'shalomma/llama-7b-embeddings')

text = ["McDonald's is a famous fast food chain.",
        'Anna loves fast food.',]

result = clf(text)

print(result)
