from tranformers import pipeline
classifier = pipeline("sentiment-analysis")
res = classifier("I love using transformers!")
print(res)