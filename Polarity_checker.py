# Load model directly
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from scipy.special import softmax
from transformers import AutoTokenizer, AutoConfig

tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")

config = AutoConfig.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
text = "I had a girl friend, she dumped me last night. I am so sad now I want to die now, I can see any hope to live anymore. But My friend gave me a hope he introduced me to gyming. I started working out and I am feeling confident in myself"

encoded_input = tokenizer(text, return_tensors='pt')
output = model(**encoded_input)

scores = output[0][0].detach().numpy()
scores = softmax(scores)

ranking = np.argsort(scores)
ranking = ranking[::-1]
for i in range(scores.shape[0]):
    l = config.id2label[ranking[i]]
    s = scores[ranking[i]]
    print(f"{i+1}) {l} {np.round(float(s), 4)}")
