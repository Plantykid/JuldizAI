import torch

class WordTokenizer:
    def __init__(self):
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        self.idx2word = {0: "<PAD>", 1: "<UNK>"}
        
    def fit(self, text):
        words = text.lower().split()
        for word in words:
            if word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word
                
    def encode(self, text):
        return [self.word2idx.get(w.lower(), 1) for w in text.split()]
        
    def decode(self, tokens):
        return " ".join([self.idx2word.get(t, "<UNK>") for t in tokens])

# Example of using AI KZ language
text_data = "JuldizAI — бұл қазақ тіліндегі жасанды интеллект моделі. ЖұлдызАИ сұрақтарға жауап береді."

tokenizer = WordTokenizer()
tokenizer.fit(text_data)

encoded = tokenizer.encode("JuldizAI жауап береді")
print("Закодированный текст:", encoded)
print("Декодированный обратно:", tokenizer.decode(encoded))
