import torch
import torch.nn as nn
from dataset import WordTokenizer
from model import JuldizModel

# 1. Text's data training
raw_text = "Сәлем! Мен ЖұлдызАИ моделімін. Мен қазақ тілін үйреніп жатырмын. JuldizAI адамдарға көмектеседі."

tokenizer = WordTokenizer()
tokenizer.fit(raw_text)
vocab_size = len(tokenizer.word2idx)

# 2. Prepare of data enter (x) and use meanings (y)
tokens = tokenizer.encode(raw_text)
x_data = torch.tensor([tokens[:-1]]) # All words, except first one
y_data = torch.tensor([tokens[1:]])  # All words, except first one (Move to 1)

# 3. JuldizAI's Initialization
model = JuldizModel(vocab_size=vocab_size, embed_dim=32, hidden_dim=64)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 4. Training (500 эпох)
print("Бастау: Обучение JuldizAI...")
for epoch in range(500):
    optimizer.zero_grad()
    output, _ = model(x_data)
    
    # Reshape to find the mistakes
    loss = criterion(output.view(-1, vocab_size), y_data.view(-1))
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch + 1}/500 | Loss: {loss.item():.4f}")

# 5. Function of text generate
def generate_text(prompt, max_len=5):
    model.eval()
    words = prompt.split()
    input_tokens = tokenizer.encode(prompt)
    input_tensor = torch.tensor([input_tokens])
    hidden = None
    
    for _ in range(max_len):
        with torch.no_grad():
            output, hidden = model(input_tensor, hidden)
            # Taking the word with most common chances  (Greedy Search)
            next_token = torch.argmax(output[0, -1, :]).item()
            input_tokens.append(next_token)
            input_tensor = torch.tensor([[next_token]])
            
    return tokenizer.decode(input_tokens)

print("\n--- Результат генерации ---")
print(generate_text("Мен", max_len=4))
