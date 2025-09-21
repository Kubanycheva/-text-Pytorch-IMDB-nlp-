import torch
import torch.nn as nn
from torchtext.vocab import Vocab
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

class SentimentModel(nn.Module):
  def __init__(self, vocab_size, embed_dim=64, hidden_dim=128, output_dim=2):
    super().__init__()
    self.embedding = nn.Embedding(vocab_size, embed_dim)
    self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
    self.fc = nn.Linear(hidden_dim, output_dim)

  def forward(self, x):
    x = self.embedding(x)
    _, (hidden, _) = self.lstm(x)
    return self.fc(hidden[-1])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

vocab = torch.load('vocab_pth', weights_only=False)

model = SentimentModel(len(vocab)).to(device)
model.load_state_dict(torch.load('model_pth', map_location=device))
model.eval()

text_app = FastAPI()

class TextIn(BaseModel):
    text: str


def preprocess(text: str):
    tokens = text.lower().split()
    ids = [vocab[token] for token in tokens]
    return torch.tensor([ids], dtype=torch.int64, device=device)

@text_app.post('/predict')
def predict(item: TextIn):
    x = preprocess(item.text)
    with torch.no_grad():
        pred = model(x)
        label = torch.argmax(pred, dim=1).item()
    return {'label': 'positive' if label == 1 else 'negative'}

if __name__ == '__main__':
    uvicorn.run(text_app, host='127.0.0.1', port=8000)