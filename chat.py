import random
import json

import torch

from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
from cbf import get_rekomendasi  # import engine CBF

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r', encoding='utf-8') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size  = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words   = data['all_words']
tags        = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "El Tempat"


def get_response(msg):
    sentence = tokenize(msg)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:

                # ==========================
                # Intent Kuliner (CBF)
                # ==========================
                if 'places' in intent:

                    rekomendasi = get_rekomendasi(
                        kategori=tag,
                        top_n=3
                    )

                    # ==========================
                    # DEBUG HASIL REKOMENDASI
                    # ==========================
                    print("\n" + "=" * 80)
                    print("HASIL REKOMENDASI YANG DIKIRIM KE WEBSITE")
                    print("=" * 80)

                    for i, item in enumerate(rekomendasi, start=1):
                        print(
                            f"{i}. "
                            f"{item['nama']:<40} "
                            f"Similarity : {item['similarity']}"
                        )

                    print("=" * 80)

                    return {
                        "type": "places",
                        "intro": intent.get(
                            "intro",
                            "Berikut rekomendasinya:"
                        ),
                        "places": rekomendasi
                    }

                # ==========================
                # Intent Biasa
                # ==========================
                return {
                    "type": "text",
                    "message": random.choice(intent["responses"])
                }

    return {
        "type": "text",
        "message": "Maaf, saya tidak mengerti. Coba tanya soal makanan di daerah Pinang & Kunciran ya!"
    }