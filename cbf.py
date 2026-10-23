import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


with open('intents.json', 'r', encoding='utf-8') as f:
    intents = json.load(f)

all_places = []

for intent in intents['intents']:
    if 'places' in intent:
        kategori = intent['tag']

        for place in intent['places']:

            dokumen = f"{kategori} {place['nama']} {place['alamat']} {place['harga']}"

            all_places.append({
                "kategori": kategori,
                "nama": place['nama'],
                "alamat": place['alamat'],
                "rating": place['rating'],
                "harga": place['harga'],
                "maps": place['maps'],
                "dokumen": dokumen
            })


# ==========================
# TF-IDF
# ==========================
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(
    [p['dokumen'] for p in all_places]
)


# ==========================
# Cosine Similarity
# ==========================
cosine_sim_matrix = cosine_similarity(
    tfidf_matrix,
    tfidf_matrix
)


# ==========================
# Fungsi Rekomendasi
# ==========================
def get_rekomendasi(kategori, top_n=3):

    kategori_lower = kategori.lower()

    idx_kategori = [
        i for i, p in enumerate(all_places)
        if p['kategori'].lower() == kategori_lower
    ]

    if not idx_kategori:
        return []

    hasil = []

    for idx in idx_kategori:

        sim_scores = [
            cosine_sim_matrix[idx][j]
            for j in idx_kategori
            if j != idx
        ]

        avg_sim = np.mean(sim_scores) if sim_scores else 0.0

        hasil.append({
            "nama": all_places[idx]["nama"],
            "alamat": all_places[idx]["alamat"],
            "rating": all_places[idx]["rating"],
            "harga": all_places[idx]["harga"],
            "maps": all_places[idx]["maps"],
            "similarity": round(float(avg_sim), 4)
        })

    hasil.sort(
        key=lambda x: (x["similarity"], x["rating"]),
        reverse=True
    )

    return hasil[:top_n]


# ==========================
# Matriks Similarity
# ==========================
def get_similarity_matrix(kategori):

    kategori_lower = kategori.lower()

    idx_kategori = [
        i for i, p in enumerate(all_places)
        if p["kategori"].lower() == kategori_lower
    ]

    if not idx_kategori:
        return {}

    nama_list = [
        all_places[i]["nama"]
        for i in idx_kategori
    ]

    matrix = {}

    for i, idx_i in enumerate(idx_kategori):

        matrix[nama_list[i]] = {}

        for j, idx_j in enumerate(idx_kategori):

            score = cosine_sim_matrix[idx_i][idx_j]

            matrix[nama_list[i]][nama_list[j]] = round(
                float(score),
                4
            )

    return matrix


if __name__ == "__main__":

    kategori = "mie ayam"      # Ganti sesuai kategori

    matrix = get_similarity_matrix(kategori)

    print("\n")
    print("=" * 100)
    print(f"MATRIKS COSINE SIMILARITY - {kategori.upper()}")
    print("=" * 100)

    for asal, tujuan in matrix.items():

        print(f"\n{asal}")

        for nama, score in tujuan.items():

            print(f"   -> {nama:<40} : {score}")

    print("\n")
    print("=" * 100)
    print("HASIL RATA-RATA COSINE SIMILARITY")
    print("=" * 100)

    hasil = get_rekomendasi(kategori, top_n=100)

    print(
        f"{'No':<4}"
        f"{'Nama Kuliner':<40}"
        f"{'Similarity':<15}"
        f"{'Persentase'}"
    )

    print("-" * 100)

    for i, item in enumerate(hasil, start=1):

        persen = item["similarity"] * 100

        print(
            f"{i:<4}"
            f"{item['nama']:<40}"
            f"{item['similarity']:<15}"
            f"{persen:.2f}%"
        )

    print("=" * 100)