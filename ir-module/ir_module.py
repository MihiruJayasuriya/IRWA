import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DOC_FOLDER = os.path.join(BASE_DIR, "data", "documents")


class WaterPolicyIR:
    def __init__(self, doc_folder=DEFAULT_DOC_FOLDER):
        self.doc_folder = doc_folder
        self.doc_names = []
        self.documents = []
        self.vectorizer = None
        self.tfidf_matrix = None

        if os.path.exists(doc_folder):
            for fname in sorted(os.listdir(doc_folder)):
                if fname.endswith(".txt"):
                    path = os.path.join(doc_folder, fname)
                    with open(path, "r", encoding="utf-8") as f:
                        self.documents.append(f.read())
                        self.doc_names.append(fname)

        if self.documents:
            self.vectorizer = TfidfVectorizer(stop_words="english")
            self.tfidf_matrix = self.vectorizer.fit_transform(self.documents)

    def search(self, query: str, top_k: int = 2):
        if not self.documents or not self.vectorizer:
            return []

        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        ranked_indices = scores.argsort()[::-1][:top_k]

        results = []
        for idx in ranked_indices:
            if scores[idx] > 0:
                results.append({
                    "document": self.doc_names[idx],
                    "score": round(float(scores[idx]), 4),
                    "snippet": self.documents[idx][:250].replace("\n", " ").strip() + "...",
                })
        return results