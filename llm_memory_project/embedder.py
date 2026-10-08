"""
embedder.py  -- MODULE 1 (Anjali): turns text into vectors.

This is the ONLY file that changes when we move from TF-IDF to real
embeddings. vector_store.py and vector_retriever.py stay the same.

  TfidfEmbedder        -> used now (offline, no model download)
  SentenceEmbedder     -> for the final review (semantic, understands meaning)
"""
from sklearn.feature_extraction.text import TfidfVectorizer


class TfidfEmbedder:
    """Lexical stand-in: a word only matches the exact same word."""

    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words="english")

    def fit(self, texts):
        self.vectorizer.fit(texts)
        return self

    def encode(self, texts):
        # dense rows: one fixed-length vector per text
        return self.vectorizer.transform(texts).toarray()

    def vocabulary(self):
        return list(self.vectorizer.get_feature_names_out())


class SentenceEmbedder:
    """
    Final-review version. Needs:  pip install sentence-transformers
    'all-MiniLM-L6-v2' gives a 384-number vector per sentence, and understands
    that "switched jobs" is related to "work".
    """

    def __init__(self, model_name="all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)

    def fit(self, texts):
        return self            # pretrained: nothing to fit

    def encode(self, texts):
        return self.model.encode(list(texts), normalize_embeddings=True)
