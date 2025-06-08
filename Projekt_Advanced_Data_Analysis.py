import pandas as pd
import re
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel
import matplotlib.pyplot as plt
import seaborn as sns

# 1. spaCy-Modell laden
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

# 1a. Stop-Wort-Liste
custom_stopwords = {
    # Höflichkeitsfloskeln
    "please", "thank", "thanks", "regards", "sincerely", "dear", "sir", "madam",

    # Standard-Stopwords
    "would", "could", "also", "even", "really", "much", "may", "might", "one", "two",
    "get", "got", "make", "made", "use", "used", "see", "said", "tell", "told", "know",

    # Füllwörter
    "like", "just", "also", "okay", "well", "right", "thankyou",

    # Fachlich irrelevante, aber häufige Begriffe in Beschwerden
    "consumer", "customer", "company", "complaint", "case", "report", "file", "issue",
    "product", "service", "loan", "account", "card", "payment", "credit", "bank", "debt",
    "application", "information", "provide", "provided", "attached", "contact", "phone",
    "email", "date", "time", "refer", "referenced", "issue", "matter", "transaction",

    # Platzhalter / Maskierungen
    "xxx", "xxxx", "000", "###"
}

# 2. Regex-Cleaning: a–z und Ziffern behalten
def clean_syntax(text):
    return re.sub(r"[^a-z0-9\s]", " ", text.lower())

# 3. Tokenisierung & Lemmatisierung mit spaCy
def preprocess_doc(doc):
    return " ".join(
        token.lemma_ for token in doc
        if (token.is_alpha or token.like_num)
           and not token.is_stop
           and token.lemma_ not in custom_stopwords
           and len(token) > 1
    )

# 4. Volles CSV einlesen
df = pd.read_csv("complaints_processed.csv")
texts = df["narrative"].fillna("").astype(str)

# 5. Vorreinigung
cleaned = texts.apply(clean_syntax)

# 6. Batch-Verarbeitung mit spaCy
processed = []
for doc in nlp.pipe(cleaned, batch_size=1000):
    processed.append(preprocess_doc(doc))
df["clean_text"] = processed

# 7a. TF-IDF-Vektorisierung
tfidf = TfidfVectorizer(max_df=0.85, min_df=3, ngram_range=(1,2))
X_tfidf = tfidf.fit_transform(df["clean_text"])

# 7b. CountVectorizer für LSA
cnt = CountVectorizer(max_df=0.85, min_df=3, ngram_range=(1,2))
X_count = cnt.fit_transform(df["clean_text"])

# 8a. LDA: 10 Themen, batch-Modus, 20 Iterationen
n_topics = 10
lda = LatentDirichletAllocation(
    n_components=n_topics,
    max_iter=20,
    learning_method="batch",
    random_state=0
)
lda.fit(X_tfidf)

# 8b. LSA (TruncatedSVD) mit 10 Komponenten
lsa = TruncatedSVD(n_components=n_topics, n_iter=10, random_state=0)
lsa.fit(X_count)

# 9. Top-Terms extrahieren
def get_top_terms(model, features, n=5):
    terms = []
    for comp in model.components_:
        idx = comp.argsort()[:-n-1:-1]
        terms.append([features[i] for i in idx])
    return terms

tfidf_feats = tfidf.get_feature_names_out()
count_feats = cnt.get_feature_names_out()

top_lda = get_top_terms(lda, tfidf_feats, n=5)
top_lsa = get_top_terms(lsa, count_feats, n=5)

# 10. Themen ausgeben
print("LDA-Themen (Top 5):")
for i, terms in enumerate(top_lda, 1):
    print(f"Topic {i}: {', '.join(terms)}")

print("\nLSA-Themen (Top 5):")
for i, terms in enumerate(top_lsa, 1):
    print(f"Topic {i}: {', '.join(terms)}")

# 11. Coherence Score für LDA berechnen
token_lists = [txt.split() for txt in df["clean_text"]]
dictionary = Dictionary(token_lists)
cm = CoherenceModel(
    topics=top_lda,
    texts=token_lists,
    dictionary=dictionary,
    coherence="c_v",
    processes=1
)
print(f"\nLDA Coherence Score: {cm.get_coherence():.4f}")

# 12. Heatmap
doc_topic = lda.transform(X_tfidf[:100])
plt.figure(figsize=(12,6))
sns.heatmap(
    doc_topic,
    cmap="viridis",
    cbar_kws={"label":"Wahrscheinlichkeit"},
    xticklabels=[f"T{i}" for i in range(1, n_topics+1)],
    yticklabels=range(1, 101)
)
plt.title("LDA: Dokument → Thema (erst 100 Dokumente)")
plt.xlabel("Thema")
plt.ylabel("Dokument")
plt.tight_layout()
plt.show()

# 13. Horizontale Balkendiagramme der LDA-Top-Terms
for idx, terms in enumerate(top_lda):
    weights = [lda.components_[idx][tfidf.vocabulary_[t]] for t in terms]
    plt.figure(figsize=(6,3))
    sns.barplot(x=weights, y=terms, orient='h')
    plt.title(f"LDA Top-Terms Thema {idx+1}")
    plt.xlabel("Gewichtung")
    plt.ylabel("Term")
    plt.tight_layout()
    plt.show()
