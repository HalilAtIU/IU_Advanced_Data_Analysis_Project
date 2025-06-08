# NLP-Analyse Projekt  
**Advanced Data Analysis (DLBDSEDA02), IU Internationale Hochschule**

In diesem Kursprojekt für die Modulprüfung **Advanced Data Analysis** entwickeln wir eine NLP-Pipeline, um aus einer großen Menge unstrukturierter Bürgerbeschwerden einer Kommune die dringlichsten Themen zu extrahieren. Aufbauend auf der Konzeptionsphase (Datenquelle, Text-Cleaning, Vektorisierungs- und Themenmodellierungs-Ansätze) wird in der Erarbeitungsphase eine vollständige Implementierung in Python realisiert:

**Datenbasis:**  
Kaggle-Datensatz „Consumer Complaints Dataset for NLP“  
[https://www.kaggle.com/datasets/shashwatwork/consume-complaints-dataset-fo-nlp](https://www.kaggle.com/datasets/shashwatwork/consume-complaints-dataset-fo-nlp)

- **Vorverarbeitung:**  
  Einlesen mit pandas, Regex-Säuberung, Batch-Lemmatisierung per spaCy

- **Vektorisierung:**  
  TF-IDF & Bag-of-Words (CountVectorizer)

- **Themenmodellierung:**  
  LDA & LSA (TruncatedSVD)

- **Evaluation:**  
  Coherence Score zur Themen-Optimierung

- **Visualisierung:**  
  Heatmaps der Dokument-Topic-Verteilungen und Barplots der Top-Terms (matplotlib, seaborn)

Ziel ist es, Entscheidungsträger:innen der Stadtverwaltung kompakte, interpretierbare Themencluster vorzulegen, damit künftige Entscheidungen stärker an den aktuell geäußerten Bürgeranliegen ausgerichtet werden können.  
