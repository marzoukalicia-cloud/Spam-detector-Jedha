# Spam-detector-Jedha
AT&amp;T Spam Detector -- Détection de SMS indésirables -- Bloc 4 Deep Learning Jedha Bootcamp

## 🎯 Contexte & Enjeu Métier
Pour un opérateur de télécommunications d'envergure mondiale comme **AT&T**, la prolifération des SMS frauduleux (spams, phishing, hameçonnage bancaire) représente un vecteur majeur d'insatisfaction client et de risques de sécurité. 

L'objectif de ce projet est de concevoir un moteur de classification de texte automatisé, capable de filtrer en temps réel les messages malveillants (`spam`) tout en garantissant la bonne distribution des messages légitimes (`ham`). 

---

## 🔬 Démarche Scientifique & Architecture des Modèles

Face à un jeu de données textuel asymétrique (forte prédominance de messages légitimes), le projet adopte une méthodologie rigoureuse en deux phases :

### Phase 1 : Conception d'une Baseline "From Scratch" (`SpamClassifierEmbedding`)
* **Tokenisation :** Utilisation du tokenizer de pointe `tiktoken` (configuration `cl100k_base` issue de l'écosystème GPT-4) appliqué sur des séquences calibrées à une longueur maximale (`MAX_LEN = 100`).
* **Architecture :** Implémentation sous **PyTorch** d'une couche d'apprentissage d'Embeddings personnalisée, couplée à un mécanisme de pooling moyen (`AdaptiveAvgPool1d`) et une couche de classification linéaire.
* **Résultat :** Une baseline légère, extrêmement rapide à converger (~99% d'accuracy sur le train), servant de point de comparaison.

### Phase 2 : Modèle Champion via Transfer Learning (`DistilBERT`)
* **Architecture :** Fine-tuning de l'architecture de type Transformer `DistilBertForSequenceClassification` via la bibliothèque Hugging Face.
* **Intérêt technique :** DistilBERT conserve 97% des capacités de compréhension contextuelle de BERT tout en étant 40% plus léger et 60% plus rapide, répondant ainsi aux contraintes de latence industrielle de traitement des flux chez AT&T.

---

## 📈 Analyse Comparative des Performances

Les modèles ont été évalués à l'aide d'un découpage stratifié (`stratify=y`) afin de maintenir l'équité des classes sur les ensembles de Train, Validation et Test.

| Critère Métier | Modèle Baseline (Embedding Custom) | Modèle Champion (DistilBERT Fine-Tuned) |
| :--- | :---: | :---: |
| **Accuracy Globale** | ~98.9% | **99.1%** |
| **F1-Score (Classe Spam)** | ~0.94 | **0.98** |
| **Analyse du Contexte** | Limité (mots-clés isolés) | **Excellente (liaisons sémantiques)** |
| **Robustesse au Bruit** | Moyenne | **Très élevée** |
| **Vitesse d'Inférence** | Ultra-rapide (faible coût) | Modérée (calcul GPU recommandé) |

### Focus Métrique : Pourquoi le F1-Score ?
Dans un système de filtrage télécom, le coût d'un **Faux Positif** (bloquer le SMS légitime urgent d'un utilisateur en le classant comme spam) est critique. DistilBERT sécurise ce point en atteignant une précision et un rappel quasi-parfaits (**F1-score de 0.98 sur la classe minoritaire**).

---

## 📂 Structure du Répertoire GitHub

```text
📦 AT-T_Spam_Detector
 ┣ 📂 notebook
 ┃ ┗ 📜 AT&T_spam_detector_AMK.ipynb  # Pipeline d'entraînement complet (PyTorch & Hugging Face)
 ┣ 📂 data
 ┃ ┣ 📜 spam.csv                     # Dataset d'entrée brut (5 572 SMS étiquetés)
 ┃ ┗ 📜 crit-re-embedding-5.csv       # Synthèse des critères et métriques de performance
 ┣ 📂 outputs
 ┃ ┗ 📂 viz                          # Courbes d'apprentissage extraites du monitoring
 ┃   ┣ 📜 accuracy V1.png            # Évolution de la précision par époque (Baseline)
 ┃   ┗ 📜 train loss V1.png          # Convergence de la fonction de coût (Baseline)
 ┗ 📜 README.md                      # Documentation technique du projet
