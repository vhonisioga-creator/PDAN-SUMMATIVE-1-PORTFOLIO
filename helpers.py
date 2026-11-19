import sys, subprocess, importlib
for pkg, name in [("pandas", "pandas"), ("numpy", "numpy"), ("matplotlib", "matplotlib"), ("seaborn", "seaborn"),
                  ("scikit-learn", "sklearn"), ("nltk", "nltk"), ("wordcloud", "wordcloud")]:
    try:
        importlib.import_module(name)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])
# Any missing library is installed automatically. The package NAME is imported, never a file path.

import re, string
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
try:
    get_ipython().run_line_magic('matplotlib', 'inline')
except Exception:
    pass
# This asks Jupyter to draw charts inside the notebook, which is needed for them to appear.
import seaborn as sns
import nltk
from IPython.display import display
from nltk.corpus import stopwords
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_selection import chi2
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords', quiet=True)
# The NLTK word list is downloaded only when it is missing, so imports never wait on the internet.

DATA_PATH = r"C:\Users\vhona\Downloads\medical_text_classification_fake_dataset.csv"
RANDOM_STATE = 42
N_TOPICS = 4
FOCUS = 'Cancer'
# Change it to 'random' for a normal stratified 80/20 split, but identical sentences then appear in both sets and scores are inflated.
SPLIT_MODE = 'sentence'

STOP_WORDS = set(stopwords.words('english')) - {'no', 'nor', 'not'}
# Negation words are kept, because removing "not" would reverse the meaning of a sentence.

LDA_EXTRA_STOPS = {'essential', 'often', 'commonly', 'well', 'key', 'first', 'step', 'standard', 'many',
                   'also', 'may', 'used', 'improves', 'improve'}
# These filler words appear across several categories, so they are removed only for topic modelling.

CACHE = {}
# This dictionary stores prepared data so repeated calls in one session are instant.

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'<.*?>', ' ', text)
    text = text.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
    text = re.sub(r'\b\d+\b', ' ', text)
    return ' '.join(w for w in text.split() if w not in STOP_WORDS)
# This lowercases the text, removes HTML tags, punctuation and stand-alone numbers, then removes stopwords.

def prepare(path):
    raw = pd.read_csv(path)
    n_raw = len(raw)
    raw.columns = [c.strip().lower() for c in raw.columns]
    if 'text' not in raw or 'label' not in raw:
        raise ValueError(f"Expected 'text' and 'label' columns, found: {list(raw.columns)}")
    data = raw.copy()
    data['category'] = data['label'].astype(str).str.strip().str.title()
    n_missing = int(data[['text', 'category']].isnull().any(axis=1).sum())
    data = data.dropna(subset=['text', 'category'])
    n_dup = int(data.duplicated(subset=['text']).sum())
    data['clean_text'] = data['text'].apply(clean_text)
    n_empty = int((data['clean_text'].str.strip() == '').sum())
    data = data[data['clean_text'].str.strip() != ''].copy()
    data['length'] = data['clean_text'].str.split().str.len()
    CACHE['stats'] = dict(raw=n_raw, missing=n_missing, dup=n_dup, unique=int(data['text'].nunique()), empty=n_empty,
                          final=len(data), avg_words=data['length'].mean(), vocab=len(set(' '.join(data['clean_text']).split())))
    return data
# This loads the file, standardises the columns, removes missing values, cleans the text, records the length,
# and stores data-quality statistics for the report. Duplicates are kept because only 20 unique sentences exist.

def split_by_sentence(data, test_size=0.2, seed=None):
    rng = np.random.RandomState(RANDOM_STATE if seed is None else seed)
    held = []
    for cat, grp in data.groupby('category'):
        sentences = sorted(grp['text'].unique())
        n_hold = max(1, int(round(test_size * len(sentences))))
        held += list(rng.choice(sentences, size=n_hold, replace=False))
    is_valid = data['text'].isin(held)
    return data[~is_valid].copy(), data[is_valid].copy()
# Every copy of a held-out sentence goes to the validation set, so the model is tested on sentences it has never seen.

def load_data():
    if 'data' not in CACHE:
        CACHE['all'] = prepare(DATA_PATH)
        if SPLIT_MODE == 'random':
            CACHE['data'] = tuple(d.copy() for d in train_test_split(
                CACHE['all'], test_size=0.2, random_state=RANDOM_STATE, stratify=CACHE['all']['category']))
        else:
            CACHE['data'] = split_by_sentence(CACHE['all'])
    return CACHE['data']
# This returns the training and validation sets, preparing them only the first time.

def load_all():
    load_data()
    return CACHE['all']

def get_features():
    if 'features' not in CACHE:
        train, valid = load_data()
        tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), min_df=2)
        Xtr_full = tfidf.fit_transform(train['clean_text'])
        Xte_full = tfidf.transform(valid['clean_text'])
        chi_scores, p_values = chi2(Xtr_full, train['category'])
        keep = p_values < 0.05
        if keep.sum() == 0:
            keep = np.ones(len(p_values), dtype=bool)
        CACHE['features'] = dict(X_train=Xtr_full[:, keep], X_test=Xte_full[:, keep],
                                 y_train=train['category'], y_test=valid['category'],
                                 terms=np.array(tfidf.get_feature_names_out()), kept_terms=np.array(tfidf.get_feature_names_out())[keep], chi_scores=chi_scores,
                                 n_before=Xtr_full.shape[1], n_after=int(keep.sum()))
    return CACHE['features']
# TF-IDF is fitted on the training sentences only, so nothing from the validation sentences leaks in.
# The chi-square test keeps only terms whose link to the category has a p-value below 0.05.

def train_baselines():
    if 'base' not in CACHE:
        f = get_features()
        CACHE['base'] = (LogisticRegression(max_iter=1000).fit(f['X_train'], f['y_train']),
                         MultinomialNB().fit(f['X_train'], f['y_train']))
    return CACHE['base']
# The two baseline models use default settings and are trained once per session.

def get_tuned():
    if 'tuned' not in CACHE:
        f = get_features()
        train, valid = load_data()
        groups = pd.factorize(train['text'])[0]
        cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
        CACHE['tuned'] = GridSearchCV(LogisticRegression(max_iter=1000),
                                      param_grid={'C': [0.1, 1, 10, 100]},
                                      scoring='f1_macro', cv=cv).fit(f['X_train'], f['y_train'], groups=groups)
    return CACHE['tuned']
# This returns the tuned model, running the grid search only once.

def fit_lda(texts, n_topics=N_TOPICS, n_top_words=10):
    cv = CountVectorizer(max_df=0.5, min_df=5, max_features=5000, stop_words=sorted(LDA_EXTRA_STOPS))
    dtm = cv.fit_transform(texts)
    lda = LatentDirichletAllocation(n_components=n_topics, learning_method='online',
                                    max_iter=10, random_state=RANDOM_STATE)
    doc_topics = lda.fit_transform(dtm)
    terms = cv.get_feature_names_out()
    for i, comp in enumerate(lda.components_):
        print(f"Topic {i + 1}:", ', '.join(terms[j] for j in comp.argsort()[-n_top_words:][::-1]))
    return lda, cv, doc_topics
# This fits LDA on a set of texts and prints the ten strongest words in each topic.

def evaluate(model, X_te, y_te, name, show=True):
    preds = model.predict(X_te)
    if show:
        print(f"\n=== {name} ===")
        print(classification_report(y_te, preds, zero_division=0))
        labels = sorted(y_te.unique())
        fig, ax = plt.subplots(figsize=(6, 5))
        ConfusionMatrixDisplay(confusion_matrix(y_te, preds, labels=labels), display_labels=labels).plot(cmap='Blues', ax=ax)
        ax.set_title(f'Confusion Matrix - {name}')
        plt.xticks(rotation=20)
        plt.show()
    return {'Model': name, 'Accuracy': accuracy_score(y_te, preds),
            'Macro Precision': precision_score(y_te, preds, average='macro', zero_division=0),
            'Macro Recall': recall_score(y_te, preds, average='macro', zero_division=0),
            'Macro F1': f1_score(y_te, preds, average='macro', zero_division=0),
            f'{FOCUS} Recall': recall_score(y_te, preds, labels=[FOCUS], average=None, zero_division=0)[0]}
# This scores a model on the validation set and returns one row of results.
# With show=True it also prints the report and draws the confusion matrix.

def get_results():
    f = get_features(); lr, nb = train_baselines(); g = get_tuned()
    rows = [evaluate(lr, f['X_test'], f['y_test'], 'LR baseline', show=False),
            evaluate(nb, f['X_test'], f['y_test'], 'NB baseline', show=False),
            evaluate(g.best_estimator_, f['X_test'], f['y_test'], 'LR tuned', show=False)]
    return pd.DataFrame(rows).set_index('Model').round(3)
# This builds the comparison table of all three models in one call.

def quality_table():
    load_data()
    s = CACHE['stats']
    return pd.DataFrame([{'Raw rows': s['raw'], 'Missing text or label': s['missing'], 'Duplicate rows': s['dup'],
                          'Unique sentences': s['unique'], 'Empty after cleaning': s['empty'], 'Final rows': s['final'],
                          'Average words': round(s['avg_words'], 1), 'Vocabulary size': s['vocab']}], index=['Dataset'])
# This builds the data-quality table for the report: rows, duplicates, unique sentences and vocabulary size.

def report_table(model, X_te, y_te):
    rep = classification_report(y_te, model.predict(X_te), output_dict=True, zero_division=0)
    keep = [k for k in rep if k != 'accuracy']
    return pd.DataFrame({k: rep[k] for k in keep}).T[['precision', 'recall', 'f1-score', 'support']].round(3)
# This returns precision, recall, F1 and support for each category as a table that is easy to screenshot.

def confusion_table(model, X_te, y_te):
    labels = sorted(y_te.unique())
    cm = confusion_matrix(y_te, model.predict(X_te), labels=labels)
    rows = []
    for i, lab in enumerate(labels):
        tp = cm[i, i]; fp = cm[:, i].sum() - tp; fn = cm[i, :].sum() - tp
        rows.append({'Category': lab, 'TP': tp, 'FP': fp, 'FN': fn,
                     'Precision': tp / (tp + fp) if tp + fp else 0.0, 'Recall': tp / (tp + fn) if tp + fn else 0.0})
    return cm, labels, pd.DataFrame(rows).set_index('Category').round(3)
# This returns the confusion matrix and a table of true positives, false positives and false negatives per category.