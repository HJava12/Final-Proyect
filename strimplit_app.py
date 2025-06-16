import os
import re
import requests
import tensorflow as tf
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
from cycler import cycler
import optuna
import optuna.visualization as vis
from wordcloud import WordCloud
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Embedding, Bidirectional, LSTM, SimpleRNN, Dense
from tensorflow.keras.utils import to_categorical
import glob

# —————————————————————————————————————————————————————
# Google Drive download helpers with HTML fallback for confirm token
# —————————————————————————————————————————————————————

def get_confirm_token(response):
    # first, try cookies
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    # fallback: search the HTML for confirm token
    m = re.search(r'href=".*?confirm=([0-9A-Za-z_]+)&', response.text)
    if m:
        return m.group(1)
    return None

def save_response_content(response, destination, chunk_size=32768):
    with open(destination, "wb") as f:
        for chunk in response.iter_content(chunk_size):
            if chunk:
                f.write(chunk)

@st.cache_data
def download_file_from_google_drive(file_id, destination):
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    # initial request
    response = session.get(URL, params={'id': file_id}, stream=True)
    token = get_confirm_token(response)
    if token:
        # re-request with confirm token
        response = session.get(URL, params={'id': file_id, 'confirm': token}, stream=True)
    save_response_content(response, destination)
    return destination

@st.cache_data
def ensure_glove(path="glove.6B.300d.txt", file_id="13UFJlS1cxKj6jkcP92gnjOe-YsIHrlYv"):
    if not os.path.exists(path):
        # download the file from Drive into path
        download_file_from_google_drive(file_id, path)
    return path

# download (cached) before anything else
glove_path = ensure_glove()

st.markdown(
    """
    <style>
    /* Fondo general de la app */
    .stApp {
      background-color: #000702;
    }
    /* Sidebar */
    [data-testid="stSidebar"] > div:first-child {
      background-color: #1f2c13;
    }
    /* Títulos y textos */
    .css-1v0mbdj, .css-1d391kg {
      color: #d4903b !important;
    }
    /* Botones */
    button {
      background-color: #a94802 !important;
      color: #ffffff !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Set random seed
tf.random.set_seed(42)

# Configuration
cfg = {
    "max_features": 10000,
    "maxlen": 300,
    "emb_dim": 300,
    "glove_path": glove_path,
    "bilstm_weights": "model_weights.weights.h5",      # DO NOT CHANGE
    "lstm_weights": "lstm_model_weights.weights.h5",
    "rnn_weights":  "rnn_model_weights.weights.h5"
}

cfg["rnn_weights"]    = ensure_file("rnn_model_weights.weights.h5",
                                    "1Of9Vlsd3HxujlsKp5fFE_qpNXQVoDppY")
cfg["lstm_weights"]   = ensure_file("lstm_model_weights.weights.h5",
                                    "1KQxzoSNb6bGTNlLY_AkpebsIkBmZhvh5")
cfg["bilstm_weights"] = ensure_file("model_weights.weights.h5",
                                    "1xKWEgJDtbdCnYjuSMWXykT9N5sb0249g")

@st.cache_data
def load_raw():
    ds = load_dataset("tweets_hate_speech_detection", split="train")
    return ds.to_pandas()

@st.cache_resource
def get_tok_emb(max_features, maxlen, emb_dim, glove_path, **kwargs):
    df_train, _ = split_data(load_raw())
    tok = Tokenizer(num_words=max_features, oov_token="<OOV>")
    tok.fit_on_texts(df_train.tweet.astype(str))

    emb_index = {}
    with open(glove_path, encoding="utf8") as f:
        for line in f:
            parts = line.strip().split()
            # debe haber exactamente emb_dim + 1 elementos
            if len(parts) != emb_dim + 1:
                continue
            word, *vector = parts
            try:
                emb_index[word] = np.asarray(vector, dtype="float32")
            except ValueError:
                # si alguna parte no es numérica, la saltamos
                continue

    # construye la matriz de embeddings
    M = np.zeros((max_features, emb_dim), dtype="float32")
    for w, i in tok.word_index.items():
        if i < max_features and w in emb_index:
            M[i] = emb_index[w]
    return tok, M


def split_data(df):
    df_train, df_val = train_test_split(
        df, test_size=0.2, stratify=df.label, random_state=42
    )
    maj = df_train[df_train.label == 0]
    min_ = df_train[df_train.label == 1]
    min_up = resample(min_, replace=True, n_samples=len(maj), random_state=42)
    return pd.concat([maj, min_up]).sample(frac=1, random_state=42), df_val

def build_rnn(cfg, embedding_matrix, units=64, dropout=0.2):
    model = Sequential([
        Input(shape=(cfg["maxlen"],)),
        Embedding(cfg["max_features"], cfg["emb_dim"], weights=[embedding_matrix], trainable=False),
        SimpleRNN(units, dropout=dropout, recurrent_dropout=dropout),
        Dense(2, activation="softmax")
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model

def build_lstm(cfg, embedding_matrix, units=64, dropout=0.2):
    model = Sequential([
        Input(shape=(cfg["maxlen"],)),
        Embedding(cfg["max_features"], cfg["emb_dim"], weights=[embedding_matrix], trainable=False),
        LSTM(units, dropout=dropout, recurrent_dropout=dropout),
        Dense(2, activation="softmax")
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model

def build_bilstm(cfg, embedding_matrix, units=64, dropout=0.2):
    model = Sequential([
        Input(shape=(cfg["maxlen"],)),
        Embedding(cfg["max_features"], cfg["emb_dim"], weights=[embedding_matrix], trainable=False),
        Bidirectional(LSTM(units, dropout=dropout, recurrent_dropout=dropout)),
        Dense(2, activation="softmax")
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model

def prepare(df, tokenizer, maxlen, num_classes=2):
    seqs = tokenizer.texts_to_sequences(df.tweet.astype(str))
    X = pad_sequences(seqs, maxlen=maxlen)
    y = to_categorical(df.label.values, num_classes=num_classes)
    return X, y

def dataset_visualization(df_bal):
    df = df_bal.copy()
    df['length'] = df.tweet.str.split().str.len()

    st.subheader("1. Class Distribution")
    fig, ax = plt.subplots()
    df.label.value_counts().plot.bar(ax=ax)
    fig.patch.set_facecolor('#3d3e36')
    ax.set_facecolor('#3d3e36')
    axes_color = '#5d612e'
    ax.spines['bottom'].set_color(axes_color)
    ax.spines['left'].set_color(axes_color)
    ax.tick_params(axis='x', colors=axes_color)
    ax.tick_params(axis='y', colors=axes_color)
    ax.set_xlabel('Class', color='white')
    ax.set_ylabel('Count', color='white')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_color('white')
    st.pyplot(fig)

    st.subheader("2. Tweet Length Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df['length'], bins=50, kde=True, ax=ax)
    ax.set_xlabel('Tweet Length')
    fig.patch.set_facecolor('#3d3e36')
    ax.set_facecolor('#3d3e36')
    axes_color = '#5d612e'
    ax.spines['bottom'].set_color(axes_color)
    ax.spines['left'].set_color(axes_color)
    ax.tick_params(axis='x', colors=axes_color)
    ax.tick_params(axis='y', colors=axes_color)
    ax.set_xlabel('Class', color='white')
    ax.set_ylabel('Count', color='white')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_color('white')
    st.pyplot(fig)

    st.subheader("3. Length by Class Boxplot")
    fig, ax = plt.subplots()
    sns.boxplot(x='label', y='length', data=df, ax=ax)
    ax.set_xlabel('Class')
    ax.set_ylabel('Length')
    fig.patch.set_facecolor('#3d3e36')
    ax.set_facecolor('#3d3e36')
    axes_color = '#5d612e'
    ax.spines['bottom'].set_color(axes_color)
    ax.spines['left'].set_color(axes_color)
    ax.tick_params(axis='x', colors=axes_color)
    ax.tick_params(axis='y', colors=axes_color)
    ax.set_xlabel('Class', color='white')
    ax.set_ylabel('Count', color='white')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_color('white')
    st.pyplot(fig)

    st.subheader("4. Top 20 Most Common Words")
    words = df.tweet.str.cat(sep=' ').split()
    freq = pd.Series(words).value_counts().head(20)
    fig, ax = plt.subplots()
    freq.plot.barh(ax=ax)
    ax.invert_yaxis()
    ax.set_xlabel('Frequency')
    fig.patch.set_facecolor('#3d3e36')
    ax.set_facecolor('#3d3e36')
    axes_color = '#5d612e'
    ax.spines['bottom'].set_color(axes_color)
    ax.spines['left'].set_color(axes_color)
    ax.tick_params(axis='x', colors=axes_color)
    ax.tick_params(axis='y', colors=axes_color)
    ax.set_xlabel('Class', color='white')
    ax.set_ylabel('Count', color='white')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_color('white')
    st.pyplot(fig)

    st.subheader("5. Word Cloud")
    corpus = df.tweet.str.cat(sep=' ')
    wc = WordCloud(width=800, height=400, background_color='white').generate(corpus)
    fig, ax = plt.subplots(figsize=(10,5))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    fig.patch.set_facecolor('#3d3e36')
    ax.set_facecolor('#3d3e36')
    axes_color = '#5d612e'
    ax.spines['bottom'].set_color(axes_color)
    ax.spines['left'].set_color(axes_color)
    ax.tick_params(axis='x', colors=axes_color)
    ax.tick_params(axis='y', colors=axes_color)
    ax.set_xlabel('Class', color='white')
    ax.set_ylabel('Count', color='white')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_color('white')
    st.pyplot(fig)

    st.subheader("6. Top 15 Bigrams")
    from collections import Counter
    bigrams = zip(words, words[1:])
    bigram_freq = Counter(bigrams).most_common(15)
    labels, counts = zip(*bigram_freq)
    labels = [' '.join(b) for b in labels]
    fig, ax = plt.subplots()
    pd.Series(counts, index=labels).plot.bar(ax=ax)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    fig.patch.set_facecolor('#3d3e36')
    ax.set_facecolor('#3d3e36')
    axes_color = '#5d612e'
    ax.spines['bottom'].set_color(axes_color)
    ax.spines['left'].set_color(axes_color)
    ax.tick_params(axis='x', colors=axes_color)
    ax.tick_params(axis='y', colors=axes_color)
    ax.set_xlabel('Class', color='white')
    ax.set_ylabel('Count', color='white')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_color('white')
    st.pyplot(fig)

    st.subheader("7. Sample Tweets by Length Quartile")
    quartiles = df['length'].quantile([0.25, 0.5, 0.75])
    for i, q in enumerate(quartiles, start=1):
        tweet = df[df['length'] <= q].tweet.sample(1).values[0]
        st.write(f"- Quartile {i} (<= {int(q)} words): {tweet}")

# Sidebar & Navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio("Go to", [
    "Train Simple RNN",
    "Train LSTM",
    "Train BiLSTM",
    "Inference",
    "Dataset Exploration",
    "Hyperparameter Tuning",
    "Model Analysis & Justification"
])

# Train Simple RNN
if app_mode == "Train Simple RNN":
    st.title("🚀 Train the Simple RNN Model")
    if st.button("Start Training RNN"):
        df_train_bal, _ = split_data(load_raw())
        tokenizer, M = get_tok_emb(**cfg)
        X_train, y_train = prepare(df_train_bal, tokenizer, cfg["maxlen"])
        model = build_rnn(cfg, M)
        with st.spinner("Training Simple RNN..."):
            history = model.fit(X_train, y_train, validation_split=0.1, epochs=3, batch_size=128)
            model.save_weights(cfg["rnn_weights"])
        st.success("Simple RNN trained and saved")
        st.write(f"Accuracy: {history.history['accuracy'][-1]:.3f}")

# Train LSTM
elif app_mode == "Train LSTM":
    st.title("🚀 Train the LSTM Model")
    if st.button("Start Training LSTM"):
        df_train_bal, _ = split_data(load_raw())
        tokenizer, M = get_tok_emb(**cfg)
        X_train, y_train = prepare(df_train_bal, tokenizer, cfg["maxlen"])
        model = build_lstm(cfg, M)
        with st.spinner("Training LSTM..."):
            history = model.fit(X_train, y_train, validation_split=0.1, epochs=3, batch_size=128)
            model.save_weights(cfg["lstm_weights"])
        st.success("LSTM trained and saved")
        st.write(f"Accuracy: {history.history['accuracy'][-1]:.3f}")

# Train BiLSTM
elif app_mode == "Train BiLSTM":
    st.title("🚀 Train the BiLSTM Model")
    if st.button("Start Training BiLSTM"):
        df_train_bal, _ = split_data(load_raw())
        tokenizer, M = get_tok_emb(**cfg)
        X_train, y_train = prepare(df_train_bal, tokenizer, cfg["maxlen"])
        model = build_bilstm(cfg, M)
        with st.spinner("Training BiLSTM..."):
            history = model.fit(X_train, y_train, validation_split=0.1, epochs=3, batch_size=128)
            model.save_weights(cfg["bilstm_weights"])
        st.success("BiLSTM trained and saved")
        st.write(f"Accuracy: {history.history['accuracy'][-1]:.3f}")

# Inference
elif app_mode == "Inference":
    st.title("📝 Text Classification Inference")
    tokenizer, M = get_tok_emb(**cfg)
    models = {}
    for name, builder, wpath in [
        ("Simple RNN", build_rnn, cfg["rnn_weights"]),
        ("LSTM", build_lstm, cfg["lstm_weights"]),
        ("BiLSTM", build_bilstm, cfg["bilstm_weights"]),
    ]:
        m = builder(cfg, M)
        try:
            m.load_weights(wpath)
            models[name] = m
        except:
            st.error(f"Missing weights for {name}. Please train first.")
    for name, model in models.items():
        st.subheader(f"{name} Model")
        txt = st.text_area(f"Enter text for {name}:", key=name)
        if st.button(f"Predict with {name}", key=f"btn_{name}") and txt:
            X = pad_sequences(tokenizer.texts_to_sequences([txt]), maxlen=cfg["maxlen"])
            probs = model.predict(X)[0]
            st.write(f"Class: {np.argmax(probs)}, Confidence: {probs}")

# Dataset Exploration
elif app_mode == "Dataset Exploration":
    st.title("📊 In-depth Dataset Exploration")
    df_bal, _ = split_data(load_raw())
    dataset_visualization(df_bal)

# Hyperparameter Tuning
elif app_mode == "Hyperparameter Tuning":
    st.title("🔧 Hyperparameter Tuning (BiLSTM Only)")
    tokenizer, M = get_tok_emb(**cfg)
    if st.button("Run Tuning"):
        study = optuna.create_study(direction="maximize", storage="sqlite:///optuna.db", load_if_exists=True)
        @st.spinner("Tuning in progress...")
        def run():
            def objective(trial):
                u = trial.suggest_int("lstm_units", 32, 128)
                d = trial.suggest_float("dropout_rate", 0.1, 0.5)
                model = build_bilstm(cfg, M, units=u, dropout=d)
                df_bal, _ = split_data(load_raw())
                X, y = prepare(df_bal, tokenizer, cfg["maxlen"])
                h = model.fit(X, y, validation_split=0.1, epochs=3, batch_size=128, verbose=0)
                return h.history["val_accuracy"][-1]
            study.optimize(objective, n_trials=5)
            return study
        study = run()
        st.success("Tuning completed")
        st.json(study.best_params)
    else:
        try:
            study = optuna.load_study(study_name="optuna_study", storage="sqlite:///optuna.db")
            st.write("**Best parameters:**")
            st.json(study.best_params)
            fig = vis.plot_optimization_history(study)
            st.plotly_chart(fig)
        except:
            st.info("No tuning study found. Click 'Run Tuning' to start.")

# Model Analysis & Justification
else:
    st.title("🧮 Model Analysis & Justification")
    tokenizer, M = get_tok_emb(**cfg)
    loaded = {}
    for name, builder, wpath in [
        ("Simple RNN", build_rnn, cfg["rnn_weights"]),
        ("LSTM", build_lstm, cfg["lstm_weights"]),
        ("BiLSTM", build_bilstm, cfg["bilstm_weights"]),
    ]:
        model = builder(cfg, M)
        try:
            model.load_weights(wpath)
            loaded[name] = model
        except:
            st.error(f"Missing weights for {name}. Please train first.")
    df_bal, df_val = split_data(load_raw())
    X_val, y_val = prepare(df_val, tokenizer, cfg["maxlen"])
    y_true = np.argmax(y_val, axis=1)
    accuracies = {}
    for name, model in loaded.items():
        st.subheader(f"{name} Analysis")
        y_prob = model.predict(X_val)
        y_pred = np.argmax(y_prob, axis=1)
        rep = classification_report(y_true, y_pred, digits=3, output_dict=True)
        rep_df = pd.DataFrame(rep).transpose()
        st.dataframe(rep_df)
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", ax=ax, cmap="Oranges")
        st.pyplot(fig)
        df_val["pred"] = y_pred
        st.markdown("**False Positives:**")
        fp = df_val[(df_val.label==0)&(df_val.pred==1)].tweet
        if len(fp)>0:
            for t in fp.sample(min(3,len(fp)), random_state=42): st.write(f"- {t}")
        else:
            st.write("_None_")
        st.markdown("**False Negatives:**")
        fn = df_val[(df_val.label==1)&(df_val.pred==0)].tweet
        if len(fn)>0:
            for t in fn.sample(min(3,len(fn)), random_state=42): st.write(f"- {t}")
        else:
            st.write("_None_")
        st.markdown("**Error Interpretation:**")
        st.write("Misclassifications may come from noisy text, limited embedding coverage, or class imbalance.")
        st.markdown("**Suggestions for Improvement:**")
        st.write("- Increase data for minority class.")
        st.write("- Enhance text cleaning and preprocessing.")
        st.write("- Try contextual embeddings (e.g., BERT).")
        st.write("- Tune model architecture and hyperparameters.")
        accuracies[name] = rep["accuracy"]
    if accuracies:
        best = max(accuracies, key=accuracies.get)
        st.subheader(f"🏆 Best Model: {best} ({accuracies[best]:.3f})")
        st.write("Chosen for highest validation accuracy.")
    else:
        st.error("No model performances available.")
