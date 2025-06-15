import os
import re
import requests
import tensorflow as tf
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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
import gensim.downloader as api

# — Estilos CSS —
st.markdown("""
    <style>
    .stApp { background-color: #000702; }
    [data-testid="stSidebar"] > div:first-child { background-color: #1f2c13; }
    .css-1v0mbdj, .css-1d391kg { color: #d4903b !important; }
    button { background-color: #a94802 !important; color: #ffffff !important; }
    </style>
""", unsafe_allow_html=True)

# Semilla aleatoria
tf.random.set_seed(42)

# — Helpers de Google Drive para pesos —
def get_confirm_token(resp):
    for k, v in resp.cookies.items():
        if k.startswith('download_warning'):
            return v
    m = re.search(r'confirm=([0-9A-Za-z_-]+)&', resp.text)
    return m.group(1) if m else None

def download_from_drive(file_id, dest):
    URL = 'https://docs.google.com/uc?export=download'
    sess = requests.Session()
    r = sess.get(URL, params={'id': file_id}, stream=True)
    token = get_confirm_token(r)
    if token:
        r = sess.get(URL, params={'id': file_id, 'confirm': token}, stream=True)
    with open(dest, 'wb') as f:
        for chunk in r.iter_content(32768):
            if chunk:
                f.write(chunk)

# IDs en Drive de tus archivos .weights.h5
DRIVE_IDS = {
    'rnn_weights':  'TU_ID_DRIVE_RNN',
    'lstm_weights': 'TU_ID_DRIVE_LSTM',
    'bilstm_weights':'TU_ID_DRIVE_BILSTM'
}

# — Configuración local —
cfg = {
    'max_features': 10000,
    'maxlen':       300,
    'emb_dim':      300,
    'rnn_weights':   'rnn_model_weights.weights.h5',
    'lstm_weights':  'lstm_model_weights.weights.h5',
    'bilstm_weights':'model_weights.weights.h5'
}

# Descarga automática al iniciar si faltan
for key, fid in DRIVE_IDS.items():
    if not os.path.exists(cfg[key]):
        download_from_drive(fid, cfg[key])

# Flag para saber si ya entrenaste
weights_exist = all(os.path.exists(cfg[k]) for k in DRIVE_IDS)

# — Funciones auxiliares de datos y embeddings —
@st.cache_data
def load_raw():
    ds = load_dataset("tweets_hate_speech_detection", split="train")
    return ds.to_pandas()

@st.cache_resource
def get_tok_emb(max_features, maxlen, emb_dim, **kw):
    df_train, _ = split_data(load_raw())
    tok = Tokenizer(num_words=max_features, oov_token="<OOV>")
    tok.fit_on_texts(df_train.tweet.astype(str))
    # Carga GloVe 300d desde gensim
    model = api.load("glove-wiki-gigaword-300")
    M = np.zeros((max_features, emb_dim))
    for w, i in tok.word_index.items():
        if i < max_features and w in model:
            M[i] = model[w]
    return tok, M

def split_data(df):
    tr, val = train_test_split(df, test_size=0.2, stratify=df.label, random_state=42)
    maj = tr[tr.label==0]
    min_ = tr[tr.label==1]
    up  = resample(min_, replace=True, n_samples=len(maj), random_state=42)
    return pd.concat([maj, up]).sample(frac=1, random_state=42), val

def build_rnn(cfg, M, units=64, dropout=0.2):
    m = Sequential([
        Input((cfg['maxlen'],)),
        Embedding(cfg['max_features'], cfg['emb_dim'], weights=[M], trainable=False),
        SimpleRNN(units, dropout=dropout, recurrent_dropout=dropout),
        Dense(2, activation='softmax')
    ])
    m.compile('adam', 'categorical_crossentropy', ['accuracy'])
    return m

def build_lstm(cfg, M, units=64, dropout=0.2):
    m = Sequential([
        Input((cfg['maxlen'],)),
        Embedding(cfg['max_features'], cfg['emb_dim'], weights=[M], trainable=False),
        LSTM(units, dropout=dropout, recurrent_dropout=dropout),
        Dense(2, activation='softmax')
    ])
    m.compile('adam', 'categorical_crossentropy', ['accuracy'])
    return m

def build_bilstm(cfg, M, units=64, dropout=0.2):
    m = Sequential([
        Input((cfg['maxlen'],)),
        Embedding(cfg['max_features'], cfg['emb_dim'], weights=[M], trainable=False),
        Bidirectional(LSTM(units, dropout=dropout, recurrent_dropout=dropout)),
        Dense(2, activation='softmax')
    ])
    m.compile('adam', 'categorical_crossentropy', ['accuracy'])
    return m

def prepare(df, tok, maxlen):
    seqs = tok.texts_to_sequences(df.tweet.astype(str))
    X = pad_sequences(seqs, maxlen=maxlen)
    y = to_categorical(df.label.values, num_classes=2)
    return X, y

def dataset_visualization(df):
    df['length'] = df.tweet.str.split().str.len()
    st.subheader("1. Class Distribution")
    fig, ax = plt.subplots()
    df.label.value_counts().plot.bar(ax=ax)
    st.pyplot(fig)
    st.subheader("2. Tweet Length Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df['length'], bins=50, kde=True, ax=ax)
    st.pyplot(fig)
    # (añade las demás gráficas como antes)

# — Sidebar & Navegación —
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio("Go to", [
    "Train RNN",
    "Train LSTM",
    "Train BiLSTM",
    "Inference",
    "Dataset Exploration",
    "Hyperparameter Tuning",
    "Model Analysis & Justification"
])

# — Train Simple RNN —
if app_mode == "Train RNN":
    st.title("🚀 Train Simple RNN")
    if weights_exist:
        st.info("RNN weights already exist; skip training.")
    elif st.button("Start Training RNN"):
        dfb, _ = split_data(load_raw())
        tok, M = get_tok_emb(**cfg)
        X, y = prepare(dfb, tok, cfg["maxlen"])
        model = build_rnn(cfg, M)
        with st.spinner("Training..."):
            h = model.fit(X, y, validation_split=0.1, epochs=3, batch_size=128)
            model.save_weights(cfg["rnn_weights"])
        st.success(f"Trained RNN (acc {h.history['accuracy'][-1]:.3f})")

# — Train LSTM —
elif app_mode == "Train LSTM":
    st.title("🚀 Train LSTM")
    if weights_exist:
        st.info("LSTM weights already exist; skip training.")
    elif st.button("Start Training LSTM"):
        dfb, _ = split_data(load_raw())
        tok, M = get_tok_emb(**cfg)
        X, y = prepare(dfb, tok, cfg["maxlen"])
        model = build_lstm(cfg, M)
        with st.spinner("Training..."):
            h = model.fit(X, y, validation_split=0.1, epochs=3, batch_size=128)
            model.save_weights(cfg["lstm_weights"])
        st.success(f"Trained LSTM (acc {h.history['accuracy'][-1]:.3f})")

# — Train BiLSTM —
elif app_mode == "Train BiLSTM":
    st.title("🚀 Train BiLSTM")
    if weights_exist:
        st.info("BiLSTM weights already exist; skip training.")
    elif st.button("Start Training BiLSTM"):
        dfb, _ = split_data(load_raw())
        tok, M = get_tok_emb(**cfg)
        X, y = prepare(dfb, tok, cfg["maxlen"])
        model = build_bilstm(cfg, M)
        with st.spinner("Training..."):
            h = model.fit(X, y, validation_split=0.1, epochs=3, batch_size=128)
            model.save_weights(cfg["bilstm_weights"])
        st.success(f"Trained BiLSTM (acc {h.history['accuracy'][-1]:.3f})")

# — Inference —
elif app_mode == "Inference":
    st.title("📝 Inference")
    tok, M = get_tok_emb(**cfg)
    models = {}
    # Carga las arquitecturas y pesos
    for name, builder, key in [
        ("RNN",      build_rnn,  "rnn_weights"),
        ("LSTM",     build_lstm, "lstm_weights"),
        ("BiLSTM",   build_bilstm,"bilstm_weights")
    ]:
        m = builder(cfg, M)
        try:
            m.load_weights(cfg[key])
            models[name] = m
        except:
            st.error(f"Missing weights for {name}. Train first.")
    # Predict por modelo
    for name, model in models.items():
        txt = st.text_area(f"Enter text for {name}", key=f"txt_{name}")
        if st.button(f"Predict {name}", key=f"btn_{name}") and txt:
            X = pad_sequences(tok.texts_to_sequences([txt]), maxlen=cfg["maxlen"])
            p = model.predict(X)[0]
            st.write(f"Class: {np.argmax(p)}, Confidence: {p}")

# — Dataset Exploration —
elif app_mode == "Dataset Exploration":
    st.title("📊 Dataset Exploration")
    dfb, _ = split_data(load_raw())
    dataset_visualization(dfb)

# — Hyperparameter Tuning —
elif app_mode == "Hyperparameter Tuning":
    st.title("🔧 Hyperparameter Tuning (BiLSTM)")
    tok, M = get_tok_emb(**cfg)
    if st.button("Run Tuning"):
        study = optuna.create_study(direction="maximize",
                                   storage="sqlite:///optuna.db",
                                   load_if_exists=True)
        with st.spinner("Tuning in progress..."):
            def objective(trial):
                u = trial.suggest_int("lstm_units", 32, 128)
                d = trial.suggest_float("dropout_rate", 0.1, 0.5)
                m = build_bilstm(cfg, M, units=u, dropout=d)
                dfb, _ = split_data(load_raw())
                X, y = prepare(dfb, tok, cfg["maxlen"])
                h = m.fit(X, y, validation_split=0.1,
                          epochs=3, batch_size=128, verbose=0)
                return h.history["val_accuracy"][-1]
            study.optimize(objective, n_trials=5)
        st.success("Tuning completed")
        st.json(study.best_params)
    else:
        st.info("Click Run Tuning to start or view existing results.")
        try:
            study = optuna.load_study(study_name=study.study_name,
                                      storage="sqlite:///optuna.db")
            fig = vis.plot_optimization_history(study)
            st.plotly_chart(fig)
        except:
            pass

# — Model Analysis & Justification —
else:
    st.title("🧮 Model Analysis & Justification")
    tok, M = get_tok_emb(**cfg)
    dfb, df_val = split_data(load_raw())
    X_val, y_val = prepare(df_val, tok, cfg["maxlen"])
    y_true = np.argmax(y_val, axis=1)
    accuracies = {}
    for name, builder, key in [
        ("RNN",    build_rnn,  "rnn_weights"),
        ("LSTM",   build_lstm, "lstm_weights"),
        ("BiLSTM", build_bilstm,"bilstm_weights")
    ]:
        model = builder(cfg, M)
        try:
            model.load_weights(cfg[key])
        except:
            st.error(f"Missing weights for {name}")
            continue
        st.subheader(f"{name} Analysis")
        y_prob = model.predict(X_val)
        y_pred = np.argmax(y_prob, axis=1)
        rep = classification_report(y_true, y_pred, output_dict=True, digits=3)
        st.dataframe(pd.DataFrame(rep).transpose())
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", ax=ax)
        st.pyplot(fig)
        accuracies[name] = rep["accuracy"]
    if accuracies:
        best = max(accuracies, key=accuracies.get)
        st.write(f"🏆 Best Model: {best} ({accuracies[best]:.3f})")
