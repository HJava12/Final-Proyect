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

# --- Estilos CSS ---
st.markdown(
    """
    <style>
    .stApp { background-color: #000702; }
    [data-testid=\"stSidebar\"] > div:first-child { background-color: #1f2c13; }
    .css-1v0mbdj, .css-1d391kg { color: #d4903b !important; }
    button { background-color: #a94802 !important; color: #ffffff !important; }
    </style>
    """,
    unsafe_allow_html=True
)

# Semilla aleatoria
tf.random.set_seed(42)

# --- Helpers para descargar desde Google Drive ---
def get_confirm_token(resp):
    for k,v in resp.cookies.items():
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

# IDs de Drive para los pesos
DRIVE_IDS = {
    'rnn_weights': 'TU_ID_DRIVE_RNN',
    'lstm_weights': 'TU_ID_DRIVE_LSTM',
    'bilstm_weights': 'TU_ID_DRIVE_BILSTM'
}

# Configuración de nombres locales
cfg = {
    'max_features': 10000,
    'maxlen': 300,
    'emb_dim': 300,
    'rnn_weights': 'rnn_model_weights.weights.h5',
    'lstm_weights': 'lstm_model_weights.weights.h5',
    'bilstm_weights': 'model_weights.weights.h5'
}

# Descarga automática de pesos si no existen
for key, fid in DRIVE_IDS.items():
    path = cfg[key]
    if not os.path.exists(path):
        download_from_drive(fid, path)

# Flag de existencia
weights_exist = all(os.path.exists(cfg[k]) for k in DRIVE_IDS)

@st.cache_data
def load_raw():
    df = load_dataset('tweets_hate_speech_detection', split='train')
    return df.to_pandas()

@st.cache_resource
def get_tok_emb(max_features, maxlen, emb_dim, **kwargs):
    df_train, _ = split_data(load_raw())
    tok = Tokenizer(num_words=max_features, oov_token='<OOV>')
    tok.fit_on_texts(df_train.tweet.astype(str))
    model = api.load('glove-wiki-gigaword-300')
    M = np.zeros((max_features, emb_dim))
    for w,i in tok.word_index.items():
        if i < max_features and w in model:
            M[i] = model[w]
    return tok, M

def split_data(df):
    train, val = train_test_split(df, test_size=0.2, stratify=df.label, random_state=42)
    maj = train[train.label==0]; min_ = train[train.label==1]
    up = resample(min_, replace=True, n_samples=len(maj), random_state=42)
    return pd.concat([maj, up]).sample(frac=1, random_state=42), val

def build_rnn(cfg, M, units=64, dropout=0.2):
    m = Sequential([
        Input((cfg['maxlen'],)),
        Embedding(cfg['max_features'], cfg['emb_dim'], weights=[M], trainable=False),
        SimpleRNN(units, dropout=dropout, recurrent_dropout=dropout),
        Dense(2, activation='softmax')
    ])
    m.compile('adam','categorical_crossentropy',['accuracy'])
    return m

def build_lstm(cfg, M, units=64, dropout=0.2):
    m = Sequential([
        Input((cfg['maxlen'],)),
        Embedding(cfg['max_features'], cfg['emb_dim'], weights=[M], trainable=False),
        LSTM(units, dropout=dropout, recurrent_dropout=dropout),
        Dense(2, activation='softmax')
    ])
    m.compile('adam','categorical_crossentropy',['accuracy'])
    return m

def build_bilstm(cfg, M, units=64, dropout=0.2):
    m = Sequential([
        Input((cfg['maxlen'],)),
        Embedding(cfg['max_features'], cfg['emb_dim'], weights=[M], trainable=False),
        Bidirectional(LSTM(units, dropout=dropout, recurrent_dropout=dropout)),
        Dense(2, activation='softmax')
    ])
    m.compile('adam','categorical_crossentropy',['accuracy'])
    return m

def prepare(df, tok, maxlen):
    seqs = tok.texts_to_sequences(df.tweet.astype(str))
    X = pad_sequences(seqs, maxlen=maxlen)
    y = to_categorical(df.label, num_classes=2)
    return X, y

def dataset_visualization(df):
    df['len'] = df.tweet.str.split().str.len()
    st.subheader('1. Class Dist.'); fig,ax=plt.subplots()
    df.label.value_counts().plot.bar(ax=ax); st.pyplot(fig)
    # ... resto visualizaciones ...

# Sidebar
st.sidebar.title('Navigation')
mode = st.sidebar.radio('Go to', ['Train RNN','Train LSTM','Train BiLSTM','Inference','Explore','Tune','Analysis'])

if mode=='Train RNN':
    st.title('Train RNN')
    if weights_exist:
        st.info('RNN weights exist, skip training')
    elif st.button('Start Training RNN'):
        dfb,_=split_data(load_raw()); tok,M=get_tok_emb(**cfg)
        X,y=prepare(dfb,tok,cfg['maxlen']); m=build_rnn(cfg,M)
        with st.spinner('Training...'): h=m.fit(X,y,validation_split=0.1,epochs=3)
        m.save_weights(cfg['rnn_weights']); st.success('Done')

elif mode=='Train LSTM':
    st.title('Train LSTM')
    if weights_exist:
        st.info('LSTM weights exist')
    elif st.button('Train LSTM'):
        dfb,_=split_data(load_raw()); tok,M=get_tok_emb(**cfg)
        X,y=prepare(dfb,tok,cfg['maxlen']); m=build_lstm(cfg,M)
        with st.spinner('Training...'): h=m.fit(X,y,validation_split=0.1,epochs=3)
        m.save_weights(cfg['lstm_weights']); st.success('Done')

elif mode=='Train BiLSTM':
    st.title('Train BiLSTM')
    if weights_exist:
        st.info('BiLSTM weights exist')
    elif st.button('Train BiLSTM'):
        dfb,_=split_data(load_raw()); tok,M=get_tok_emb(**cfg)
        X,y=prepare(dfb,tok,cfg['maxlen']); m=build_bilstm(cfg,M)
        with st.spinner('Training...'): h=m.fit(X,y,validation_split=0.1,epochs=3)
        m.save_weights(cfg['bilstm_weights']); st.success('Done')

elif mode=='Inference':
    st.title('Inference')
    tok,M=get_tok_emb(**cfg)
    models={}
    for name,build,w in [('RNN',build_rnn,'rnn_weights'),('LSTM',build_lstm,'lstm_weights'),('BiLSTM',build_bilstm,'bilstm_weights')]:
        m=globals()[f'build_{name.lower()}'](cfg,M)
        try: m.load_weights(cfg[w]); models[name]=m
        except: st.error(f'Missing {name} weights')
    for name,m in models.items():
        txt=st.text_area(name,key=name)
        if st.button(f'Predict {name}',key='btn_'+name) and txt:
            X=pad_sequences(tok.texts_to_sequences([txt]),maxlen=cfg['maxlen'])
            p=m.predict(X)[0]; st.write(np.argmax(p),p)


# Dataset Exploration
elif app_mode == "Dataset Exploration":
    st.title("📊 Dataset Exploration")
    df_bal, _ = split_data(load_raw())
    dataset_visualization(df_bal)

# Tuning
elif app_mode == "Hyperparameter Tuning":
    st.title("🔧 Hyperparameter Tuning")
    tok, M = get_tok_emb(**cfg)
    if st.button("Run Tuning"):
        study = optuna.create_study(direction="maximize")
        @st.spinner("Tuning...")
        def obj(trial):
            u = trial.suggest_int("units", 32, 128)
            d = trial.suggest_float("dropout", 0.1, 0.5)
            mdl = build_bilstm(cfg, M, units=u, dropout=d)
            df_bal, _ = split_data(load_raw())
            X, y = prepare(df_bal, tok, cfg["maxlen"])
            h = mdl.fit(X, y, validation_split=0.1, epochs=3, batch_size=128, verbose=0)
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
