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
    /* Fondo general de la app */
    .stApp {
      background-color: #000702;
    }
    /* Sidebar */
    [data-testid=\"stSidebar\"] > div:first-child {
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

# Semilla aleatoria
tf.random.set_seed(42)

# Configuración
cfg = {
    "max_features": 10000,
    "maxlen": 300,
    "emb_dim": 300,
    "bilstm_weights": "model_weights.weights.h5",
    "lstm_weights": "lstm_model_weights.weights.h5",
    "rnn_weights":  "rnn_model_weights.weights.h5"
}

@st.cache_data
def load_raw():
    ds = load_dataset("tweets_hate_speech_detection", split="train")
    return ds.to_pandas()

@st.cache_resource
def get_tok_emb(max_features, maxlen, emb_dim, **kwargs):
    df_train, _ = split_data(load_raw())
    tok = Tokenizer(num_words=max_features, oov_token="<OOV>")
    tok.fit_on_texts(df_train.tweet.astype(str))
    model = api.load("glove-wiki-gigaword-300")
    M = np.zeros((max_features, emb_dim))
    for w, i in tok.word_index.items():
        if i < max_features and w in model:
            M[i] = model[w]
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

    # 1. Class Distribution
    st.subheader("1. Class Distribution")
    fig, ax = plt.subplots()
    df.label.value_counts().plot.bar(ax=ax)
    fig.patch.set_facecolor('#3d3e36')
    ax.set_facecolor('#3d3e36')
    axes_color = '#5d612e'
    ax.spines['bottom'].set_color(axes_color)
    ax.spines['left'].set_color(axes_color)
    ax.set_xlabel('Class', color='white')
    ax.set_ylabel('Count', color='white')
    ax.tick_params(colors='white')
    st.pyplot(fig)

    # 2. Tweet Length Distribution
    st.subheader("2. Tweet Length Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df['length'], bins=50, kde=True, ax=ax)
    fig.patch.set_facecolor('#3d3e36')
    ax.set_facecolor('#3d3e36')
    ax.set_xlabel('Length', color='white')
    ax.tick_params(colors='white')
    st.pyplot(fig)

    # 3. Boxplot Length by Class
    st.subheader("3. Length by Class Boxplot")
    fig, ax = plt.subplots()
    sns.boxplot(x='label', y='length', data=df, ax=ax)
    fig.patch.set_facecolor('#3d3e36')
    ax.set_facecolor('#3d3e36')
    ax.set_xlabel('Class', color='white')
    ax.set_ylabel('Length', color='white')
    ax.tick_params(colors='white')
    st.pyplot(fig)

    # 4. Top 20 Words
    st.subheader("4. Top 20 Most Common Words")
    words = df.tweet.str.cat(sep=' ').split()
    freq = pd.Series(words).value_counts().head(20)
    fig, ax = plt.subplots()
    freq.plot.barh(ax=ax)
    ax.invert_yaxis()
    fig.patch.set_facecolor('#3d3e36')
    ax.set_facecolor('#3d3e36')
    ax.tick_params(colors='white')
    st.pyplot(fig)

    # 5. Word Cloud
    st.subheader("5. Word Cloud")
    wc = WordCloud(width=800, height=400, background_color='white').generate(' '.join(words))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    fig.patch.set_facecolor('#3d3e36')
    ax.set_facecolor('#3d3e36')
    st.pyplot(fig)

    # 6. Top 15 Bigrams
    from collections import Counter
    bigrams = list(zip(words, words[1:]))
    bigram_freq = Counter(bigrams).most_common(15)
    labels, counts = zip(*bigram_freq)
    labels = [' '.join(b) for b in labels]
    fig, ax = plt.subplots()
    pd.Series(counts, index=labels).plot.bar(ax=ax)
    fig.patch.set_facecolor('#3d3e36')
    ax.set_facecolor('#3d3e36')
    ax.tick_params(colors='white')
    st.pyplot(fig)

    # 7. Sample Tweets by Quartile
    st.subheader("7. Sample Tweets by Length Quartile")
    for q in [0.25, 0.5, 0.75]:
        tweet = df[df['length'] <= df['length'].quantile(q)].tweet.sample(1).values[0]
        st.write(f"- Quartile {int(q*100)}%: {tweet}")

# Navegación
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

# Entrenamiento Simple RNN
if app_mode == "Train Simple RNN":
    st.title("🚀 Train Simple RNN Model")
    if st.button("Start Training RNN"):
        df_bal, _ = split_data(load_raw())
        tok, M = get_tok_emb(**cfg)
        X, y = prepare(df_bal, tok, cfg["maxlen"])
        model = build_rnn(cfg, M)
        with st.spinner("Training RNN..."):
            h = model.fit(X, y, validation_split=0.1, epochs=3, batch_size=128)
            model.save_weights(cfg["rnn_weights"])
        st.success("Simple RNN trained")
        st.write(f"Accuracy: {h.history['accuracy'][-1]:.3f}")

# Entrenamiento LSTM
elif app_mode == "Train LSTM":
    st.title("🚀 Train LSTM Model")
    if st.button("Start Training LSTM"):
        df_bal, _ = split_data(load_raw())
        tok, M = get_tok_emb(**cfg)
        X, y = prepare(df_bal, tok, cfg["maxlen"])
        model = build_lstm(cfg, M)
        with st.spinner("Training LSTM..."):
            h = model.fit(X, y, validation_split=0.1, epochs=3, batch_size=128)
            model.save_weights(cfg["lstm_weights"])
        st.success("LSTM trained")
        st.write(f"Accuracy: {h.history['accuracy'][-1]:.3f}")

# Entrenamiento BiLSTM
elif app_mode == "Train BiLSTM":
    st.title("🚀 Train BiLSTM Model")
    if st.button("Start Training BiLSTM"):
        df_bal, _ = split_data(load_raw())
        tok, M = get_tok_emb(**cfg)
        X, y = prepare(df_bal, tok, cfg["maxlen"])
        model = build_bilstm(cfg, M)
        with st.spinner("Training BiLSTM..."):
            h = model.fit(X, y, validation_split=0.1, epochs=3, batch_size=128)
            model.save_weights(cfg["bilstm_weights"])
        st.success("BiLSTM trained")
        st.write(f"Accuracy: {h.history['accuracy'][-1]:.3f}")

# Inference
elif app_mode == "Inference":
    st.title("📝 Text Classification Inference")
    tok, M = get_tok_emb(**cfg)
    models = {}
    for name, builder, w in [
        ("Simple RNN", build_rnn, cfg["rnn_weights"]),
        ("LSTM", build_lstm, cfg["lstm_weights"]),
        ("BiLSTM", build_bilstm, cfg["bilstm_weights"])
    ]:
        m = builder(cfg, M)
        try:
            m.load_weights(w)
            models[name] = m
        except:
            st.error(f"Missing weights for {name}")
    for name, m in models.items():
        txt = st.text_area(f"Enter text for {name}", key=f"txt_{name}")
        if st.button(f"Predict {name}", key=f"btn_{name}") and txt:(f"Predict {name}", key=name):
            X = pad_sequences(tok.texts_to_sequences([txt]), maxlen=cfg["maxlen"])
            p = m.predict(X)[0]
            st.write(f"Class: {np.argmax(p)}, Conf: {p}")

# Exploración
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
