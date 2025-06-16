import os
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
import requests
import tempfile
import os

# —————————————————————————————————————————————————————
# Custom CSS styling
# —————————————————————————————————————————————————————
st.markdown(
    """
    <style>
    .stApp { background-color: #000702; }
    [data-testid="stSidebar"] > div:first-child { background-color: #1f2c13; }
    .css-1v0mbdj, .css-1d391kg { color: #d4903b !important; }
    button { background-color: #a94802 !important; color: #ffffff !important; }
    </style>
    """,
    unsafe_allow_html=True
)

# For reproducibility
tf.random.set_seed(42)

emb_dim = 300
REPO_URL = "https://drive.google.com/uc?id="

cfg = {
    "max_features": 10000,
    "maxlen": 300,
    "emb_dim": emb_dim,
    "rnn_weights": REPO_URL + "1-NeKSzLnwFz8ZpFIYZK7YgJJAp36idJJ",
    "lstm_weights": REPO_URL + "1Ij5J90SCvcrSgnpGeTfe9WOcwOoiMv6t",
    "bilstm_weights": REPO_URL + "1knBcIKITOs47mxtLtCC8hXR4edzxx28g",
}

@st.cache_data
def load_raw():
    ds = load_dataset("tweets_hate_speech_detection", split="train")
    return ds.to_pandas()

@st.cache_data
def get_tokenizer(max_features):
    df_train, _ = split_data(load_raw())
    tok = Tokenizer(num_words=max_features, oov_token="<OOV>")
    tok.fit_on_texts(df_train.tweet.astype(str))
    return tok

def split_data(df):
    df_train, df_val = train_test_split(
        df, test_size=0.2, stratify=df.label, random_state=42
    )
    maj = df_train[df_train.label == 0]
    min_ = df_train[df_train.label == 1]
    min_up = resample(min_, replace=True, n_samples=len(maj), random_state=42)
    df_bal = pd.concat([maj, min_up]).sample(frac=1, random_state=42)
    return df_bal, df_val

# Model builders without pre-trained embeddings

def build_rnn(cfg, units=64, dropout=0.2):
    model = Sequential([
        Input(shape=(cfg["maxlen"],)),
        Embedding(cfg["max_features"], cfg["emb_dim"]),
        SimpleRNN(units, dropout=dropout, recurrent_dropout=dropout),
        Dense(2, activation="softmax")
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def build_lstm(cfg, units=64, dropout=0.2):
    model = Sequential([
        Input(shape=(cfg["maxlen"],)),
        Embedding(cfg["max_features"], cfg["emb_dim"]),
        LSTM(units, dropout=dropout, recurrent_dropout=dropout),
        Dense(2, activation="softmax")
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def build_bilstm(cfg, units=64, dropout=0.2):
    model = Sequential([
        Input(shape=(cfg["maxlen"],)),
        Embedding(cfg["max_features"], cfg["emb_dim"]),
        Bidirectional(LSTM(units, dropout=dropout, recurrent_dropout=dropout)),
        Dense(2, activation="softmax")
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model

@st.cache_resource
def download_weights(url):
    response = requests.get(url)
    response.raise_for_status()
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.h5')
    temp_file.write(response.content)
    temp_file.close()
    return temp_file.name

loaded = {}

for name, builder, wurl in [
    ("Simple RNN", build_rnn, cfg["rnn_weights"]),
    ("LSTM", build_lstm, cfg["lstm_weights"]),
    ("BiLSTM", build_bilstm, cfg["bilstm_weights"]),
]:
    model = builder(cfg)
    try:
        local_path = download_weights(wurl)
        model.load_weights(local_path)
        loaded[name] = model
        os.unlink(local_path)  # borra el archivo temporal después de cargar
    except Exception as e:
        st.error(f"Error loading weights for {name}: {e}")


def prepare(df, tokenizer, maxlen, num_classes=2):
    seqs = tokenizer.texts_to_sequences(df.tweet.astype(str))
    X = pad_sequences(seqs, maxlen=maxlen)
    y = to_categorical(df.label.values, num_classes=num_classes)
    return X, y

# Visualization helper stays the same
def dataset_visualization(df_bal):
    df = df_bal.copy()
    df['length'] = df.tweet.str.split().str.len()

    st.subheader("1. Class Distribution")
    fig, ax = plt.subplots()
    df.label.value_counts().plot.bar(ax=ax)
    fig.patch.set_facecolor('#3d3e36')
    ax.set_facecolor('#3d3e36')
    ax.tick_params(colors='white')
    st.pyplot(fig)

    st.subheader("2. Tweet Length Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df['length'], bins=50, kde=True, ax=ax)
    fig.patch.set_facecolor('#3d3e36')
    ax.set_facecolor('#3d3e36')
    ax.tick_params(colors='white')
    st.pyplot(fig)

    st.subheader("3. Length by Class Boxplot")
    fig, ax = plt.subplots()
    sns.boxplot(x='label', y='length', data=df, ax=ax)
    fig.patch.set_facecolor('#3d3e36')
    ax.set_facecolor('#3d3e36')
    ax.tick_params(colors='white')
    st.pyplot(fig)

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

    st.subheader("5. Word Cloud")
    wc = WordCloud(width=800, height=400, background_color='white').generate(' '.join(words))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    fig.patch.set_facecolor('#3d3e36')
    ax.set_facecolor('#3d3e36')
    st.pyplot(fig)

    st.subheader("6. Top 15 Bigrams")
    from collections import Counter
    bigrams = list(zip(words, words[1:]))
    bigram_freq = Counter(bigrams).most_common(15)
    labels, counts = zip(*bigram_freq)
    labels = [' '.join(b) for b in labels]
    fig, ax = plt.subplots()
    pd.Series(counts, index=labels).plot.bar(ax=ax)
    fig.patch.set_facecolor('#3d3e36')
    ax.set_facecolor('#3d3e36')
    ax.tick_params(colors='black')
    st.pyplot(fig)

    st.subheader("7. Sample Tweets by Length Quartile")
    for q in [0.25, 0.5, 0.75]:
        tweet = df[df['length'] <= df['length'].quantile(q)].tweet.sample(1).values[0]
        st.write(f"- Quartile {int(q*100)}%: {tweet}")

# Sidebar & Navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio("Go to", [
    "Train Simple RNN", "Train LSTM", "Train BiLSTM",
    "Inference", "Dataset Exploration",
    "Hyperparameter Tuning", "Model Analysis & Justification"
])

tokenizer = get_tokenizer(cfg["max_features"])

# Train Simple RNN
if app_mode == "Train Simple RNN":
    st.title("🚀 Train the Simple RNN Model")
    if st.button("Start Training LSTM"):
        tf.keras.backend.clear_session()
        model = build_lstm(cfg)
        try:
            with st.spinner("Training LSTM..."):
                history = model.fit(
                    X_train, y_train,
                    validation_split=0.1,
                    epochs=3,
                    batch_size=64,  # prueba con 64 si 128 consume mucha RAM
                    verbose=1
                )
                model.save_weights(cfg["lstm_weights"])
            st.success("LSTM trained and saved")
            st.write(f"Accuracy: {history.history['accuracy'][-1]:.3f}")
        except Exception as e:
            st.error(f"Training failed: {e}")


# Train LSTM
elif app_mode == "Train LSTM":
    st.title("🚀 Train the LSTM Model")
    if st.button("Start Training LSTM"):
        df_train_bal, _ = split_data(load_raw())
        X_train, y_train = prepare(df_train_bal, tokenizer, cfg["maxlen"])
        model = build_lstm(cfg)
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
        X_train, y_train = prepare(df_train_bal, tokenizer, cfg["maxlen"])
        model = build_bilstm(cfg)
        with st.spinner("Training BiLSTM..."):
            history = model.fit(X_train, y_train, validation_split=0.1, epochs=3, batch_size=128)
            model.save_weights(cfg["bilstm_weights"])
        st.success("BiLSTM trained and saved")
        st.write(f"Accuracy: {history.history['accuracy'][-1]:.3f}")

# Inference
elif app_mode == "Inference":
    st.title("📝 Text Classification Inference")
    models = {}
    for name, builder, wpath in [
        ("Simple RNN", build_rnn, cfg["rnn_weights"]),
        ("LSTM", build_lstm, cfg["lstm_weights"]),
        ("BiLSTM", build_bilstm, cfg["bilstm_weights"]),
    ]:
        m = builder(cfg)
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
    if st.button("Run Tuning"):
        study = optuna.create_study(direction="maximize", storage="sqlite:///optuna.db", load_if_exists=True)
        @st.spinner("Tuning in progress...")
        def run():
            def objective(trial):
                u = trial.suggest_int("lstm_units", 32, 128)
                d = trial.suggest_float("dropout_rate", 0.1, 0.5)
                model = build_bilstm(cfg, units=u, dropout=d)
                df_bal, _ = split_data(load_raw())
                X, y = prepare(df_bal, tokenizer, cfg["maxlen"])
                h = model.fit(X, y, validation_split=0.1, epochs=3, batch_size=128, verbose=0)
                return h.history["val_accuracy"][ -1 ]
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
    loaded = {}
    for name, builder, wpath in [
        ("Simple RNN", build_rnn, cfg["rnn_weights"]),
        ("LSTM", build_lstm, cfg["lstm_weights"]),
        ("BiLSTM", build_bilstm, cfg["bilstm_weights"]),
    ]:
        m = builder(cfg)
        try:
            m.load_weights(wpath)
            loaded[name] = m
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
