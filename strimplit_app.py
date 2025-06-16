import os
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

# Configuración de rutas de pesos
ebdim = 300
cfg = {
    "max_features": 10000,
    "maxlen": 300,
    "emb_dim": ebdim,
    "rnn_weights":  "rnn_model_weights.weights.h5",
    "lstm_weights": "lstm_model_weights.weights.h5",
    "bilstm_weights": "model_weights.weights.h5"
}

# Verificación de existencia de pesos individuales
weights_exist = {
    "Simple RNN": os.path.exists(cfg["rnn_weights"]),
    "LSTM": os.path.exists(cfg["lstm_weights"]),
    "BiLSTM": os.path.exists(cfg["bilstm_weights"])  
}

@st.cache_data
def load_raw():
    ds = load_dataset("tweets_hate_speech_detection", split="train")
    return ds.to_pandas()

@st.cache_resource
def get_tokenizer(max_features):
    df_train, _ = split_data(load_raw())
    tok = Tokenizer(num_words=max_features, oov_token="<OOV>")
    tok.fit_on_texts(df_train.tweet.astype(str))
    return tok


def split_data(df):
    df_train, df_val = train_test_split(df, test_size=0.2, stratify=df.label, random_state=42)
    maj = df_train[df_train.label == 0]
    min_ = df_train[df_train.label == 1]
    min_up = resample(min_, replace=True, n_samples=len(maj), random_state=42)
    return pd.concat([maj, min_up]).sample(frac=1, random_state=42), df_val


def build_rnn(cfg, units=64, dropout=0.2):
    model = Sequential([
        Input(shape=(cfg["maxlen"],)),
        Embedding(cfg["max_features"], cfg["emb_dim"], trainable=True),
        SimpleRNN(units, dropout=dropout, recurrent_dropout=dropout),
        Dense(2, activation="softmax")
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def build_lstm(cfg, units=64, dropout=0.2):
    model = Sequential([
        Input(shape=(cfg["maxlen"],)),
        Embedding(cfg["max_features"], cfg["emb_dim"], trainable=True),
        LSTM(units, dropout=dropout, recurrent_dropout=dropout),
        Dense(2, activation="softmax")
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def build_bilstm(cfg, units=64, dropout=0.2):
    model = Sequential([
        Input(shape=(cfg["maxlen"],)),
        Embedding(cfg["max_features"], cfg["emb_dim"], trainable=True),
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
    fig.patch.set_facecolor('#3d3e36'); ax.set_facecolor('#3d3e36'); ax.tick_params(colors='white')
    st.pyplot(fig)

    st.subheader("2. Tweet Length Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df['length'], bins=50, kde=True, ax=ax)
    fig.patch.set_facecolor('#3d3e36'); ax.set_facecolor('#3d3e36'); ax.tick_params(colors='white')
    st.pyplot(fig)

    st.subheader("3. Length by Class Boxplot")
    fig, ax = plt.subplots()
    sns.boxplot(x='label', y='length', data=df, ax=ax)
    fig.patch.set_facecolor('#3d3e36'); ax.set_facecolor('#3d3e36'); ax.tick_params(colors='white')
    st.pyplot(fig)

    st.subheader("4. Top 20 Most Common Words")
    words = df.tweet.str.cat(sep=' ').split()
    freq = pd.Series(words).value_counts().head(20)
    fig, ax = plt.subplots()
    freq.plot.barh(ax=ax); ax.invert_yaxis()
    fig.patch.set_facecolor('#3d3e36'); ax.set_facecolor('#3d3e36'); ax.tick_params(colors='white')
    st.pyplot(fig)

    st.subheader("5. Word Cloud")
    wc = WordCloud(width=800, height=400, background_color='white').generate(' '.join(words))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wc, interpolation='bilinear'); ax.axis('off')
    fig.patch.set_facecolor('#3d3e36'); ax.set_facecolor('#3d3e36')
    st.pyplot(fig)

    st.subheader("6. Top 15 Bigrams")
    from collections import Counter
    bigrams = list(zip(words, words[1:]))
    labels, counts = zip(*Counter(bigrams).most_common(15))
    labels = [' '.join(b) for b in labels]
    fig, ax = plt.subplots()
    pd.Series(counts, index=labels).plot.bar(ax=ax)
    fig.patch.set_facecolor('#3d3e36'); ax.set_facecolor('#3d3e36'); ax.tick_params(colors='white')
    st.pyplot(fig)

    st.subheader("7. Sample Tweets by Length Quartile")
    for q in [0.25, 0.5, 0.75]:
        tweet = df[df['length'] <= df['length'].quantile(q)].tweet.sample(1).iat[0]
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

# Entrenamiento de modelos
if app_mode.startswith("Train"):
    model_map = {
        "Train Simple RNN": (build_rnn, cfg["rnn_weights"], "Simple RNN"),
        "Train LSTM": (build_lstm, cfg["lstm_weights"], "LSTM"),
        "Train BiLSTM": (build_bilstm, cfg["bilstm_weights"], "BiLSTM")
    }
    build_fn, wpath, name = model_map[app_mode]
    st.title(f"🚀 {app_mode} Model")
    if weights_exist[name]:
        st.info(f"{name} weights already exist. Go to Inference to use the model.")
    else:
        if st.button(f"Start Training {name}"):
            df_bal, _ = split_data(load_raw())
            tok = get_tokenizer(cfg["max_features"])
            X, y = prepare(df_bal, tok, cfg["maxlen"])
            model = build_fn(cfg)
            with st.spinner(f"Training {name}..."):
                try:
                    h = model.fit( X, y, validation_split=0.1, epochs=1, batch_size=32 )
                except Exception as e:
                    st.error(f"Training failed for {name}: {e}")
                    return
                model.save_weights(wpath)
                weights_exist[name] = True
            st.success(f"{name} trained")
            st.write(f"Accuracy: {h.history['accuracy'][-1]:.3f}")

# Inferencia
elif app_mode == "Inference":
    st.title("📝 Text Classification Inference")
    st.write("Working dir:", os.getcwd())
    st.write("Files:", os.listdir("."))
    tok = get_tokenizer(cfg["max_features"])
    models = {}
    for name, build_fn, wpath in [
        ("Simple RNN", build_rnn, cfg["rnn_weights"]),
        ("LSTM", build_lstm, cfg["lstm_weights"]),
        ("BiLSTM", build_bilstm, cfg["bilstm_weights"])  
    ]:
        exists = os.path.exists(wpath)
        st.write(f"{name} weights exist? {exists}")
        if exists:
            try:
                m = build_fn(cfg)
                m.load_weights(wpath)
                models[name] = m
            except Exception as e:
                st.error(f"Error loading {name} weights: {e}")
        else:
            st.error(f"Missing weights for {name}")

    for name, m in models.items():
        txt = st.text_area(f"Enter text for {name}", key=name)
        if st.button(f"Predict {name}", key=f"btn_{name}") and txt:
            X = pad_sequences(tok.texts_to_sequences([txt]), maxlen=cfg["maxlen"])
            p = m.predict(X)[0]
            st.write(f"Class: {np.argmax(p)} (conf: {p.max():.3f})")

# Dataset Exploration
elif app_mode == "Dataset Exploration":
    st.title("📊 Dataset Exploration")
    df_bal, _ = split_data(load_raw())
    dataset_visualization(df_bal)

# Tuning
elif app_mode == "Hyperparameter Tuning":
    st.title("🔧 Hyperparameter Tuning")
    tok = get_tokenizer(cfg["max_features"])
    if st.button("Run Tuning"):
        study = optuna.create_study(direction="maximize")
        def objective(trial):
            u = trial.suggest_int("units", 32, 128)
            d = trial.suggest_float("dropout", 0.1, 0.5)
            mdl = build_bilstm(cfg, units=u, dropout=d)
            df_bal, _ = split_data(load_raw())
            X, y = prepare(df_bal, tok, cfg["maxlen"])
            h = mdl.fit(X, y, validation_split=0.1, epochs=3, batch_size=128, verbose=0)
            return h.history["val_accuracy"][-1]
        with st.spinner("Tuning..."):
            study.optimize(objective, n_trials=5)
        st.success("Tuning completed")
        st.json(study.best_params)
    else:
        try:
            study = optuna.load_study(study_name="optuna_study", storage="sqlite:///optuna.db")
            st.write("**Best parameters:**")
            st.json(study.best_params)
            st.plotly_chart(vis.plot_optimization_history(study))
        except:
            st.info("No tuning study found. Click 'Run Tuning' to start.")

# Model Analysis & Justification
else:
    st.title("🧮 Model Analysis & Justification")
    tok = get_tokenizer(cfg["max_features"])
    df_bal, df_val = split_data(load_raw())
    X_val, y_val = prepare(df_val, tok, cfg["maxlen"])
    y_true = np.argmax(y_val, axis=1)
    accuracies = {}
    for name, build_fn, wpath in [
        ("Simple RNN", build_rnn, cfg["rnn_weights"]),
        ("LSTM", build_lstm, cfg["lstm_weights"]),
        ("BiLSTM", build_bilstm, cfg["bilstm_weights"])  
    ]:
        if not weights_exist[name]:
            st.error(f"Missing weights for {name}. Please train first.")
            continue
        model = build_fn(cfg)
        try:
            model.load_weights(wpath)
        except Exception as e:
            st.error(f"Error loading {name} weights: {e}")
            continue
        st.subheader(f"{name} Analysis")
        y_prob = model.predict(X_val)
        y_pred = np.argmax(y_prob, axis=1)
        rep = classification_report(y_true, y_pred, digits=3, output_dict=True)
        st.dataframe(pd.DataFrame(rep).transpose())
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", ax=ax, cmap="Oranges")
        st.pyplot(fig)
        df_val["pred"] = y_pred
        st.markdown("**False Positives:**")
        fp = df_val[(df_val.label==0)&(df_val.pred==1)].tweet
        if not fp.empty:
            for t in fp.sample(min(3,len(fp)), random_state=42): st.write(f"- {t}")
        else:
            st.write("_None_")
        st.markdown("**False Negatives:**")
        fn = df_val[(df_val.label==1)&(df_val.pred==0)].tweet
        if not fn.empty:
            for t in fn.sample(min(3,len(fn)), random_state=42): st.write(f"- {t}")
        else:
            st.write("_None_")
        accuracies[name] = rep["accuracy"]
    if accuracies:
        best = max(accuracies, key=accuracies.get)
        st.subheader(f"🏆 Best Model: {best} ({accuracies[best]:.3f})")
        st.write("Chosen for highest validation accuracy.")
    else:
        st.error("No model performances available.")
