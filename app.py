"""
CNN Demo: Streamlit front end for CNN_Text_Classification.ipynb

Training happens in Colab. This app only LOADS the exported artifacts.

Folder layout:
    app.py
    requirements.txt
    artifacts/                  <- unzip cnn_artifacts.zip here
        sentiment_cnn1d.keras   tokenizer.json   config.json
        results.json            training_history.png
        mnist_cnn2d.keras       results_2d.json  training_history_2d.png
        mnist_samples.npz

Run:  streamlit run app.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps

ART = Path(__file__).parent / "artifacts"

st.set_page_config(page_title="CNN Demo", layout="wide")


# ---------------------------------------------------------------------------
# Loading (cached so models load once, not on every click)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model...")
def load_model(path: str):
    import tensorflow as tf  # imported here so the page appears quickly

    return tf.keras.models.load_model(path)


@st.cache_data
def load_json(path: str):
    with open(path) as f:
        return json.load(f)


@st.cache_data
def load_tokenizer(path: str):
    """Read the Keras Tokenizer JSON. We re-implement texts_to_sequences below,
    so the app doesn't depend on the legacy keras.preprocessing.text module."""
    cfg = load_json(path)["config"]
    word_index = cfg["word_index"]
    if isinstance(word_index, str):  # Keras stores it as a JSON string
        word_index = json.loads(word_index)
    return {
        "word_index": word_index,
        "lower": cfg.get("lower", True),
        "filters": cfg.get("filters", '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'),
        "split": cfg.get("split", " "),
        "num_words": cfg.get("num_words"),
        "oov_token": cfg.get("oov_token"),
    }


@st.cache_data
def load_samples(path: str):
    data = np.load(path)
    return data["images"], data["labels"]


def missing(*names):
    return [n for n in names if not (ART / n).exists()]


# ---------------------------------------------------------------------------
# Text preprocessing: same steps as the notebook's Tokenizer + pad_sequences
# ---------------------------------------------------------------------------
def text_to_sequence(text: str, tok: dict) -> list[int]:
    if tok["lower"]:
        text = text.lower()
    text = text.translate(str.maketrans({c: tok["split"] for c in tok["filters"]}))
    words = [w for w in text.split(tok["split"]) if w]

    wi = tok["word_index"]
    oov_idx = wi.get(tok["oov_token"]) if tok["oov_token"] else None
    num_words = tok["num_words"]
    seq = []
    for w in words:
        i = wi.get(w)
        if i is not None and (num_words is None or i < num_words):
            seq.append(i)
        elif oov_idx is not None:
            seq.append(oov_idx)
    return seq


def pad_post(seqs: list[list[int]], maxlen: int) -> np.ndarray:
    """padding='post', truncating='post', value 0 (matches training)."""
    out = np.zeros((len(seqs), maxlen), dtype="int32")
    for r, s in enumerate(seqs):
        s = s[:maxlen]
        out[r, : len(s)] = s
    return out


# ---------------------------------------------------------------------------
# Image preprocessing: make an uploaded picture look like an MNIST digit
# ---------------------------------------------------------------------------
def to_mnist(img: Image.Image) -> np.ndarray:
    g = ImageOps.grayscale(img)
    arr = np.asarray(g, dtype="float32")
    if arr.mean() > 127:  # MNIST = white digit on black background
        arr = 255 - arr
    arr[arr < 30] = 0  # remove light background noise

    ys, xs = np.nonzero(arr)
    if len(xs):  # crop to the digit
        arr = arr[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]

    # fit into 20x20 (keeping aspect ratio), then centre on a 28x28 canvas
    digit = Image.fromarray(arr.astype("uint8"))
    h, w = arr.shape
    scale = 20 / max(h, w)
    digit = digit.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)
    canvas = Image.new("L", (28, 28), 0)
    canvas.paste(digit, ((28 - digit.width) // 2, (28 - digit.height) // 2))
    return np.asarray(canvas, dtype="float32") / 255.0


def enlarge(img28: np.ndarray, size: int = 196) -> Image.Image:
    return Image.fromarray((img28.squeeze() * 255).astype("uint8")).resize((size, size), Image.NEAREST)


# ---------------------------------------------------------------------------
# Shared "model performance" section
# ---------------------------------------------------------------------------
def show_performance(results: dict, labels: list[str], history_png: str):
    m = results["metrics"]
    cols = st.columns(len(m))
    for col, (name, value) in zip(cols, m.items()):
        col.metric(name.replace("_", " ").title(), f"{value:.2%}")

    left, right = st.columns(2)
    with left:
        st.markdown("**Confusion matrix** (rows = true, columns = predicted)")
        cm = pd.DataFrame(
            results["confusion_matrix"],
            index=[f"True {l}" for l in labels],
            columns=[f"Pred {l}" for l in labels],
        )
        st.dataframe(cm.style.background_gradient(cmap="Blues"), width="stretch")
    with right:
        st.markdown(f"**Training history** (best epoch: {results['best_epoch']})")
        if (ART / history_png).exists():
            st.image(str(ART / history_png), width="stretch")
        else:
            st.line_chart(pd.DataFrame(results["history"]))


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
st.title("CNN Demo")
st.caption("A 1D CNN for sentence sentiment and a 2D CNN for handwritten digits. Both were trained in Google Colab.")

if not ART.exists():
    st.error("No `artifacts/` folder found. Unzip `cnn_artifacts.zip` into a folder named `artifacts` next to `app.py`.")
    st.stop()

with st.sidebar:
    st.header("About")
    st.write("This app loads the saved models and runs predictions. It does not retrain them.")
    if (ART / "config.json").exists():
        cfg = load_json(str(ART / "config.json"))
        import tensorflow as tf

        st.write(f"Trained with TensorFlow **{cfg['tensorflow_version']}** / Keras **{cfg['keras_version']}**")
        st.write(f"Running TensorFlow **{tf.__version__}**")
        if tf.__version__ != cfg["tensorflow_version"]:
            st.warning("TensorFlow version differs from training. If loading fails, match the versions in requirements.txt.")

tab_text, tab_img = st.tabs(["Text sentiment (1D CNN)", "Digit recognition (2D CNN)"])

# ----------------------------- 1D CNN -------------------------------------
with tab_text:
    need = missing("sentiment_cnn1d.keras", "tokenizer.json", "config.json")
    if need:
        st.warning(f"Missing files: {', '.join(need)}")
    else:
        model_1d = load_model(str(ART / "sentiment_cnn1d.keras"))
        tok = load_tokenizer(str(ART / "tokenizer.json"))
        cfg = load_json(str(ART / "config.json"))

        st.subheader("Try your own sentences")
        text = st.text_area(
            "One sentence per line",
            value="The movie was brilliant and memorable\nThe story was weak and disappointing\nThe movie was not good",
            height=130,
        )
        threshold = st.slider("Decision threshold for Positive", 0.05, 0.95, float(cfg.get("threshold", 0.5)), 0.05)

        if st.button("Predict sentiment", type="primary"):
            sentences = [s.strip() for s in text.splitlines() if s.strip()]
            if not sentences:
                st.info("Type at least one sentence.")
            else:
                seqs = [text_to_sequence(s, tok) for s in sentences]
                x = pad_post(seqs, cfg["max_length"])
                probs = model_1d.predict(x, verbose=0).ravel()
                oov = cfg.get("oov_index", 1)

                rows = []
                for s, seq, p in zip(sentences, seqs, probs):
                    pos = p >= threshold
                    rows.append({
                        "Sentence": s,
                        "Prediction": "Positive" if pos else "Negative",
                        "Confidence": float(p if pos else 1 - p),
                        "P(positive)": float(p),
                        "Unknown words": seq.count(oov),
                        "Truncated": len(seq) > cfg["max_length"],
                    })
                df = pd.DataFrame(rows)
                st.dataframe(
                    df,
                    hide_index=True,
                    width="stretch",
                    column_config={
                        "Confidence": st.column_config.ProgressColumn(format="%.1f%%", min_value=0, max_value=1),
                        "P(positive)": st.column_config.NumberColumn(format="%.3f"),
                    },
                )
                if df["Unknown words"].gt(0).any():
                    st.caption(
                        f"Unknown words are not in the training vocabulary ({cfg['vocab_size'] - 1} words), "
                        "so the model can't use them. Predictions for those sentences are less reliable."
                    )
                if df["Truncated"].any():
                    st.caption(f"Sentences longer than {cfg['max_length']} words were cut off, as in training.")

        if (ART / "results.json").exists():
            with st.expander("Model performance on the test set", expanded=False):
                show_performance(load_json(str(ART / "results.json")), ["Neg", "Pos"], "training_history.png")

# ----------------------------- 2D CNN -------------------------------------
with tab_img:
    need = missing("mnist_cnn2d.keras")
    if need:
        st.warning("Missing `mnist_cnn2d.keras`. Run the 2D CNN cells in Colab and export the zip again.")
    else:
        model_2d = load_model(str(ART / "mnist_cnn2d.keras"))
        st.subheader("Classify a handwritten digit")

        has_samples = (ART / "mnist_samples.npz").exists()
        options = (["Sample test image"] if has_samples else []) + ["Upload an image"]
        source = st.radio("Image source", options, horizontal=True)

        img28, true_label = None, None
        if source == "Sample test image":
            images, labels = load_samples(str(ART / "mnist_samples.npz"))
            i = st.selectbox(
                "Pick a sample",
                range(len(images)),
                format_func=lambda k: f"Sample {k + 1} (true digit {labels[k]})",
            )
            img28, true_label = images[i].squeeze(), int(labels[i])
        else:
            up = st.file_uploader("PNG or JPG of a single digit", type=["png", "jpg", "jpeg"])
            st.caption("Dark digit on a light background works. It is auto-inverted, cropped and centred like MNIST.")
            if up:
                img28 = to_mnist(Image.open(up))

        if img28 is not None:
            probs = model_2d.predict(img28.reshape(1, 28, 28, 1), verbose=0)[0]
            pred = int(np.argmax(probs))

            c1, c2 = st.columns([1, 2])
            with c1:
                st.image(enlarge(img28), caption="What the model sees (28x28)")
            with c2:
                m1, m2 = st.columns(2)
                m1.metric("Predicted digit", pred)
                m2.metric("Confidence", f"{probs[pred]:.1%}")
                if true_label is not None:
                    if pred == true_label:
                        st.success(f"Correct. The true digit is {true_label}.")
                    else:
                        st.error(f"Wrong. The true digit is {true_label}.")
                st.bar_chart(pd.DataFrame({"Probability": probs}, index=[str(d) for d in range(10)]))

        if (ART / "results_2d.json").exists():
            with st.expander("Model performance on the MNIST test set", expanded=False):
                res2 = load_json(str(ART / "results_2d.json"))
                show_performance(res2, [str(d) for d in range(10)], "training_history_2d.png")
                if res2.get("top_confusions"):
                    st.markdown("**Most common mistakes**")
                    st.dataframe(
                        pd.DataFrame(res2["top_confusions"]).rename(
                            columns={"true": "True digit", "predicted": "Predicted as", "count": "Images"}
                        ),
                        hide_index=True,
                    )
