import os, re, json, tempfile, time
import streamlit as st
import mlflow
from mlflow import sklearn

# Text normalization: lowercase, remove punctuation, collapse whitespace
def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Define project paths and MLflow tracking URI (fallbacks to local 'mlruns')
PROJECT_ROOT  = os.path.abspath(os.path.dirname(__file__))
DEFAULT_STORE = os.path.join(PROJECT_ROOT, "mlruns")
TRACK_URI     = os.getenv(
    "MLFLOW_TRACKING_URI",
    f"file:///{DEFAULT_STORE.replace(os.sep, '/')}"
)
mlflow.set_tracking_uri(TRACK_URI)
mlflow.set_experiment("ai-text-clf")

@st.cache_resource
# Load the latest Random Forest gridsearch run artifacts
def load_artifacts():
    exp = mlflow.get_experiment_by_name("ai-text-clf")
    if exp is None:
        raise RuntimeError("No experiment 'ai-text-clf' at " + TRACK_URI)

    # Find the most recent finished run tagged 'rf-gridsearch'
    runs = mlflow.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string=(
            "attributes.status = 'FINISHED' "
            "and tags.mlflow.runName = 'rf-gridsearch'"
        ),
        order_by=["start_time DESC"],
        max_results=1
    )

    if runs.empty:
        raise RuntimeError("No rf-gridsearch training runs found.")

    row    = runs.iloc[0]
    run_id = row["run_id"]

    # Load model and vectorizer from the run
    model  = sklearn.load_model(f"runs:/{run_id}/model")
    vect   = sklearn.load_model(f"runs:/{run_id}/vectorizer")

    # Attempt to download the label map, fallback to default if missing
    try:
        map_path = mlflow.artifacts.download_artifacts(
            f"runs:/{run_id}/label_map.json",
            dst_path=tempfile.mkdtemp()
        )
        with open(map_path) as f:
            label_map = json.load(f)
    except Exception:
        label_map = {"AI": 0, "Human": 1}

    return model, vect, label_map["AI"]

# Load artifacts once and cache for session
clf, vectorizer, AI_ID = load_artifacts()

tab_detect, tab_method = st.tabs(["🔍 Detector", "📖 Methodology"])

with tab_detect:
    st.header("AI vs Human Text Detector 🚦")
    text_in = st.text_area("Paste text here:", height=200)

    if st.button("Predict"):
        # Preprocess and vectorize input
        clean = preprocess_text(text_in)
        X = vectorizer.transform([clean])

        # Generate prediction and probability
        pred = clf.predict(X)[0]
        ai_p = clf.predict_proba(X)[0][AI_ID] * 100

        # Measure inference latency
        latency = (
            time.time() - st.session_state.get("_t0", time.time())
        ) * 1000

        label = "AI-Generated" if pred == AI_ID else "Human-Written"
        st.success(f"**{label}**  \nAI probability ≈ {ai_p:,.1f}%  •  {latency:.0f} ms")

        # Log inference details to MLflow under a nested run
        with mlflow.start_run(run_name="inference", nested=True) as run:
            mlflow.log_params({
                "text_length": len(text_in),
                "predicted_class": int(pred)
            })
            mlflow.log_metrics({
                "latency_ms": latency,
                "ai_probability": ai_p
            })

        # Reset timer for next prediction
        st.session_state["_t0"] = time.time()

        st.divider()
        st.subheader("MLflow Tracking Logs for This Prediction")

        st.markdown("**Parameters**")
        st.json({
            "text_length": len(text_in),
            "predicted_class": int(pred)
        })

        st.markdown("**Metrics**")
        st.json({
            "latency_ms": latency,
            "ai_probability": ai_p
        })

        st.markdown(f"**Run ID:** `{run.info.run_id}`")

with tab_method:
    st.header("Underlying Methodology")
    st.markdown(
    """
### Dataset  
The dataset used for this project is the **[AI Text Detection Pile](https://huggingface.co/datasets/artem9k/ai-text-detection-pile)** by Artem Yatsenko.  
* **Size:** 1.39 M documents  
* **Class balance:** 73.8 % human / 26.2 % AI  
* **Human sources:** Reddit Writing Prompts, OpenAI WebText, HC3, IvyPanda essays  
* **AI sources:** GPT-2, GPT-3, GPT-J, ChatGPT

### Pre-processing  
1. Lower-case  
2. Remove punctuation / extra whitespace  
3. Tokenise & build a TF-IDF (1–2 grams, 5 000 features, English stop-words removed)

### Model Selection  
Evaluated five classifiers (LogReg, RF, GBoost, SVM, K-NN).  
Random Forest + GridSearch gave the best **macro F1** and was exported.

### Deployment pipeline  
```text
raw text → cleaning → TF-IDF → Random Forest → label
```

### MLflow Tracking  
Training, hyper-parameters, model & vectorizer are logged to **./mlruns**.

**Open the MLflow UI locally**

```powershell
# in project root (same folder that contains 'mlruns')
python -m mlflow ui `
  --backend-store-uri "file:///%CD%/mlruns" `
  --port 5000

#Then browse to http://localhost:5000. """, unsafe_allow_html=True )
