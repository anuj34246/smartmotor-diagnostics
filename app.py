import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import kurtosis, skew
from scipy.fft import fft
import joblib

# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="Motor Fault Detection",
    page_icon="⚙️",
    layout="wide"
)

# =========================
# CUSTOM CSS
# =========================

# =========================
# MODERN UI CSS
# =========================

st.markdown("""
<style>

/* =========================
MAIN BACKGROUND
========================= */

.stApp {
    background: linear-gradient(135deg, #0F172A, #111827, #1E293B);
    color: white;
}

/* =========================
REMOVE STREAMLIT DEFAULT
========================= */

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* =========================
MAIN CONTAINER
========================= */

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* =========================
TITLE
========================= */

h1 {
    text-align: center;
    font-size: 3.5rem !important;
    font-weight: 800;
    background: linear-gradient(to right, #00AAFF, #00FFAA);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 10px;
}

/* =========================
SUBHEADINGS
========================= */

h2, h3 {
    color: #00FFAA !important;
    text-align: center;
}

/* =========================
TEXT
========================= */

p {
    text-align: center;
    color: #E5E7EB;
    font-size: 18px;
}

/* =========================
UPLOAD BOX
========================= */

[data-testid="stFileUploader"] {
    background-color: rgba(255,255,255,0.05);
    border: 2px dashed #00AAFF;
    border-radius: 20px;
    padding: 20px;
    margin-top: 20px;
}

/* =========================
BUTTON
========================= */

.stButton > button {

    background: linear-gradient(90deg, #00AAFF, #00FFAA);

    color: white;

    border: none;

    border-radius: 12px;

    height: 3.5em;

    width: 100%;

    font-size: 18px;

    font-weight: bold;

    transition: 0.3s;

    box-shadow: 0px 0px 15px rgba(0,170,255,0.4);
}

.stButton > button:hover {

    transform: scale(1.03);

    box-shadow: 0px 0px 25px rgba(0,255,170,0.6);
}

/* =========================
SUCCESS BOX
========================= */

.stSuccess {

    background-color: rgba(0,255,170,0.15);

    border-radius: 15px;

    padding: 15px;

    border-left: 5px solid #00FFAA;
}

/* =========================
DATAFRAME
========================= */

[data-testid="stDataFrame"] {

    border-radius: 15px;

    overflow: hidden;

    border: 1px solid rgba(255,255,255,0.1);
}

/* =========================
METRIC STYLE
========================= */

[data-testid="metric-container"] {

    background-color: rgba(255,255,255,0.05);

    border: 1px solid rgba(255,255,255,0.1);

    padding: 15px;

    border-radius: 15px;

    box-shadow: 0px 0px 10px rgba(0,170,255,0.2);
}

/* =========================
PLOT CONTAINER
========================= */

.element-container:has(canvas) {

    background-color: rgba(255,255,255,0.03);

    padding: 15px;

    border-radius: 20px;

    box-shadow: 0px 0px 15px rgba(0,0,0,0.3);
}

/* =========================
SCROLLBAR
========================= */

::-webkit-scrollbar {
    width: 10px;
}

::-webkit-scrollbar-thumb {
    background: #00AAFF;
    border-radius: 10px;
}

/* =========================
CARD EFFECT
========================= */

.css-1r6slb0 {

    background-color: rgba(255,255,255,0.04);

    border-radius: 20px;

    padding: 20px;
}

</style>
""", unsafe_allow_html=True)

# =========================
# TITLE
# =========================

# =========================
# HERO SECTION
# =========================

st.markdown("""
<div style='text-align:center; padding:20px;'>

<img src='https://cdn-icons-png.flaticon.com/512/602/602182.png'
width='120'>

<h1>SmartMotor Diagnostics</h1>

<p style='font-size:22px; color:#D1D5DB;'>

SVM Trained Motor Fault Detection System

</p>

<p style='font-size:18px; color:#9CA3AF;'>

✅ Healthy Condition &nbsp;&nbsp;&nbsp;
✅ Inner Race Fault &nbsp;&nbsp;&nbsp;
✅ Outer Race Fault &nbsp;&nbsp;&nbsp;
✅ Ball Fault

</p>

</div>
""", unsafe_allow_html=True)

# =========================
# LOAD MODEL
# =========================

model = joblib.load("svm_model.pkl")
scaler = joblib.load("scaler.pkl")

# =========================
# FEATURE EXTRACTION
# =========================

def extract_features(seg):

    # -------------------
    # TIME DOMAIN FEATURES
    # -------------------

    rms = np.sqrt(np.mean(seg**2))

    k = kurtosis(seg)

    c = np.max(np.abs(seg)) / rms

    s = skew(seg)

    # -------------------
    # FFT
    # -------------------

    f = np.abs(fft(seg))

    # -------------------
    # BAND ENERGIES
    # -------------------

    band1 = np.log(np.sum(f[0:min(200, len(f))]) + 1)

    band2 = np.log(np.sum(f[200:min(800, len(f))]) + 1)

    band3 = np.log(np.sum(f[800:min(1500, len(f))]) + 1)

    # -------------------
    # SPECTRAL CENTROID
    # -------------------

    centroid = np.sum(np.arange(len(f))*f) / np.sum(f)

    # -------------------
    # FFT PEAKS
    # -------------------

    peak1 = np.max(f[0:min(200, len(f))])

    peak2 = np.max(f[200:min(800, len(f))])

    peak3 = np.max(f[800:min(1500, len(f))])

    # -------------------
    # RETURN FEATURES
    # -------------------

    return [

        rms,
        k,
        c,
        s,

        band1,
        band2,
        band3,

        centroid,

        peak1,
        peak2,
        peak3
    ]

# =========================
# FILE UPLOAD
# =========================

uploaded_file = st.file_uploader(
    "📂 Upload CSV File",
    type=["csv"],
    key="uploader"
)

# =========================
# MAIN PROCESS
# =========================

if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)
    st.write("Uploaded File:", uploaded_file.name)

    signal = data.iloc[:,0].values
    st.write("Signal Mean:", np.mean(signal))
    st.write("Signal Std:", np.std(signal))

    # preprocess
    signal = signal - np.mean(signal)

    # segment
    window = 4096
    all_features = []

    for i in range(0, len(signal)-window, window*2):

        seg = signal[i:i+window]

        feat = extract_features(seg)

        all_features.append(feat)

# Average features from all segments
    features = np.mean(all_features, axis=0)
    #seg = signal[:500]

    # extract features
    #features = extract_features(seg)

    features = np.array(features).reshape(1,-1)

    # scale
    features = scaler.transform(features)

    # predict
    pred = model.predict(features)

    labels = {
        0: "✅ Healthy Condition",
        1: "⚠️ Inner Race Fault",
        2: "⚠️ Outer Race Fault",
        3: "⚠️ Ball Fault"
    }

    result = labels[pred[0]]

    # =========================
    # PREDICTION OUTPUT
    # =========================

    st.subheader("🔍 Prediction Result")

    st.success(result)

    # =========================
    # SIGNAL PLOT
    # =========================

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("📈 Time Domain Signal")

        fig, ax = plt.subplots(figsize=(7,4))

        ax.plot(signal[:3000])

        ax.set_xlabel("Samples")
        ax.set_ylabel("Amplitude")

        st.pyplot(fig)

    # =========================
    # FFT PLOT
    # =========================

    with col2:

        st.subheader("📊 FFT Spectrum")

        fft_signal = np.abs(fft(signal[:3000]))

        fig2, ax2 = plt.subplots(figsize=(7,4))

        ax2.plot(fft_signal[:1000])

        ax2.set_xlabel("Frequency Bin")
        ax2.set_ylabel("Magnitude")

        st.pyplot(fig2)

    # =========================
    # FEATURE TABLE
    # =========================

    st.subheader("🧠 Extracted Features")

    feature_names = [

    "RMS",
    "Kurtosis",
    "Crest Factor",
    "Skewness",

    "Band Energy 1",
    "Band Energy 2",
    "Band Energy 3",

    "Spectral Centroid",

    "FFT Peak 1",
    "FFT Peak 2",
    "FFT Peak 3"
]

    feat_df = pd.DataFrame({
        "Feature": feature_names,
        "Value": features.flatten()
    })

    st.dataframe(feat_df)

# =========================
# FOOTER


st.markdown("---")

st.markdown("""
<div style='
    text-align: center;
    padding: 20px;
    border-radius: 15px;
    background: linear-gradient(135deg, #0F2027, #203A43, #2C5364);
    color: white;
    margin-top: 30px;
    box-shadow: 0px 0px 15px rgba(0,255,170,0.3);
'>

<h3 style='color:#00FFAA;'>
⚙️ SmartMotor Diagnostics
</h3>

<p style='font-size:18px; color:#E0E0E0;'>
Developed as a Final Year Project by
</p>

<p style='font-size:17px; line-height:1.8;'>

<span style='color:#00AAFF;'>Anuj Kumar</span> (2215077)<br>

<span style='color:#00AAFF;'>Rudraveer Singh</span> (2215068)<br>

<span style='color:#00AAFF;'>Ankit Kumar Barnwal</span> (2215014)

</p>

<p style='font-size:18px; color:#E0E0E0;'>
Under the Guidance of
</p>

<p style='line-height:1.6;'>

<span style='color:#FFD700; font-size:20px;'>
Dr. Rajdeep Dasgupta
</span><br>

Associate Professor<br>
EIE, NIT Silchar

</p>

</div>
""", unsafe_allow_html=True)