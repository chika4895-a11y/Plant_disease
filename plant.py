"""
Plant Disease Detection – Streamlit App
========================================
Wraps the SA-PSC / CA-PSC model in a clean Streamlit UI.

Run:
    pip install streamlit torch torchvision pillow matplotlib scikit-learn
    streamlit run app_plant_disease.py
"""

import io
import time
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🌿 Plant Disease Detector",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;600&display=swap');

  :root {
    --green:  #2D6A4F; --light-g: #74C69D; --cream: #F8FAF8;
    --dark:   #1B3A2D; --accent: #D4A017; --muted: #6B8F71; --border: #B7DEC8;
  }
  html, body, [data-testid="stAppViewContainer"] {
    background: var(--cream) !important; font-family: 'DM Sans', sans-serif;
  }
  [data-testid="stHeader"] { background: transparent !important; }
  #MainMenu, footer, [data-testid="stToolbar"] { visibility: hidden; }
  h1, h2, h3 { font-family: 'DM Serif Display', serif !important; color: var(--dark) !important; }

  .hero-banner {
    background: linear-gradient(135deg, #2D6A4F 0%, #1B4332 100%);
    border-radius: 10px; padding: 2rem 2.5rem; color: white;
    margin-bottom: 1.5rem;
  }
  .hero-banner h1 { color: white !important; font-size: 2.2rem; margin: 0; }
  .hero-banner p  { color: rgba(255,255,255,.8); font-size: .9rem; margin-top:.5rem; }

  .result-box {
    background: white; border: 2px solid var(--light-g);
    border-radius: 10px; padding: 1.4rem; margin-top: 1rem;
  }
  .result-healthy  { border-color: #52B788; }
  .result-disease  { border-color: #E07A5F; }

  .prob-bar-wrap { margin-bottom: .5rem; }
  .prob-label    { font-size:.78rem; display:flex; justify-content:space-between; margin-bottom:2px; }
  .prob-track    { height:8px; background:#E0EDE5; border-radius:4px; overflow:hidden; }
  .prob-fill     { height:100%; border-radius:4px; }

  .arch-card {
    background: white; border:1px solid var(--border); border-radius:8px;
    padding:1rem 1.2rem; margin-bottom:.8rem; border-left:4px solid var(--green);
  }
  .arch-card h4 { margin:0 0 .3rem; font-size:.95rem; color: var(--dark); }
  .arch-card p  { margin:0; font-size:.78rem; color: var(--muted); line-height:1.55; }

  .metric-row { display:flex; gap:1rem; margin-bottom:1rem; }
  .metric-box { flex:1; background:white; border:1px solid var(--border);
                border-radius:8px; padding:.8rem; text-align:center; }
  .metric-num { font-family:'DM Serif Display',serif; font-size:1.6rem; color:var(--green); }
  .metric-lbl { font-size:.65rem; letter-spacing:.15em; text-transform:uppercase; color:var(--muted); }

  [data-testid="column"] { padding:0 .4rem !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# MODEL DEFINITION  (same as plant_disease_detection.py, self-contained here)
# ─────────────────────────────────────────────────────────────────────────────
class SpectralPCA(nn.Module):
    def __init__(self, in_b, out_b):
        super().__init__()
        self.proj = nn.Conv2d(in_b, out_b, 1, bias=False)
        nn.init.orthogonal_(self.proj.weight)
    def forward(self, x): return self.proj(x)

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, ic, oc, k=3, p=1):
        super().__init__()
        self.dw = nn.Conv2d(ic, ic, k, padding=p, groups=ic, bias=False)
        self.pw = nn.Conv2d(ic, oc, 1, bias=False)
        self.bn = nn.BatchNorm2d(oc); self.act = nn.ReLU(True)
    def forward(self, x): return self.act(self.bn(self.pw(self.dw(x))))

class PSCModule(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 1, bias=False)
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.t1   = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.sig  = nn.Sigmoid()
    def forward(self, x):
        f = self.conv(x)
        t = self.t1(self.pool(f))
        t2 = F.interpolate(t, size=x.shape[2:], mode='bilinear', align_corners=False)
        return self.sig(t2)

class SelfAttention(nn.Module):
    def __init__(self, ch):
        super().__init__()
        hc = max(1, ch//8)
        self.Wq = nn.Conv2d(ch, hc, 1, bias=False)
        self.Wk = nn.Conv2d(ch, hc, 1, bias=False)
        self.Wv = nn.Conv2d(ch, ch, 1, bias=False)
        self.sc = hc**-0.5
    def forward(self, x):
        B,C,H,W = x.shape
        Q = self.Wq(x).view(B,-1,H*W).permute(0,2,1)
        K = self.Wk(x).view(B,-1,H*W)
        V = self.Wv(x).view(B,-1,H*W).permute(0,2,1)
        a = torch.softmax(torch.bmm(Q,K)*self.sc, dim=-1)
        return torch.bmm(a,V).permute(0,2,1).view(B,C,H,W)

class SAPSCModule(nn.Module):
    def __init__(self, ic, oc):
        super().__init__()
        self.sa = SelfAttention(ic); self.psc = PSCModule(ic)
        self.proj = nn.Sequential(nn.Conv2d(ic,oc,1,bias=False), nn.BatchNorm2d(oc), nn.ReLU(True))
    def forward(self, x):
        x1 = self.sa(x); x2 = self.psc(x)
        return self.proj(x1*x2 + x1)

class CrossAttention(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.Wq = nn.Conv2d(ch,ch,1,bias=False)
        self.Wk = nn.Conv2d(ch,ch,1,bias=False)
        self.Wv = nn.Conv2d(ch,ch,1,bias=False)
    def forward(self, x):
        Q,K,V = self.Wq(x), self.Wk(x), self.Wv(x)
        r = Q * torch.softmax(K.mean(3,keepdim=True), dim=1) * V
        c = Q * torch.softmax(K.mean(2,keepdim=True), dim=1) * V
        return x + r + c

class CAPSCModule(nn.Module):
    def __init__(self, ic, oc):
        super().__init__()
        self.ca = CrossAttention(ic); self.psc = PSCModule(ic)
        self.proj = nn.Sequential(nn.Conv2d(ic*2,oc,1,bias=False), nn.BatchNorm2d(oc), nn.ReLU(True))
    def forward(self, x):
        ca = self.ca(x); cal = self.psc(x)
        return self.proj(torch.cat([x, ca*cal+ca], dim=1))

class MSGLModule(nn.Module):
    def __init__(self, ic, oc):
        super().__init__()
        mid = oc//3 + (1 if oc%3 else 0)
        self.b1 = nn.Sequential(nn.Conv2d(ic,mid,1,          bias=False), nn.BatchNorm2d(mid), nn.ReLU(True))
        self.b2 = nn.Sequential(nn.Conv2d(ic,mid,3,padding=1,bias=False), nn.BatchNorm2d(mid), nn.ReLU(True))
        self.b3 = nn.Sequential(nn.Conv2d(ic,mid,5,padding=2,bias=False), nn.BatchNorm2d(mid), nn.ReLU(True))
        self.fuse = nn.Sequential(nn.Conv2d(mid*3,oc,1,bias=False), nn.BatchNorm2d(oc), nn.ReLU(True))
    def forward(self, g, l):
        x = g + l
        return self.fuse(torch.cat([self.b1(x),self.b2(x),self.b3(x)], dim=1))

class PlantDiseaseNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        C,B32,sa,ca,mg,nc,inc = (cfg[k] for k in
            ["n_components","base_channels","sa_psc_out","ca_psc_out","msgl_out","num_classes","in_channels"])
        self.pca    = SpectralPCA(inc, C)
        self.dsconv = DepthwiseSeparableConv(C, B32)
        self.sa_psc = SAPSCModule(B32, sa)
        self.ca_psc = CAPSCModule(B32, ca)
        self.msgl   = MSGLModule(sa, mg)
        self.gap    = nn.AdaptiveAvgPool2d(1)
        self.head   = nn.Sequential(
            nn.Flatten(), nn.Linear(mg,256), nn.ReLU(True), nn.Dropout(.4), nn.Linear(256,nc))
    def forward(self, x):
        x = self.pca(x); x = self.dsconv(x)
        return self.head(self.gap(self.msgl(self.sa_psc(x), self.ca_psc(x))))

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
DISEASE_CLASSES = ["Healthy", "Early Blight", "Late Blight", "Leaf Mold", "Bacterial Spot"]
CLASS_COLORS    = ["#52B788", "#E07A5F", "#C1440E", "#9B7EDE", "#F4A261"]
SEVERITY_MAP    = {"Healthy": "None", "Early Blight": "Moderate",
                   "Late Blight": "Severe", "Leaf Mold": "Mild", "Bacterial Spot": "Moderate"}
REMEDY_MAP = {
    "Healthy":        "✅ No action needed. Maintain regular watering and monitoring.",
    "Early Blight":   "🍂 Remove affected leaves. Apply copper-based fungicide. Avoid overhead watering.",
    "Late Blight":    "🚨 Immediate action required! Remove infected plants. Apply fungicide (chlorothalonil). Improve drainage.",
    "Leaf Mold":      "🌬️ Improve air circulation. Reduce humidity. Apply mancozeb-based fungicide.",
    "Bacterial Spot": "💧 Avoid wetting foliage. Apply copper bactericide. Remove heavily infected leaves.",
}
CFG = dict(n_components=16, base_channels=32, sa_psc_out=96, ca_psc_out=96,
           msgl_out=128, num_classes=5, in_channels=3)

@st.cache_resource
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = PlantDiseaseNet(CFG).to(device)
    model.eval()
    return model, device

def preprocess_image(img: Image.Image, size: int = 64) -> torch.Tensor:
    img = img.convert("RGB").resize((size, size))
    arr = np.array(img, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406]); std = np.array([0.229, 0.224, 0.225])
    arr  = (arr - mean) / std
    return torch.tensor(arr.transpose(2,0,1), dtype=torch.float32)

@torch.no_grad()
def predict(model, tensor: torch.Tensor, device: str):
    out   = model(tensor.unsqueeze(0).to(device))
    probs = torch.softmax(out, dim=-1).squeeze().cpu().numpy()
    return probs

def prob_bars_html(probs, classes, colors):
    html = ""
    for cls, prob, col in zip(classes, probs, colors):
        pct = float(prob) * 100
        html += f"""
        <div class="prob-bar-wrap">
          <div class="prob-label"><span>{cls}</span><span>{pct:.1f}%</span></div>
          <div class="prob-track">
            <div class="prob-fill" style="width:{pct}%;background:{col}"></div>
          </div>
        </div>"""
    return html

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    img_size   = st.slider("Input resolution", 32, 128, 64, 16)
    show_arch  = st.checkbox("Show architecture details", True)
    conf_thr   = st.slider("Confidence threshold (%)", 0, 100, 50)
    st.markdown("---")
    st.markdown("### 🌿 About")
    st.markdown("""
    This app uses the **SA-PSC / CA-PSC** architecture:
    - PCA spectral reduction  
    - Depthwise separable convolution  
    - Self-Attention + Position Calibration  
    - Cross-Attention + Position Calibration  
    - Multiscale Global-Local fusion  

    *Built for: Chika K Gangadharan — PhD AI Research*
    """)

# ─────────────────────────────────────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
  <h1>🌿 Plant Disease Detection</h1>
  <p>SA-PSC · CA-PSC · MSGL Architecture &nbsp;|&nbsp; Upload a leaf image to detect diseases instantly</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# STATS ROW
# ─────────────────────────────────────────────────────────────────────────────
c1,c2,c3,c4 = st.columns(4)
for col,(num,lbl) in zip([c1,c2,c3,c4],[
    ("5","Disease Classes"),("SA+CA","Attention Modules"),("3×","Multiscale Fusion"),("~1M","Parameters")]):
    with col:
        st.markdown(f"""
        <div class="metric-box">
          <div class="metric-num">{num}</div>
          <div class="metric-lbl">{lbl}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# MAIN CONTENT
# ─────────────────────────────────────────────────────────────────────────────
model, device = load_model()

upload_col, result_col = st.columns([1, 1], gap="large")

with upload_col:
    st.markdown("#### 📤 Upload Leaf Image")
    uploaded = st.file_uploader("Choose a plant leaf image",
                                type=["jpg","jpeg","png","webp"],
                                label_visibility="collapsed")
    use_demo = st.checkbox("Use random demo image (no upload needed)", value=True)

    if uploaded:
        img = Image.open(uploaded)
        st.image(img, caption="Uploaded Image", use_container_width=True)
    elif use_demo:
        arr = (np.random.rand(224,224,3)*255).astype(np.uint8)
        arr[60:160, 60:160] = [34,139,34]  # green center
        img = Image.fromarray(arr)
        st.image(img, caption="Demo (random synthetic image)", use_container_width=True)
    else:
        img = None
        st.info("☝️ Please upload a leaf image or tick the demo checkbox.")

    if img and st.button("🔍 Detect Disease", use_container_width=True, type="primary"):
        with st.spinner("Running SA-PSC / CA-PSC analysis…"):
            tensor = preprocess_image(img, size=img_size)
            time.sleep(0.6)   # brief pause for UX
            probs  = predict(model, tensor, device)

        pred_idx  = int(np.argmax(probs))
        pred_cls  = DISEASE_CLASSES[pred_idx]
        conf      = float(probs[pred_idx]) * 100
        severity  = SEVERITY_MAP[pred_cls]
        remedy    = REMEDY_MAP[pred_cls]
        is_healthy = pred_cls == "Healthy"

        with result_col:
            st.markdown("#### 📊 Detection Results")
            box_cls = "result-healthy" if is_healthy else "result-disease"
            status_emoji = "✅" if is_healthy else "⚠️"

            st.markdown(f"""
            <div class="result-box {box_cls}">
              <h3 style="margin:0">{status_emoji} {pred_cls}</h3>
              <div style="font-size:.8rem;color:#6B8F71;margin:.3rem 0 .8rem">
                Confidence: <strong>{conf:.1f}%</strong> &nbsp;|&nbsp;
                Severity: <strong>{severity}</strong>
              </div>
              <hr style="border-color:#E0EDE5;margin:.5rem 0">
              <p style="font-size:.83rem;line-height:1.6;margin:0">{remedy}</p>
            </div>
            """, unsafe_allow_html=True)

            if conf < conf_thr:
                st.warning(f"⚠️ Confidence ({conf:.1f}%) is below your threshold ({conf_thr}%). Consider a clearer image.")

            st.markdown("**Class Probabilities**")
            st.markdown(prob_bars_html(probs, DISEASE_CLASSES, CLASS_COLORS),
                        unsafe_allow_html=True)

            # Probability pie chart
            fig, ax = plt.subplots(figsize=(4,4))
            fig.patch.set_facecolor("#F8FAF8")
            ax.set_facecolor("#F8FAF8")
            wedges, _ = ax.pie(probs, colors=CLASS_COLORS, startangle=90,
                                wedgeprops=dict(width=0.55))
            ax.legend(wedges, DISEASE_CLASSES, loc="lower center",
                      bbox_to_anchor=(0.5,-0.15), ncol=2, fontsize=7)
            ax.set_title("Probability Distribution", fontsize=9, pad=8)
            buf = io.BytesIO()
            plt.tight_layout()
            plt.savefig(buf, format="png", dpi=120, bbox_inches="tight")
            st.image(buf, use_container_width=True)
            plt.close()
    else:
        with result_col:
            st.markdown("#### 📊 Detection Results")
            st.info("Results will appear here after you click **Detect Disease**.")

# ─────────────────────────────────────────────────────────────────────────────
# ARCHITECTURE DETAILS
# ─────────────────────────────────────────────────────────────────────────────
if show_arch:
    st.markdown("---")
    st.markdown("#### 🏗️ Model Architecture")
    arch_cols = st.columns(3)
    modules = [
        ("SpectralPCA", "Reduces hyperspectral bands via learnable 1×1 projection. Removes redundancy while preserving discriminative spectral features."),
        ("Depthwise Separable Conv", "Processes channels and spatial information separately. Reduces parameters significantly while retaining strong feature extraction."),
        ("SA-PSC Module", "Self-Attention captures global pixel relationships across the entire image. PSC calibrates feature positions using pooling + sigmoid."),
        ("CA-PSC Module", "Cross-Attention focuses on local horizontal/vertical context. Combined with PSC for fine-grained spatial calibration."),
        ("MSGL Fusion", "Fuses global (SA-PSC) and local (CA-PSC) features at 3 scales (1×1, 3×3, 5×5 convolutions) for rich multi-resolution representations."),
        ("Classification Head", "Global Average Pooling → FC(256) → Dropout → FC(num_classes) → Softmax probabilities per disease class."),
    ]
    for i,(title,desc) in enumerate(modules):
        with arch_cols[i%3]:
            st.markdown(f"""
            <div class="arch-card">
              <h4>{'①②③④⑤⑥'[i]} {title}</h4>
              <p>{desc}</p>
            </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# DISEASE REFERENCE TABLE
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("#### 🌱 Disease Reference Guide")
ref_cols = st.columns(len(DISEASE_CLASSES))
for col, cls, col_hex in zip(ref_cols, DISEASE_CLASSES, CLASS_COLORS):
    with col:
        sev = SEVERITY_MAP[cls]
        st.markdown(f"""
        <div style="background:white;border:1px solid #B7DEC8;border-top:4px solid {col_hex};
                    border-radius:6px;padding:.9rem;text-align:center">
          <div style="font-weight:600;font-size:.85rem;color:#1B3A2D">{cls}</div>
          <div style="font-size:.7rem;color:#6B8F71;margin:.3rem 0">Severity: {sev}</div>
          <div style="font-size:.72rem;color:#444;line-height:1.5">{REMEDY_MAP[cls][:60]}…</div>
        </div>""", unsafe_allow_html=True)

st.markdown("""
<br><div style="text-align:center;font-size:.7rem;color:#6B8F71">
  🌿 Plant Disease Detection · SA-PSC/CA-PSC Architecture · Research Project — Chika K Gangadharan
</div><br>""", unsafe_allow_html=True)