# ============================================================
# ZENETICA — Day 1: Aesthetic DNA Engine
# What this file does:
#   1. Downloads a real fashion dataset
#   2. Loads CLIP — a powerful vision AI model
#   3. Converts every fashion image into a "style vector"
#   4. Clusters those vectors to find the brand's style DNA
#   5. Visualizes it as an interactive 2D map
# ============================================================

import os
import numpy as np
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
import plotly.graph_objects as go
from datasets import load_dataset
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")


# --- Small dotenv loader (no external dependency) -------------------------
def _load_dotenv(path='.env'):
    if not os.path.isfile(path):
        return
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if '=' not in line:
                continue
            k, v = line.split('=', 1)
            k = k.strip()
            v = v.strip().strip('"').strip("'")
            if k and v:
                os.environ.setdefault(k, v)


# Load local .env if present (copy .env.example -> .env and edit)
_load_dotenv('.env')


# ── CONFIGURATION ───────────────────────────────────────────
# These are settings you can change easily in one place

NUM_IMAGES   = int(os.getenv("NUM_IMAGES", 200))    # How many catalog images to process
                      # Start with 200 — fast and still meaningful
                      # Increase to 500+ once it's working

NUM_CLUSTERS = int(os.getenv("NUM_CLUSTERS", 8))      # How many style "territories" to find in the brand
                      # Think of each cluster as one aesthetic mood:
                      # Think of each cluster as one aesthetic mood:
                      # e.g. "minimalist basics", "bold statement pieces"

BRAND_NAME   = "Zara" # Just a label — we'll simulate two brands

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# CUDA = your GPU (much faster). If you don't have one, it uses CPU.
# Don't worry — CPU works fine for 200 images.
print(f"Running on: {DEVICE}")


# ── STEP 1: LOAD THE FASHION DATASET ───────────────────────
# We use the 'Marqo/fashion-200k' dataset from Hugging Face.
# It's a real fashion catalog with product images and descriptions.
# load_dataset() downloads it automatically — no manual download needed.

print("\n[1/5] Loading fashion dataset...")

# Configuration: prefer a local dataset directory if provided, otherwise load from HF Hub
# Dataset selection: allow environment override. Try known Marqo variants, then fall back to CIFAR-10 for smoke tests.
DATASET_NAME = os.getenv("DATASET_NAME", "")
_PREFERRED_DATASETS = [
    DATASET_NAME,
    "Marqo/fashion200k",
    "Marqo/fashion-200k",
    "cifar10",
]
_PREFERRED_DATASETS = [d for d in _PREFERRED_DATASETS if d]
LOCAL_DATA_DIR = os.getenv("LOCAL_DATA_DIR", "")
# Token - support multiple env var names for convenience
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HF_HUB_TOKEN")

# If a token was provided, also set the newer env var names expected by the HF libraries
if HF_TOKEN:
    os.environ.setdefault("HF_HUB_TOKEN", HF_TOKEN)
    os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", HF_TOKEN)

if LOCAL_DATA_DIR:
    # Load from a local imagefolder (images organized in subfolders per class)
    print(f"    Loading local dataset from: {LOCAL_DATA_DIR}")
    dataset = load_dataset("imagefolder", data_dir=LOCAL_DATA_DIR, split="train")
else:
    last_exc = None
    for candidate in _PREFERRED_DATASETS:
        try:
            print(f"    Trying dataset from Hugging Face Hub: {candidate}")

             # ── Why this logic exists ──────────────────────────────
            # Different datasets use different split names.
            # Marqo/fashion200k uses split="data" (not "train")
            # Most other HF datasets use split="train"
            # We try "data" first for Marqo, then fall back to "train"
            # for everything else (like cifar10)

            if "fashion200k" in candidate.lower() or "fashion-200k" in candidate.lower():
            # Marqo's fashion dataset specifically uses "data" as split name
                split_to_use = "data"
            else:
                split_to_use = "train"

            dataset = load_dataset(candidate, split=split_to_use)
            print(f"    Successfully loaded dataset: {candidate}")
            break

        except Exception as e:
            print(f"    Failed to load {candidate}: {e}")
            last_exc = e
    else:
        print("\nERROR: Failed to load any of the preferred datasets from the Hub.")
        print("Tried:")
        for d in _PREFERRED_DATASETS:
            print("  -", d)
        print("If the dataset is private or rate-limited, set HF_TOKEN in a local .env file or export it in your shell.")
        print("Alternatively, set LOCAL_DATA_DIR to point to a local image folder and retry.")
        if last_exc:
            raise last_exc

# We only take the first NUM_IMAGES items to keep it fast
dataset = dataset.select(range(NUM_IMAGES))

print(f"    Loaded {len(dataset)} fashion items")
print(f"    Sample item keys: {list(dataset[0].keys())}")


# ── STEP 2: LOAD CLIP ──────────────────────────────────────
# CLIP (Contrastive Language-Image Pretraining) is a model
# created by OpenAI. It was trained on 400 MILLION image-text
# pairs from the internet.
#
# What makes CLIP special:
#   - It understands BOTH images and text in the same "space"
#   - Similar images get similar vectors (numbers)
#   - "navy blazer" text is close to actual navy blazer images
#   - This makes it perfect for fashion understanding
#
# The model has two parts:
#   - Vision encoder: converts images → 512 numbers
#   - Text encoder: converts text → 512 numbers
#
# We're using the vision encoder today.

print("\n[2/5] Loading CLIP model (downloads ~600MB on first run)...")

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Move model to GPU if available, otherwise CPU
model = model.to(DEVICE)
model.eval()  # Put model in evaluation mode (turns off dropout etc.)

print("    CLIP loaded successfully")
print("    Architecture: Vision Transformer (ViT-B/32)")
print("    Embedding size: 512 dimensions per image")


# ── STEP 3: ENCODE EVERY IMAGE INTO A VECTOR ──────────────
# This is the core ML step.
#
# For each fashion image, CLIP's vision encoder produces
# a vector of 512 numbers. This vector captures:
#   - Colors (warm/cool tones, specific hues)
#   - Textures (smooth, rough, shiny, matte)
#   - Silhouette (oversized, fitted, structured)
#   - Style mood (minimal, maximalist, edgy, classic)
#
# Two visually similar items will have vectors that are
# mathematically "close" to each other.
# Two totally different items will have vectors that are "far."
#
# This is called an "embedding" — turning raw pixels into
# a meaningful mathematical representation.

print("\n[3/5] Encoding fashion images with CLIP...")

embeddings = []   # This list will hold all our 512-dim vectors
valid_indices = [] # Track which images loaded successfully
labels_list = []   # Store image descriptions for tooltips later

for i, item in enumerate(tqdm(dataset, desc="    Processing images")):
    try:
        # Get the image from the dataset (robust to different dataset field names)
        # Many HF image datasets use the key 'image' or 'img'
        image = None
        for k in ("image", "img", "images", "image0"):
            if k in item:
                image = item[k]
                break
        if image is None:
            # nothing to do for this item
            continue

        # Make sure it's RGB (some images are RGBA or grayscale)
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Resize to what CLIP expects: 224x224 pixels
        image = image.resize((224, 224))

        # CLIPProcessor prepares the image for the model:
        #   - Normalizes pixel values to [-1, 1]
        #   - Converts to a PyTorch tensor
        #   - Adds a batch dimension
        inputs = processor(
            images=image,
            return_tensors="pt",  # Return PyTorch tensors
            padding=True
        )
        # Move tensors to the chosen DEVICE
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        # torch.no_grad() tells PyTorch NOT to track gradients
        # We don't need gradients because we're not training,
        # just doing a forward pass (inference)
        with torch.no_grad():
            # Get the image embedding from CLIP
            image_features = model.get_image_features(
                pixel_values=inputs["pixel_values"]
            )

        # The return type may be a tensor or a model output object depending on
        # transformers version. Handle both cases.
        if hasattr(image_features, 'cpu'):
            feats = image_features
        elif hasattr(image_features, 'pooler_output'):
            feats = image_features.pooler_output
        elif hasattr(image_features, 'last_hidden_state'):
            # fallback: mean-pool last_hidden_state
            feats = image_features.last_hidden_state.mean(dim=1)
        else:
            raise RuntimeError("Could not extract tensor from CLIP output")

        # Move from GPU to CPU, convert to numpy array
        # .squeeze() removes the batch dimension (1, 512) → (512,)
        embedding = feats.cpu().numpy().squeeze()

        embeddings.append(embedding)
        valid_indices.append(i)

        # Store the description for tooltip display
        desc = item.get("title", item.get("description", f"Item {i}"))
        if isinstance(desc, str):
            labels_list.append(desc[:60])  # Truncate long descriptions
        else:
            labels_list.append(f"Item {i}")

    except Exception as e:
        # Log and skip images that fail to load (avoid silent failures)
        print(f"Skipping image {i}: {e}")
        continue

# Stack all embeddings into a 2D numpy array
# Shape: (NUM_IMAGES, 512) — rows=images, columns=embedding dims
embeddings = np.array(embeddings)
print(f"    Encoded {len(embeddings)} images")
print(f"    Embedding matrix shape: {embeddings.shape}")


# ── STEP 4: NORMALIZE EMBEDDINGS ──────────────────────────
# Normalization scales every embedding vector to length 1.
# This is important because:
#   - Without normalization, longer vectors dominate similarity
#   - With L2 normalization, cosine similarity = dot product
#   - K-Means works better on normalized vectors
# Think of it as making all arrows the same length,
# so only their DIRECTION matters (which = the style)

embeddings_norm = normalize(embeddings, norm="l2")
print("    Embeddings normalized (L2)")


# ── STEP 5: K-MEANS CLUSTERING ────────────────────────────
# K-Means is an unsupervised ML algorithm.
# "Unsupervised" means we give it NO labels — it discovers
# structure on its own.
#
# How K-Means works:
#   1. Place K random "centroids" (center points) in the space
#   2. Assign each item to its nearest centroid
#   3. Move centroids to the average position of their cluster
#   4. Repeat steps 2-3 until nothing changes (convergence)
#
# In our case:
#   - Each cluster = a "style territory" the brand owns
#   - Items in the same cluster look stylistically similar
#   - The centroid = the "average" aesthetic of that cluster
#
# n_init=10 means it tries 10 random starts and picks the best
# random_state=42 makes results reproducible (same result every run)

print(f"\n[4/5] Running K-Means clustering (K={NUM_CLUSTERS})...")

kmeans = KMeans(
    n_clusters=NUM_CLUSTERS,
    n_init=10,
    random_state=42,
    max_iter=300
)

# fit_predict does two things: learns the clusters AND
# assigns each item to a cluster in one step
cluster_labels = kmeans.fit_predict(embeddings_norm)

# Count items in each cluster
unique, counts = np.unique(cluster_labels, return_counts=True)
print("    Cluster distribution:")
for cluster_id, count in zip(unique, counts):
    print(f"      Cluster {cluster_id}: {count} items")


# ── STEP 6: PCA — REDUCE TO 2D FOR VISUALIZATION ─────────
# Our embeddings live in 512-dimensional space.
# Humans can only see 2D or 3D.
# PCA (Principal Component Analysis) finds the 2 directions
# in 512D space that capture the MOST variation in the data,
# and projects everything onto those 2 axes.
#
# Think of it like taking a 3D sculpture and finding the
# angle that gives the most informative 2D shadow.
#
# We ONLY use PCA for visualization — all the actual
# ML (clustering, similarity) happens in the full 512D space.

print("\n[5/5] Reducing to 2D with PCA for visualization...")

pca = PCA(n_components=2, random_state=42)
embeddings_2d = pca.fit_transform(embeddings_norm)

# How much of the variation is explained by our 2 axes?
variance_explained = pca.explained_variance_ratio_.sum() * 100
print(f"    Variance explained by 2 components: {variance_explained:.1f}%")
print(f"    (Higher = the 2D map captures more of the real structure)")


# ── STEP 7: BUILD INTERACTIVE PLOTLY VISUALIZATION ────────
# This creates the Brand DNA Map — the centrepiece of Day 1.
# Plotly makes it interactive: zoom, pan, hover for details.

print("\nBuilding Brand DNA Map...")

# 8 distinct colors for 8 clusters — one per style territory
CLUSTER_COLORS = [
    "#7F77DD", "#1D9E75", "#D85A30", "#378ADD",
    "#639922", "#BA7517", "#D4537E", "#888780"
]

# Style territory names — in a real product, you'd use CLIP
# text similarity to auto-label these. For now we name them.
TERRITORY_NAMES = [
    "Minimalist Basics", "Bold Statement", "Smart Casual",
    "Evening Wear", "Streetwear", "Resort / Summer",
    "Power Dressing", "Transitional Layers"
]

fig = go.Figure()

# Plot each cluster as a separate scatter trace
# This lets us toggle clusters on/off in the legend
for cluster_id in range(NUM_CLUSTERS):
    # Boolean mask — True for items in this cluster
    mask = cluster_labels == cluster_id

    # Get 2D coordinates for this cluster's items
    x_vals = embeddings_2d[mask, 0]
    y_vals = embeddings_2d[mask, 1]

    # Get hover text for this cluster's items
    hover_texts = [labels_list[i] for i, m in enumerate(mask) if m]

    fig.add_trace(go.Scatter(
        x=x_vals,
        y=y_vals,
        mode="markers",
        name=f"Cluster {cluster_id}: {TERRITORY_NAMES[cluster_id]}",
        marker=dict(
            size=8,
            color=CLUSTER_COLORS[cluster_id],
            opacity=0.8,
            line=dict(width=0.5, color="white")
        ),
        text=hover_texts,
        hovertemplate=(
            f"<b>{TERRITORY_NAMES[cluster_id]}</b><br>"
            "%{text}<br>"
            "<extra></extra>"  # Removes the trace name from hover box
        )
    ))

# Add cluster centroid markers
# PCA-transform the centroids from 512D → 2D
centroids_2d = pca.transform(
    normalize(kmeans.cluster_centers_, norm="l2")
)

fig.add_trace(go.Scatter(
    x=centroids_2d[:, 0],
    y=centroids_2d[:, 1],
    mode="markers+text",
    name="Style Territory Centers",
    marker=dict(
        size=16,
        color="white",
        line=dict(width=2, color="#333333"),
        symbol="diamond"
    ),
    text=[f"C{i}" for i in range(NUM_CLUSTERS)],
    textposition="top center",
    textfont=dict(size=10, color="#333333"),
    hovertemplate="<b>Centroid %{text}</b><br><extra></extra>"
))

# Layout styling
fig.update_layout(
    title=dict(
        text=f"ZENETICA — {BRAND_NAME} Aesthetic DNA Map",
        font=dict(size=18),
        x=0.5
    ),
    xaxis=dict(
        title="Style Axis 1 (PCA Component 1)",
        showgrid=True,
        gridcolor="rgba(0,0,0,0.06)",
        zeroline=False
    ),
    yaxis=dict(
        title="Style Axis 2 (PCA Component 2)",
        showgrid=True,
        gridcolor="rgba(0,0,0,0.06)",
        zeroline=False
    ),
    legend=dict(
        title="Style Territories",
        font=dict(size=11),
        bgcolor="rgba(255,255,255,0.8)"
    ),
    plot_bgcolor="white",
    paper_bgcolor="white",
    width=900,
    height=620,
    hovermode="closest",
    font=dict(family="Arial, sans-serif")
)

# Save as interactive HTML file (opens in browser)
output_path = "outputs/zenetica_dna_map.html"
fig.write_html(output_path)
print(f"    DNA Map saved → {output_path}")

# Also save as static image for README
fig.write_image("outputs/zenetica_dna_map.png")
print(f"    PNG saved → outputs/zenetica_dna_map.png")


# ── STEP 8: SAVE EMBEDDINGS FOR DAY 2 ─────────────────────
# We save the embeddings and cluster info so Day 2 can
# load them instantly without re-running CLIP
# (CLIP encoding takes time — no need to repeat it)

np.save("models/embeddings.npy", embeddings_norm)
np.save("models/cluster_labels.npy", cluster_labels)
np.save("models/embeddings_2d.npy", embeddings_2d)
np.save("models/centroids.npy", kmeans.cluster_centers_)

# Save item labels for Day 2
import json
with open("models/item_labels.json", "w") as f:
    json.dump(labels_list, f)

print("\n    Embeddings saved to models/ folder for Day 2")
print("\n" + "="*55)
print("  DAY 1 COMPLETE")
print(f"  Brand DNA map: {output_path}")
print(f"  Style territories found: {NUM_CLUSTERS}")
print(f"  Items encoded: {len(embeddings)}")
print(f"  Open outputs/zenetica_dna_map.html in your browser")
print("="*55)