# 🔬 GeoVision Methodology & Research Formulation

This document details the mathematical formulation, loss functions, metrics, and algorithmic design implemented in the **GeoVision** platform.

---

## 1. Geospatial Processing & Sliding-Window Reconstruction

High-resolution satellite rasters exceed standard GPU memory limits. GeoVision employs a patch-based sliding window tiler and seamless Hann-window blended reconstructor.

### Patch Extraction & Tiling
Given an image $I \in \mathbb{R}^{H \times W \times C}$, patch size $P$, and overlap stride $S$:
$$N_h = \left\lceil \frac{H - P}{S} \right\rceil + 1, \quad N_w = \left\lceil \frac{W - P}{S} \right\rceil + 1$$

### Hann Window Blending
To prevent seamline artifacts along patch edges, predictions are weighted using a 2D separable Hann window $W(x, y) = w_x(x) \cdot w_y(y)$:
$$w(n) = 0.5 - 0.5 \cos\left(\frac{2\pi n}{P - 1}\right), \quad 0 \le n \le P - 1$$
The final prediction map $\hat{Y}$ is the normalized weighted sum:
$$\hat{Y}(x, y) = \frac{\sum_{i} W_i(x, y) \hat{Y}_i(x, y)}{\sum_i W_i(x, y)}$$

---

## 2. Semantic Land-Cover Segmentation

### Compound Loss Formulation (`DiceCELoss`)
To address severe class imbalance between background and urban structures, GeoVision combines Cross-Entropy Loss with Multi-Class Soft Dice Loss:

$$\mathcal{L}_{\text{DiceCE}} = \alpha \mathcal{L}_{\text{CE}} + \beta \mathcal{L}_{\text{Dice}}$$

Where Soft Dice Loss across $C$ classes is defined as:
$$\mathcal{L}_{\text{Dice}} = 1 - \frac{1}{C} \sum_{c=1}^C \frac{2 \sum_{i} p_{ic} g_{ic} + \epsilon}{\sum_{i} p_{ic}^2 + \sum_{i} g_{ic}^2 + \epsilon}$$

### Evaluation Metrics
- **Mean Intersection over Union (mIoU)**:
  $$\text{mIoU} = \frac{1}{C} \sum_{c=1}^C \frac{\text{TP}_c}{\text{TP}_c + \text{FP}_c + \text{FN}_c}$$
- **Overall Accuracy (OA)**:
  $$\text{OA} = \frac{\sum_c \text{TP}_c}{\sum_c (\text{TP}_c + \text{FP}_c + \text{FN}_c + \text{TN}_c)}$$

---

## 3. Bi-Temporal Change Detection

### Siamese UNet Architecture
Given pre-event image $I_{T_1}$ and post-event image $I_{T_2}$, feature representations $F_1, F_2 = E(I_{T_1}), E(I_{T_2})$ are extracted via shared Siamese encoder weights. The difference representation is concatenated or subtracted:
$$F_{\text{diff}} = |F_1 - F_2| \oplus F_1 \oplus F_2$$
and decoded through skip-connected convolutional layers.

### Combined Change Loss (`ChangeLoss`)
$$\mathcal{L}_{\text{Change}} = \lambda_{\text{BCE}} \mathcal{L}_{\text{BCE}}(p, y) + \lambda_{\text{Dice}} \mathcal{L}_{\text{BinaryDice}}(p, y)$$

---

## 4. Metric Learning & Semantic Vector Retrieval

### OpenCLIP Vision-Text Projection
Satellite image patches $x$ and text queries $t$ are projected into a joint 512-dimensional metric space:
$$v_{\text{img}} = \frac{f_{\theta}(x)}{\|f_{\theta}(x)\|_2}, \quad v_{\text{text}} = \frac{g_{\phi}(t)}{\|g_{\phi}(t)\|_2}$$

Cosine similarity between normalized embeddings:
$$\text{sim}(v_{\text{img}}, v_{\text{text}}) = v_{\text{img}}^\top v_{\text{text}} \in [-1, 1]$$

### Ranking Metrics
- **Recall@K**:
  $$\text{Recall@}K = \frac{1}{|Q|} \sum_{q \in Q} \mathbb{I}(\text{rank}(q) \le K)$$
- **Mean Reciprocal Rank (MRR)**:
  $$\text{MRR} = \frac{1}{|Q|} \sum_{q \in Q} \frac{1}{\text{rank}(q)}$$

---

## 5. Grounded VLM & Citation Verification Engine

To prevent hallucination in conversational Earth observation queries, GeoVision implements a two-phase grounding pipeline:

1. **Evidence Synthesis**:
   Structured visual facts are tagged with unique evidence identifiers:
   - `[DET-i]`: Object detection bounding boxes and confidence scores.
   - `[SEG-i]`: Land-cover class percentages and square meter coverage.
   - `[CHG-i]`: Change ratio and altered area metrics.
   - `[GEO-i]`: Spatial bounds, pixel scale, and CRS information.
   - `[RET-i]`: Vector search similarity rank and score.

2. **Verification & Precision Assessment**:
   `CitationVerifier` extracts all cited evidence tokens and verifies their existence against the known structured context.
   $$\text{Citation Precision} = \frac{|\text{Citations} \cap \text{Known Evidence}|}{|\text{Citations}|}$$
   If any citation is fabricated or invalid, the answer is flagged with a strict hallucination warning.
