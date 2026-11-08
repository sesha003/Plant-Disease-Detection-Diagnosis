# Image-Based Plant Disease Detection — How It Works

A farmer takes a photo of a leaf, and the model says which plant it is and whether it has a disease (for example "Tomato — Early blight"). This document explains the whole project: the data, the pipeline, and — in detail — how a convolutional neural network (CNN) turns pixels into a diagnosis, including how filters work and how the kernel calculations are done by hand.

---

## Contents

1. [The project in one page](#1-the-project-in-one-page)
2. [The dataset](#2-the-dataset)
3. [The pipeline, step by step](#3-the-pipeline-step-by-step)
4. [What an image looks like to a computer](#4-what-an-image-looks-like-to-a-computer)
5. [Convolution: filters and kernels](#5-convolution-filters-and-kernels)
6. [Kernel calculations worked by hand](#6-kernel-calculations-worked-by-hand)
7. [Colour images: kernels with depth](#7-colour-images-kernels-with-depth)
8. [The other layers: ReLU, BatchNorm, pooling](#8-the-other-layers-relu-batchnorm-pooling)
9. [Our SimpleCNN, layer by layer](#9-our-simplecnn-layer-by-layer)
10. [From features to a diagnosis: softmax and loss](#10-from-features-to-a-diagnosis-softmax-and-loss)
11. [How the network learns its kernels](#11-how-the-network-learns-its-kernels)
12. [Transfer learning: ResNet, EfficientNet](#12-transfer-learning-resnet-efficientnet)
13. [Evaluation metrics](#13-evaluation-metrics)
14. [Looking inside: filters, feature maps, Grad-CAM](#14-looking-inside-filters-feature-maps-grad-cam)
15. [Limitations worth discussing](#15-limitations-worth-discussing)
16. [Cheat sheet of formulas](#16-cheat-sheet-of-formulas)

---

## 1. The project in one page

| | |
|---|---|
| **Task** | Image classification — one leaf photo in, one of 38 classes out |
| **Classes** | crop + condition, e.g. `Tomato___Early_blight`, `Apple___healthy` (14 crops, 26 diseases, plus healthy classes = 38) |
| **Input** | RGB image, resized to 224 × 224 pixels |
| **Output** | 38 probabilities that sum to 1; the highest one is the prediction |
| **Why it matters** | Early detection lets farmers treat a field before a disease spreads, and a phone app can do this without a plant pathologist on site |

**Three architectures are compared**, matching the proposal:

1. **SimpleCNN** — a small CNN we design and train from scratch (~1.18 M parameters). Shows we understand the building blocks.
2. **ResNet-18 / EfficientNet-B0** — networks pretrained on ImageNet (1.2 M everyday photos) and adapted to leaves. This is *transfer learning*.

---

## 2. The dataset

**Source:** [huggingface.co/datasets/mohanty/PlantVillage](https://huggingface.co/datasets/mohanty/PlantVillage) — from Mohanty, Hughes & Salathé, *"Using Deep Learning for Image-Based Plant Disease Detection"*, Frontiers in Plant Science, 2016. Licence CC BY-SA 3.0.

| Fact | Value |
|---|---|
| Images | 54,306 leaf photos |
| Crops / diseases | 14 crops, 26 diseases → 38 classes |
| Official split | 80 / 20, grouped by physical leaf → 43,596 train / 10,709 test (colour) |
| Variants | `color` (original), `grayscale`, `segmented` (background removed) |
| Native size | 256 × 256 px, one leaf on a plain background |

### 2.1 The "same leaf" problem (why the model could cheat)

Each physical leaf was photographed several times (different angles, lighting). If photo A of leaf #17 is in the training set and photo B of the same leaf is in the test set, the model is tested on something it has practically already seen. Test accuracy would look better than it really is — this is called **data leakage**.

The dataset ships a file, `leaf_grouping/leaf-map.json`, that tells us which photos show the same leaf. The pipeline:

1. assigns every image a `leaf_id` (same rules as the official loading script),
2. uses the official leaf-grouped train/test split,
3. carves the validation set out of the training set **by leaf, not by image** (`group_split` in `data.py`),
4. **checks** that no `leaf_id` appears in two splits before training starts (`check_no_leakage`), and stops if it does.

### 2.2 Class imbalance

The classes are not the same size — some tomato classes have several thousand images, some healthy classes only a few hundred. Two consequences:

- **Accuracy alone can hide problems**: a model that is bad on small classes can still score high. That's why we also report **macro-F1**, which weights every class equally (§13).
- Optional fix: set `data.class_weighting: true` to give rare classes more weight in the loss: `w_c = N / (K · n_c)`, where N = total images, K = number of classes, n_c = images in class c.

### 2.3 Why we don't call `load_dataset(...)`

The Hugging Face repo uses a Python *loading script* (`plant_village.py`). Recent versions of the `datasets` library (4.0 and newer) no longer execute loading scripts, so `load_dataset("mohanty/PlantVillage", "color")` fails on a fresh install. Our `data.py` downloads the same raw files with `huggingface_hub` (`data.zip`, the split lists and the leaf map) and reproduces the script's logic, so it works with any library version.

---

## 3. The pipeline, step by step

```
 Hugging Face Hub
      │  data.zip, splits/*.txt, leaf-map.json          (data.py: download_plantvillage)
      ▼
 Index images → (path, label, leaf_id)                  (data.py: read_split)
      ▼
 Leaf-grouped split: train / val / test + leakage check (data.py: group_split, check_no_leakage)
      ▼
 Preprocess + augment                                   (data.py: build_transforms)
      ▼
 Model: SimpleCNN | ResNet | EfficientNet              (models.py)
      ▼
 Train: AdamW + cosine LR + early stopping              (train.py)
      ▼
 Evaluate on test: accuracy, F1, confusion matrix       (evaluate.py)
      ▼
 Explain: filters, feature maps, Grad-CAM               (visualize.py)
      ▼
 Deploy: ONNX / TorchScript / Streamlit demo             (export.py, app/)
```

### 3.1 Preprocessing

| Step | Train | Val / Test | Why |
|---|---|---|---|
| Resize / crop | `RandomResizedCrop(224, scale 0.8–1.0)` | `Resize(256)` → `CenterCrop(224)` | Fixed input size; random crops teach the model the leaf can be anywhere |
| Flips | horizontal + vertical | — | A leaf is still the same leaf when flipped |
| Rotation | ±20° | — | Photos aren't always taken straight |
| Colour jitter | brightness ±30 %, contrast ±20 %, saturation ±20 % | — | Sunlight vs shade, different phone cameras |
| To tensor | pixel / 255 → [0, 1] | same | Networks train better on small numbers |
| Normalise | (x − mean) / std per channel | same | Matches what the pretrained ImageNet models expect |

**Normalisation, worked example.** A red-channel pixel with value 150:

```
scale:      150 / 255              = 0.5882
normalise: (0.5882 − 0.485) / 0.229 = 0.4508
```

(ImageNet means are R 0.485, G 0.456, B 0.406; standard deviations 0.229, 0.224, 0.225.)

Augmentation is applied **only to training images**, randomly and freshly every epoch, so the model sees a slightly different version of every photo each time. Validation and test images are never augmented — we want to measure the model on realistic, unmodified inputs.

---

## 4. What an image looks like to a computer

A colour image is a 3-D grid of numbers, a **tensor** of shape `channels × height × width`:

```
224 × 224 RGB image  →  tensor of shape (3, 224, 224)  =  150,528 numbers
```

Each of the 3 channels (red, green, blue) is a 224 × 224 matrix of brightness values. A healthy leaf is mostly high green values; a brown lesion lowers green and raises red relative to it; a yellow halo raises both red and green.

Why not feed those 150,528 numbers straight into a regular (fully-connected) neural network? A single layer of just 1,000 neurons would already need **150 million** weights, it would ignore that neighbouring pixels belong together, and a spot in the top-left corner would have to be learned separately from the same spot in the bottom-right. CNNs solve all three problems.

---

## 5. Convolution: filters and kernels

### 5.1 The idea

A **kernel** (also called a **filter**) is a small grid of weights, typically 3 × 3. It slides across the image; at every position it multiplies its weights with the pixels underneath, adds everything up, and writes one number into the output. The output grid is called a **feature map** — it shows *where in the image* the pattern the kernel is looking for appears.

```
      image patch            kernel              output
   ┌────┬────┬────┐     ┌────┬────┬────┐
   │ x1 │ x2 │ x3 │     │ w1 │ w2 │ w3 │
   ├────┼────┼────┤  ⊙  ├────┼────┼────┤   →   y = Σ xᵢ·wᵢ + b
   │ x4 │ x5 │ x6 │     │ w4 │ w5 │ w6 │
   ├────┼────┼────┤     ├────┼────┼────┤
   │ x7 │ x8 │ x9 │     │ w7 │ w8 │ w9 │
   └────┴────┴────┘     └────┴────┴────┘
```

As a formula (this is what PyTorch's `Conv2d` computes — technically *cross-correlation*, the kernel is not flipped):

$$
Y[i,j] \;=\; b \;+\; \sum_{m=0}^{k-1}\sum_{n=0}^{k-1} X[i\cdot s+m,\; j\cdot s+n]\;\cdot\;K[m,n]
$$

where *k* = kernel size, *s* = stride, *b* = bias.

**The three properties that make CNNs work for images:**

| Property | Meaning | Why it helps for leaves |
|---|---|---|
| **Local connectivity** | each output looks at only a k × k neighbourhood | lesion edges, spots and veins are local patterns |
| **Weight sharing** | the same 9 weights are used at every position | a 3 × 3 kernel has 9 weights, not millions |
| **Translation equivariance** | if the spot moves, its response in the feature map moves with it | a lesion is detected wherever it is on the leaf |

### 5.2 Hand-made kernels: what they detect

Before deep learning, people designed kernels by hand. A CNN's first layer ends up learning kernels that look very much like these — but it chooses them itself, specifically for leaves.

| Kernel | Weights | Detects | On a leaf |
|---|---|---|---|
| Identity | `[0 0 0; 0 1 0; 0 0 0]` | nothing — copies the image | — |
| Box blur | all `1/9` | averages 3 × 3 → removes noise | smooths texture |
| Gaussian blur | `[1 2 1; 2 4 2; 1 2 1] / 16` | weighted average, softer blur | smooths texture |
| Sharpen | `[0 −1 0; −1 5 −1; 0 −1 0]` | boosts the centre against its neighbours | makes small spots stand out |
| Sobel x | `[−1 0 1; −2 0 2; −1 0 1]` | brightness change left → right (vertical edges) | lesion borders, leaf margin |
| Sobel y | `[−1 −2 −1; 0 0 0; 1 2 1]` | brightness change top → bottom (horizontal edges) | lesion borders, veins |
| Laplacian | `[0 1 0; 1 −4 1; 0 1 0]` | second derivative — blobs and spots | rust pustules, leaf spots |

Notice: **a kernel whose weights sum to 0** (Sobel, Laplacian) gives 0 on flat, uniform regions and responds only where something *changes*. **A kernel whose weights sum to 1** (blurs, sharpen) keeps the overall brightness.

> Run `python -m plantdx.visualize kernels --image leaf.jpg` to see all of these applied to a real leaf.

### 5.3 Output size

$$
O \;=\; \left\lfloor \frac{W - K + 2P}{S} \right\rfloor + 1
$$

*W* = input size, *K* = kernel size, *P* = padding, *S* = stride.

| Layer | W | K | P | S | Output |
|---|---|---|---|---|---|
| Our conv (3 × 3, "same") | 224 | 3 | 1 | 1 | ⌊(224 − 3 + 2)/1⌋ + 1 = **224** |
| Same, no padding | 224 | 3 | 0 | 1 | ⌊221/1⌋ + 1 = **222** |
| Max-pool 2 × 2 | 224 | 2 | 0 | 2 | ⌊222/2⌋ + 1 = **112** |
| ResNet first conv (7 × 7, stride 2) | 224 | 7 | 3 | 2 | ⌊223/2⌋ + 1 = **112** |
| Toy example below | 5 | 3 | 0 | 1 | **3** |
| Toy example, padding 1 | 5 | 3 | 1 | 1 | **5** |
| Toy example, padding 1, stride 2 | 5 | 3 | 1 | 2 | ⌊4/2⌋ + 1 = **3** |

- **Padding** adds a border of zeros so the kernel can be centred on edge pixels. With K = 3 and P = 1 the size stays the same ("same" padding).
- **Stride** is how far the kernel jumps each step. Stride 2 halves the size.

---

## 6. Kernel calculations worked by hand

### 6.1 A 5 × 5 image and a 3 × 3 kernel (stride 1, no padding)

Input *X*:
```
1  2  0  1  3
0  1  3  2  1
2  1  0  1  0
1  3  2  0  2
0  1  1  2  1
```

Kernel *K* (a vertical-edge detector: bright on the left minus bright on the right):
```
1  0  -1
1  0  -1
1  0  -1
```

**Output position (0,0)** — the kernel sits on the top-left 3 × 3 patch:

```
patch        kernel       element-wise product
1  2  0      1  0 -1       1   0   0
0  1  3   ⊙  1  0 -1   =   0   0  -3
2  1  0      1  0 -1       2   0   0
```
Sum: 1 + 0 + 0 + 0 + 0 − 3 + 2 + 0 + 0 = **0**

**Output position (0,1)** — slide one step right (columns 1–3):

```
patch        kernel       product
2  0  1      1  0 -1       2  0 -1
1  3  2   ⊙  1  0 -1   =   1  0 -2
1  0  1      1  0 -1       1  0 -1
```
Sum: 2 − 1 + 1 − 2 + 1 − 1 = **0**

**Output position (0,2)** — columns 2–4:

```
patch        product
0  1  3      0  0 -3
3  2  1  →   3  0 -1
0  1  0      0  0  0
```
Sum: −3 + 3 − 1 = **−1**

Doing this for all 9 positions gives the **3 × 3 feature map**:
```
 0   0  -1
-2   2   2
 0   2   0
```
Positive values = brighter on the left than on the right; negative = the reverse; 0 = no vertical edge.

### 6.2 The same with padding 1 → output stays 5 × 5

Add a ring of zeros around *X* and repeat. The top-left output now has the kernel centred on pixel (0,0):

```
padded patch   kernel     product
0  0  0        1  0 -1     0  0  0
0  1  2   ⊙    1  0 -1  =  0  0 -2
0  0  1        1  0 -1     0  0 -1
```
Sum = **−3**. The full 5 × 5 output:
```
-3  -2   0  -1   3
-4   0   0  -1   4
-5  -2   2   2   3
-5   0   2   0   3
-4  -2   2   0   2
```
(The centre 3 × 3 is identical to §6.1 — padding only adds the border positions.)

### 6.3 Padding 1, stride 2 → 3 × 3

Take every second position of the padded result (rows/cols 0, 2, 4):
```
-3   0   3
-5   2   3
-4   2   2
```

### 6.4 Why an edge kernel finds a lesion border

A healthy area (value 200) next to a dark lesion (value 50):
```
200 200 200  50  50
200 200 200  50  50
200 200 200  50  50        Sobel x:  -1  0  1
200 200 200  50  50                  -2  0  2
200 200 200  50  50                  -1  0  1
```
- Patch entirely inside the healthy area: (−1 − 2 − 1)·200 + (1 + 2 + 1)·200 = **0** → "nothing here".
- Patch across the border: (−1 − 2 − 1)·200 + (1 + 2 + 1)·50 = −800 + 200 = **−600** → "strong edge here".

Feature map:
```
0  -600  -600
0  -600  -600
0  -600  -600
```
The feature map lights up exactly along the border of the lesion. Deeper layers combine many such maps: "edge" + "brown colour" + "round shape" + "yellow halo around it" → "this looks like early blight".

---

## 7. Colour images: kernels with depth

With an RGB input, **one filter is a 3-D block**: one 3 × 3 kernel *per input channel*, so 3 × 3 × 3 = 27 weights + 1 bias. The three channel results are **added together** into a single number — one filter always produces **one** feature map.

Worked example on a 3 × 3 patch:

```
Red patch      Red kernel        Green patch    Green kernel      Blue patch     Blue kernel
1 0 2          1  0 -1           0 1 1          0  1  0           1 1 0          1 0  0
0 1 1          0  1  0           1 2 0          1 -1  1           0 0 1          0 1  0
2 1 0         -1  0  1           1 0 1          0  1  0           2 1 1          0 0 -1
```

| Channel | Element-wise products that aren't zero | Sum |
|---|---|---|
| Red | 1·1 + 2·(−1) + 1·1 + 2·(−1) | **−2** |
| Green | 1·1 + 1·1 + 2·(−1) | **0** |
| Blue | 1·1 + 0·1 + 1·(−1) | **0** |

Output = −2 + 0 + 0 + bias. With bias *b* = 3 → **1**.

**A layer has many filters.** `Conv2d(3, 32, kernel_size=3)` has 32 such filters, so it outputs 32 feature maps (32 channels). The next layer's filters then have depth 32 (3 × 3 × 32 weights each), and so on.

**Parameters of a conv layer:**

$$
\text{params} = K \cdot K \cdot C_{in} \cdot C_{out} \;(+\,C_{out} \text{ biases})
$$

`Conv2d(3, 32, 3)`: 3·3·3·32 = 864 weights (+32 biases). In our model we set `bias=False` because the BatchNorm layer right after it has its own shift parameter, which makes a conv bias redundant.

**Computation (multiply–adds) of a conv layer:**

$$
\text{MACs} = H_{out} \cdot W_{out} \cdot K^2 \cdot C_{in} \cdot C_{out}
$$

---

## 8. The other layers: ReLU, BatchNorm, pooling

### 8.1 ReLU — the non-linearity

$$\text{ReLU}(x) = \max(0, x)$$

```
input   [-3,  -2,  0,  1,  4]
ReLU    [ 0,   0,  0,  1,  4]
```
Without a non-linearity, stacking 8 conv layers would collapse into one big linear filter — the network could only ever learn one-step patterns. ReLU lets layers build on each other. It also acts as "keep only where the pattern was found": negative responses (pattern absent) are set to zero.

### 8.2 Batch normalisation

For each channel, over a mini-batch: subtract the mean, divide by the standard deviation, then rescale with two learnable parameters γ (scale) and β (shift):

$$
\hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}, \qquad y = \gamma\,\hat{x} + \beta
$$

Example — four activations `[2, 4, 6, 8]`: μ = 5, σ² = 5

```
x̂ = [(2−5)/√5, (4−5)/√5, (6−5)/√5, (8−5)/√5] = [−1.342, −0.447, 0.447, 1.342]
```
With γ = 1, β = 0 that's the output. BatchNorm keeps activations in a stable range, so we can use a higher learning rate and the network trains faster and more reliably. During training it uses the batch's statistics; during evaluation it uses running averages collected while training (`model.eval()` switches this).

### 8.3 Max pooling

A 2 × 2 window with stride 2 keeps the largest value of each block:
```
1  3 │ 2  4
5  6 │ 1  2          6  4
─────┼─────    →
7  2 │ 9  0          8  9
1  8 │ 3  4
```
Effects: height and width are halved (4× fewer numbers, 4× less computation for the next layer); the strongest response survives, so small shifts of a spot don't change the result; and the next layer's kernels effectively see a twice-as-large area of the original image.

### 8.4 Global average pooling (GAP)

At the end, each of the 256 feature maps (14 × 14) is averaged into one number → a vector of 256 values. Example: the 4 × 4 map above averages to 58 / 16 = **3.625**. GAP means "how strongly was feature *k* present anywhere on the leaf", and it has no parameters — much less overfitting than flattening 256 × 14 × 14 = 50,176 values into a big fully-connected layer.

### 8.5 Receptive field — how much of the leaf one neuron "sees"

Each 3 × 3 conv grows the visible area by 2 × (current jump); each 2 × 2 pool grows it by 1 × jump and doubles the jump.

| After | Receptive field | Jump |
|---|---|---|
| input | 1 px | 1 |
| block 1 (conv, conv, pool) | 3 → 5 → 6 px | 2 |
| block 2 | 10 → 14 → 16 px | 4 |
| block 3 | 24 → 32 → 36 px | 8 |
| block 4 | 52 → 68 → **76 px** | 16 |

So early neurons see a few pixels (edges, colour), and the last block's neurons each see a 76 × 76 px region — about a third of the image, big enough to cover a whole lesion with its halo. GAP then combines all of them.

---

## 9. Our SimpleCNN, layer by layer

```
Input (3, 224, 224)
│
├─ Block 1: Conv3×3(3→32) → BN → ReLU → Conv3×3(32→32) → BN → ReLU → MaxPool2×2   → (32, 112, 112)
├─ Block 2: Conv3×3(32→64) → BN → ReLU → Conv3×3(64→64) → BN → ReLU → MaxPool2×2  → (64, 56, 56)
├─ Block 3: Conv3×3(64→128) → BN → ReLU → Conv3×3(128→128) → BN → ReLU → MaxPool  → (128, 28, 28)
├─ Block 4: Conv3×3(128→256) → BN → ReLU → Conv3×3(256→256) → BN → ReLU → MaxPool → (256, 14, 14)
│
├─ Global average pool                                                         → (256)
├─ Dropout (p = 0.3)
└─ Linear 256 → 38                                                             → (38) logits
```

Design choices:
- **Channels double while the size halves** (32 → 64 → 128 → 256; 224 → 112 → 56 → 28 → 14). Early layers need few filters for simple things like edges; deeper layers need many for the many combinations (spot shapes, textures, colours).
- **Two 3 × 3 convs per block**: two stacked 3 × 3 convs see a 5 × 5 area, but with 2 · 9 = 18 weights per channel pair instead of 25, and with an extra ReLU in between (more expressive).
- **Dropout** randomly switches off 30 % of the 256 features during training, so the classifier can't rely on any single feature.

### 9.1 Parameter count (calculated)

Conv weights: K·K·C_in·C_out (no bias). BN: 2·C (γ and β). Linear: 256·38 + 38.

| Layer | Calculation | Parameters |
|---|---|---|
| B1 conv1 | 3·3·3·32 | 864 |
| B1 BN1 | 2·32 | 64 |
| B1 conv2 | 3·3·32·32 | 9,216 |
| B1 BN2 | 2·32 | 64 |
| **Block 1** | | **10,208** |
| B2 conv1 + BN | 3·3·32·64 + 2·64 | 18,560 |
| B2 conv2 + BN | 3·3·64·64 + 2·64 | 36,992 |
| **Block 2** | | **55,552** |
| B3 conv1 + BN | 3·3·64·128 + 2·128 | 73,984 |
| B3 conv2 + BN | 3·3·128·128 + 2·128 | 147,712 |
| **Block 3** | | **221,696** |
| B4 conv1 + BN | 3·3·128·256 + 2·256 | 295,424 |
| B4 conv2 + BN | 3·3·256·256 + 2·256 | 590,336 |
| **Block 4** | | **885,760** |
| Classifier | 256·38 + 38 | **9,766** |
| **Total** | | **1,182,982** |

Three-quarters of the parameters are in the last block — that's where the most filters with the most depth live. For comparison, ResNet-18 has about 11.2 M parameters.

### 9.2 Computation (calculated, per image)

| Layer | H·W out | K²·C_in·C_out | MACs |
|---|---|---|---|
| B1 conv1 | 224·224 | 9·3·32 | 43.4 M |
| B1 conv2 | 224·224 | 9·32·32 | 462.4 M |
| B2 conv1 | 112·112 | 9·32·64 | 231.2 M |
| B2 conv2 | 112·112 | 9·64·64 | 462.4 M |
| B3 conv1 | 56·56 | 9·64·128 | 231.2 M |
| B3 conv2 | 56·56 | 9·128·128 | 462.4 M |
| B4 conv1 | 28·28 | 9·128·256 | 231.2 M |
| B4 conv2 | 28·28 | 9·256·256 | 462.4 M |
| **Total (convs)** | | | **≈ 2.59 G** |

Note the convs run *before* each pool, so block *n* works at the resolution of the previous block's output. Each block costs about the same — halving H and W (4× fewer positions) is balanced by doubling C_in and C_out (4× more work per position).

---

## 10. From features to a diagnosis: softmax and loss

### 10.1 Logits → probabilities (softmax)

The final linear layer outputs 38 raw scores called **logits**. Softmax turns them into probabilities:

$$
p_i = \frac{e^{z_i}}{\sum_j e^{z_j}}
$$

Example with three classes, logits z = [2.0, 1.0, 0.1]:

| class | z | e^z | p = e^z / 11.213 |
|---|---|---|---|
| Tomato early blight | 2.0 | 7.389 | **0.659** |
| Tomato late blight | 1.0 | 2.718 | 0.242 |
| Tomato healthy | 0.1 | 1.105 | 0.099 |
| **sum** | | 11.213 | 1.000 |

### 10.2 Cross-entropy loss

$$
L = -\log p_{\text{true class}}
$$

- If the true class is early blight: L = −log(0.659) = **0.417** (fairly confident and correct → small loss).
- If the true class is healthy: L = −log(0.099) = **2.317** (wrong → large loss).

The loss punishes confident mistakes very hard (−log of a small number is large), which is exactly what we want.

### 10.3 Label smoothing (ε = 0.1)

Instead of a target of 100 % for the true class and 0 % for the others, we use:

- true class: 1 − ε + ε/K = 1 − 0.1 + 0.1/38 = **0.9026**
- every other class: ε/K = 0.1/38 = **0.0026**

This stops the model from becoming over-confident (pushing logits to infinity), which usually improves accuracy on new images and makes the probabilities more honest.

---

## 11. How the network learns its kernels

At the start, all kernel weights are random numbers. Training repeats:

1. **Forward pass** — a batch of 64 images goes through the network → 64 × 38 probabilities → average loss L.
2. **Backward pass (backpropagation)** — the chain rule computes ∂L/∂w for *every* weight: "if I nudge this weight up a little, does the loss go up or down, and by how much?"
3. **Update** — move every weight a small step against its gradient: w ← w − η · ∂L/∂w (η = learning rate).

**The gradient of a kernel is itself a convolution.** For a kernel weight K[m,n]:

$$
\frac{\partial L}{\partial K[m,n]} = \sum_{i,j} \frac{\partial L}{\partial Y[i,j]} \cdot X[i+m,\, j+n]
$$

In words: correlate the input with the error signal of the output. The weight changes most where the input pattern lined up with a large error. Over thousands of batches, the first-layer kernels turn into edge and colour detectors and deeper kernels into detectors for spots, rings, mould textures, curled leaf shapes — whatever helps separate the 38 classes.

**Mini example of one update:** weight w = 0.50, gradient ∂L/∂w = 0.20, learning rate η = 0.001 → new w = 0.50 − 0.001 · 0.20 = **0.4998**. Tiny steps, millions of times.

### Training settings used

| Setting | Value | Why |
|---|---|---|
| Optimiser | AdamW | Adam adapts the step size per weight; "W" = decoupled weight decay (keeps weights small, less overfitting) |
| Learning rate | 1e-3 (scratch), 3e-4 – 5e-4 (pretrained) | pretrained weights are already good — smaller steps avoid destroying them |
| LR schedule | cosine annealing | big steps early, fine steps at the end |
| Batch size | 64 | |
| Label smoothing | 0.1 | §10.3 |
| Mixed precision | on GPU | ~2× faster, half the memory |
| Early stopping | patience 3–6 epochs on validation macro-F1 | stop when the model stops improving on unseen leaves |
| Checkpoint | best validation macro-F1 | the test set is used **once**, at the very end |

---

## 12. Transfer learning: ResNet, EfficientNet

### 12.1 The idea

A network trained on ImageNet's 1.2 million photos has already learned general visual features: edges, textures, colours, shapes. Those are just as useful for leaves. So instead of starting from random kernels:

1. take the pretrained network,
2. replace its last layer (1,000 ImageNet classes) with a new one for our 38 classes,
3. **warm-up**: freeze the backbone and train only the new head for 1 epoch (`freeze_backbone_epochs: 1`) — otherwise the big random gradients of the new head would wreck the good pretrained kernels,
4. **fine-tune**: unfreeze everything and train with a small learning rate.

Result: higher accuracy in fewer epochs than training from scratch, and much less risk of overfitting.

### 12.2 ResNet — residual connections

Very deep plain networks are hard to train: the gradient fades as it passes back through many layers. ResNet adds a **shortcut** around every pair of convs:

$$
y = F(x) + x
$$

The block only has to learn the *difference* F(x) from its input, and the gradient can flow straight back through the "+ x" path. ResNet-18 has 18 layers and ~11.2 M parameters; its first layer is a 7 × 7, stride-2 conv with 64 filters.

### 12.3 EfficientNet — balanced scaling

EfficientNet scales network **depth** (layers), **width** (channels) and **input resolution** together with one compound factor instead of just stacking more layers. Its building block (MBConv) uses depthwise-separable convolutions plus "squeeze-and-excitation", which learns a weight per channel ("this feature map matters more for this image"). EfficientNet-B0 has ~4.0 M parameters and is usually as accurate as much bigger networks.

### 12.4 What to expect

| Model | Params | Trained from | Typical outcome on PlantVillage |
|---|---|---|---|
| SimpleCNN | 1.18 M | scratch | good, needs more epochs |
| ResNet-18 | 11.2 M | ImageNet | very high, fast convergence |
| EfficientNet-B0 | 4.0 M | ImageNet | very high, efficient |

The original paper reached 99.35 % accuracy with GoogLeNet transfer learning on the colour images. Very high numbers are normal here — the interesting part of the comparison is the **trade-off** between accuracy, size and speed, and the limitations in §15.

---

## 13. Evaluation metrics

For one class *c*: TP = correctly predicted as *c*, FP = wrongly predicted as *c*, FN = *c* images predicted as something else.

| Metric | Formula | Question it answers |
|---|---|---|
| Accuracy | correct / all | How often is the model right? |
| Precision_c | TP / (TP + FP) | When it says "late blight", how often is it really late blight? |
| Recall_c | TP / (TP + FN) | Of all late-blight leaves, how many did it catch? |
| F1_c | 2 · P · R / (P + R) | Balance of precision and recall |
| **Macro-F1** | mean of F1_c over all 38 classes | Is it good on **every** class, including the small ones? |
| Weighted-F1 | F1_c averaged, weighted by class size | Overall F1 that follows the data distribution |
| Top-3 accuracy | true class among the 3 highest probabilities | Useful for an app that shows "most likely: A, B or C" |

Example: 100 late-blight test leaves; the model predicts late blight 95 times, 90 correct. → TP = 90, FP = 5, FN = 10 → precision = 90/95 = 0.947, recall = 90/100 = 0.900, F1 = 2·0.947·0.900/(0.947 + 0.900) = **0.923**.

**For farmers, recall on diseases matters most**: missing a disease (FN) costs a harvest; a false alarm (FP) costs a second look.

**Confusion matrix** (`confusion_matrix.png`): rows = true class, columns = predicted class, each row normalised to 1. The diagonal is "correct"; anything off the diagonal shows *which* classes get mixed up — for example tomato early blight vs target spot, which genuinely look similar. `top_confusions.csv` lists the most common mix-ups.

---

## 14. Looking inside: filters, feature maps, Grad-CAM

| Command | What you see |
|---|---|
| `visualize kernels --image leaf.jpg` | the hand-made kernels of §5.2 applied to a leaf |
| `visualize filters --checkpoint best.pt` | the learned first-layer kernels (3 × 3 for SimpleCNN, 7 × 7 for ResNet), shown as tiny RGB images — expect edge-like and colour-blob patterns |
| `visualize featuremaps --checkpoint best.pt --image leaf.jpg` | outputs of the first 3 conv layers: which parts of the leaf each filter responds to |
| `visualize gradcam --checkpoint best.pt --image leaf.jpg` | heat map of the regions that drove the decision |

**Grad-CAM** (Selvaraju et al., 2017). Take the last conv stage's feature maps A_k (e.g. 256 maps of 14 × 14). For the predicted class score y_c:

$$
\alpha_k = \frac{1}{Z}\sum_{i,j}\frac{\partial y_c}{\partial A_k[i,j]}, \qquad
\text{CAM} = \text{ReLU}\Big(\sum_k \alpha_k A_k\Big)
$$

α_k = "how important is feature map *k* for this class" (average gradient). The weighted sum is upsampled to 224 × 224 and overlaid on the photo. **What you want to see:** heat on the lesions. **Red flag:** heat on the background or image border — the model may be using shortcuts instead of the disease (see §15).

---

## 15. Limitations worth discussing

1. **Lab photos vs field photos.** PlantVillage images show one leaf, on a plain background, in controlled light. Real farm photos have soil, other leaves, shadows and hands in them. The original authors tested their model on images collected from other sources and accuracy dropped to about 31 % — very high PlantVillage accuracy does not mean the model is ready for a field. Possible next steps: test on a field dataset (e.g. PlantDoc), stronger augmentation, or fine-tuning on field images.
2. **Background shortcuts.** Since each class was often photographed in its own session, background and lighting can correlate with the class. Train once on `color` and once on `segmented` (background removed) and compare; check Grad-CAM.
3. **Closed set.** The model can only answer one of 38 classes. A leaf of a crop it has never seen, or a disease not in the list, will still get one of those 38 labels — a confidence threshold ("not sure, please check") is a sensible addition to an app.
4. **One disease per image.** A leaf with two problems at once can't be described.
5. **Class imbalance.** Report macro-F1 and per-class recall, not just accuracy.

---

## 16. Cheat sheet of formulas

| What | Formula |
|---|---|
| Convolution | Y[i,j] = b + Σ_m Σ_n X[i·s+m, j·s+n] · K[m,n] |
| Output size | O = ⌊(W − K + 2P) / S⌋ + 1 |
| Conv parameters | K² · C_in · C_out (+ C_out bias) |
| Conv MACs | H_out · W_out · K² · C_in · C_out |
| ReLU | max(0, x) |
| BatchNorm | y = γ · (x − μ) / √(σ² + ε) + β |
| Softmax | p_i = e^{z_i} / Σ_j e^{z_j} |
| Cross-entropy | L = −log p_true |
| Label smoothing | target = 1 − ε + ε/K (true), ε/K (others) |
| Gradient descent | w ← w − η · ∂L/∂w |
| Kernel gradient | ∂L/∂K[m,n] = Σ_ij ∂L/∂Y[i,j] · X[i+m, j+n] |
| Residual block | y = F(x) + x |
| Depthwise-separable saving | ratio ≈ 1/C_out + 1/K² |
| Class weight | w_c = N / (K · n_c) |
| F1 | 2PR / (P + R) |
| Grad-CAM | ReLU(Σ_k α_k A_k), α_k = mean_ij ∂y_c/∂A_k |

---

*Reference: Mohanty, S. P., Hughes, D. P., & Salathé, M. (2016). Using deep learning for image-based plant disease detection. Frontiers in Plant Science, 7, 1419. doi:10.3389/fpls.2016.01419*
