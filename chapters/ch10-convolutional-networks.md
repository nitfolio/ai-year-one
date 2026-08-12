# Chapter 10 — Convolutional Networks

**Time: 12–14 days** (Weeks 20–22 of the plan)

**Prerequisite:** Chapter 8's trainer, Chapter 9's optimizers. §7.7's residual preview.

**What you'll be able to do at the end:** derive convolution from the problem it solves rather than accept it as a formula; implement `conv2d` forward and backward from scratch; compute output shapes, parameter counts and receptive fields in your head; explain why ResNets made 100-layer networks trainable; and train a competitive image classifier.

CNNs also matter beyond vision — the *ideas* here (weight sharing, receptive fields, residual connections) reappear throughout the field, and residual connections are load-bearing in every transformer you'll build in Chapter 12.

---

## 10.0 Why MLPs fail on images

Take a modest 224×224 colour image. Flattened, that's **150,528 inputs**. A single hidden layer of 1,000 units needs

```
150,528 × 1,000 ≈ 150 million parameters
```

for *one layer*. That's already larger than most models of its era, and it hasn't done anything yet.

But the parameter count isn't the deepest problem. Two structural failures matter more:

**No translation invariance.** A cat in the top-left corner and the same cat in the bottom-right produce completely different input vectors. An MLP must learn "cat" separately at every position. Every training example teaches it about one location only.

**It ignores spatial structure entirely.** Here's the proof, and it's a five-minute experiment worth running (exercise 1): apply a **fixed random permutation** to the pixels of every image in your dataset — same permutation for all of them. Retrain.

An MLP's accuracy is **completely unchanged.** A CNN's collapses.

That tells you something exact: the MLP was never using the fact that neighbouring pixels are related. All that structure — which is most of what an image *is* — was thrown away before learning started.

---

## 10.1 The three ideas

Convolution is what you get if you take three observations about images seriously.

**1. Local connectivity.** Pixels near each other are related; pixels far apart mostly aren't. So a unit should look at a small patch — say 3×3 — not the whole image. That alone cuts parameters by orders of magnitude.

**2. Weight sharing.** An edge detector useful in the top-left is useful everywhere. So use the *same* small set of weights at every position. This is the big one: it reduces parameters enormously **and** it means every training example teaches the filter about every location at once.

**3. Translation equivariance.** Shift the input, and the output shifts identically. This is a mathematical guarantee that falls out of weight sharing, not something learned.

Careful with the terminology: convolution gives **equivariance** (shift in → shift out), not **invariance** (shift in → same output). Pooling and global average pooling add approximate invariance later.

**The parameter comparison, for one layer producing 64 feature maps from a 224×224×3 image:**

| | Parameters |
|---|---|
| Fully connected (to 64×224×224 outputs) | ~4.8 × 10¹¹ |
| 3×3 convolution, 3→64 channels | **1,792** |

Eight orders of magnitude, and the convolution generalizes better. That's what encoding a correct assumption about your data buys you.

---

## 10.2 The operation

Slide a small kernel over the input; at each position compute the dot product of the kernel with the patch beneath it.

```
input 5×5              kernel 3×3          output 3×3
┌─────────────┐        ┌───────┐
│ a b c · ·   │        │ w w w │           each output =
│ d e f · ·   │   ⊛    │ w w w │     =     Σ (kernel · patch)
│ g h i · ·   │        │ w w w │
│ · · · · ·   │        └───────┘
│ · · · · ·   │
└─────────────┘
```

Every output value is a dot product — §2.3 again. A convolutional layer is asking, at every spatial position, *how much does this patch look like the pattern I'm tuned to?*

### Terminology note

What deep learning calls "convolution" is technically **cross-correlation** — true convolution flips the kernel first. Since the kernel is learned, the flip makes no difference to what the network can represent, so everyone dropped it. Mentioned only so the signal-processing literature doesn't confuse you.

### The four knobs

- **Kernel size `k`** — how big the patch is. 3×3 is the modern default.
- **Stride `s`** — how far the kernel moves each step. `s=2` halves the spatial size.
- **Padding `p`** — zeros added around the border. `p=(k−1)/2` with `s=1` preserves size ("same" padding).
- **Dilation `d`** — gaps between kernel elements. Enlarges the receptive field without more parameters.

### The output size formula

Memorize this. You'll use it constantly:

```
out = ⌊ (in + 2p − d(k−1) − 1) / s ⌋ + 1
```

For the common case `d = 1`:

```
out = ⌊ (in + 2p − k)/s ⌋ + 1
```

Quick checks:
- `in=32, k=3, p=1, s=1` → `32`. Same size. This is why 3×3 with padding 1 is everywhere.
- `in=32, k=3, p=1, s=2` → `16`. Halved.
- `in=32, k=5, p=0, s=1` → `28`. Shrinks by `k−1`.

---

## 10.3 Channels — the part people get wrong

This is the most common confusion in CNNs, so read it slowly.

A convolutional layer with `C_in` input channels and `C_out` output channels has:

```
C_out filters, each of shape (C_in, k, k)
```

**Each filter spans all input channels.** It slides over the spatial dimensions only, and at each position it computes a dot product over the entire `(C_in, k, k)` volume — producing **one number**, which lands in **one output channel**.

So it is *not* "a k×k kernel applied to each channel separately." Every output channel is a learned combination of all input channels.

**Parameter count:**

```
k × k × C_in × C_out   +   C_out       (weights + biases)
```

Check the earlier number: `3 × 3 × 3 × 64 + 64 = 1,728 + 64 = 1,792` ✓

**Shape convention** (PyTorch):

```
input:   (N, C_in,  H,     W)
weight:  (C_out, C_in, kh, kw)
output:  (N, C_out, H_out, W_out)
```

---

## 10.4 Implementing it

Naive version first. Slow, obviously correct, and the one you should write before anything clever.

```python
import numpy as np

def conv2d_naive(X, W, b, stride=1, pad=0):
    """
    X : (N, C_in, H, Wd)
    W : (C_out, C_in, kh, kw)
    b : (C_out,)
    returns (N, C_out, H_out, W_out)
    """
    N, C_in, H, Wd = X.shape
    C_out, _, kh, kw = W.shape

    Xp = np.pad(X, ((0,0), (0,0), (pad,pad), (pad,pad)))
    H_out = (H + 2*pad - kh) // stride + 1
    W_out = (Wd + 2*pad - kw) // stride + 1

    out = np.zeros((N, C_out, H_out, W_out))
    for n in range(N):
        for co in range(C_out):
            for i in range(H_out):
                for j in range(W_out):
                    h0, w0 = i*stride, j*stride
                    patch = Xp[n, :, h0:h0+kh, w0:w0+kw]     # (C_in, kh, kw)
                    out[n, co, i, j] = np.sum(patch * W[co]) + b[co]
    return out
```

Six nested loops. It will be thousands of times slower than PyTorch. **Write it anyway** — this is the build-before-you-import rule, and having written it, the fast version below will make sense instead of being magic.

---

## 10.5 im2col: convolution as matrix multiplication

The trick that made CNNs practical, and the reason GPUs — which are matmul machines — are so good at them.

**The idea:** extract every patch the kernel will see, lay each one out as a row of a matrix, and the entire convolution becomes a single matrix multiply.

```python
def im2col(X, kh, kw, stride=1, pad=0):
    """(N,C,H,W) -> (N*H_out*W_out, C*kh*kw)"""
    N, C, H, Wd = X.shape
    Xp = np.pad(X, ((0,0), (0,0), (pad,pad), (pad,pad)))
    H_out = (H + 2*pad - kh) // stride + 1
    W_out = (Wd + 2*pad - kw) // stride + 1

    cols = np.zeros((N, C, kh, kw, H_out, W_out))
    for i in range(kh):
        for j in range(kw):
            cols[:, :, i, j, :, :] = Xp[:, :,
                                        i : i + stride*H_out : stride,
                                        j : j + stride*W_out : stride]
    return cols.transpose(0, 4, 5, 1, 2, 3).reshape(N*H_out*W_out, -1)


def conv2d_fast(X, W, b, stride=1, pad=0):
    N, _, H, Wd = X.shape
    C_out, C_in, kh, kw = W.shape
    H_out = (H + 2*pad - kh) // stride + 1
    W_out = (Wd + 2*pad - kw) // stride + 1

    cols = im2col(X, kh, kw, stride, pad)            # (N*H_out*W_out, C_in*kh*kw)
    W_row = W.reshape(C_out, -1)                     # (C_out, C_in*kh*kw)

    out = cols @ W_row.T + b                         # (N*H_out*W_out, C_out)
    return out.reshape(N, H_out, W_out, C_out).transpose(0, 3, 1, 2)
```

Expect a 100–1000× speedup over the naive version. **Measure it** (exercise 5) — seeing the number yourself is the point.

### And the backward pass comes free

This is the elegant part. Once convolution is `cols @ W_row.T`, the backward pass is just **matmul backward**, which you derived in §2.9 and implemented in §5.8:

```
dW_row = dOut.T @ cols          then reshape to (C_out, C_in, kh, kw)
dcols  = dOut @ W_row           then col2im back to (N, C_in, H, W)
db     = dOut.sum(axis=0)
```

`col2im` is the inverse of `im2col`, and it **accumulates** where patches overlap — because a pixel that appeared in several patches influenced the loss through several routes. That's the multi-path rule (§3.5) once more.

**Convolution's backward pass is itself a convolution** (a "full" one with a flipped kernel). You can derive that directly if you want, but the im2col route gets you a correct implementation without any of that work, which is why it's the path this chapter takes.

---

## 10.6 Receptive field

The **receptive field** of a unit is the region of the *original input* that can affect it.

Stack two 3×3 convolutions and each output unit sees a 5×5 input region. Three gives 7×7.

```
r_l = r_(l−1) + (k_l − 1) · ∏_(i<l) s_i
```

### Why VGG replaced big kernels with stacks of 3×3

Compare one 7×7 convolution against three stacked 3×3 convolutions, both with `C` input and output channels:

| | Receptive field | Parameters | Nonlinearities |
|---|---|---|---|
| One 7×7 | 7×7 | `49C²` | 1 |
| Three 3×3 | 7×7 | `27C²` | **3** |

**Same receptive field, 45% fewer parameters, three times the nonlinearity.** That's a straight win, and it's why essentially every modern CNN uses 3×3 kernels almost exclusively.

**Compute your network's receptive field.** If the final layer's receptive field is smaller than the objects you're trying to classify, the network physically cannot see them — no amount of training fixes that.

---

## 10.7 Pooling

Downsample by summarizing each small region:

```python
def maxpool2d(X, k=2, stride=2):
    N, C, H, W = X.shape
    H_out, W_out = (H - k)//stride + 1, (W - k)//stride + 1
    out = np.zeros((N, C, H_out, W_out))
    for i in range(H_out):
        for j in range(W_out):
            out[:, :, i, j] = X[:, :, i*stride:i*stride+k,
                                       j*stride:j*stride+k].max(axis=(2,3))
    return out
```

Pooling reduces spatial size, enlarges the receptive field cheaply, and adds **approximate translation invariance** — small shifts don't change the max.

Backward for max pooling: gradient flows only to the position that *was* the max. Everything else gets zero. It's a router.

**Modern practice has largely moved on.** Strided convolutions do the downsampling instead, since they can learn how to summarize rather than being told. **Global average pooling** at the end — averaging each channel over all positions — replaced the giant fully-connected layers of AlexNet and VGG, cutting most of their parameters.

---

## 10.8 The architectures, and what each contributed

| Year | Model | The idea it introduced |
|---|---|---|
| 1998 | LeNet-5 | Convolution + pooling + FC. Proved the concept on digits. |
| 2012 | AlexNet | Depth + ReLU + dropout + GPUs. Won ImageNet by a huge margin, started the era. |
| 2014 | VGG | Only 3×3 convs, very deep, uniform design. Showed depth was the lever. |
| 2014 | GoogLeNet | Parallel multi-scale paths; 1×1 bottlenecks for cheap channel reduction. |
| 2015 | **ResNet** | Residual connections. 152 layers. The breakthrough. |
| 2017+ | MobileNet, EfficientNet | Depthwise separable convs; principled scaling of depth/width/resolution. |
| 2022 | ConvNeXt | CNNs redesigned with transformer-era training recipes; competitive again. |

### 1×1 convolutions

A 1×1 kernel has no spatial extent at all, so what does it do? It **mixes channels without mixing space** — it's a per-pixel linear layer applied across the channel dimension.

Uses: reduce `C_in`→`C_out` cheaply before an expensive 3×3 (bottlenecks), or increase channel count afterwards. Parameter cost is `C_in × C_out`, with no `k²` factor.

### Depthwise separable convolutions

Split a convolution into two cheaper pieces:

1. **Depthwise** — one `k×k` filter per input channel, spatial mixing only. Cost `k²·C`.
2. **Pointwise** — a 1×1 convolution, channel mixing only. Cost `C·C_out`.

```
standard:   k²·C·C_out
separable:  k²·C + C·C_out
```

For `k=3, C=C_out=256`: 590K versus 68K — an 8.6× reduction with a small accuracy cost. This is the core of MobileNet and why CNNs run on phones.

---

## 10.9 ResNet, properly

### The problem it solved

By 2015 the obvious move was "go deeper," and it stopped working. A 56-layer plain network had **higher training error** than a 20-layer one.

Read that again: **training** error, not test error. That rules out overfitting entirely.

It's also mathematically strange, because a 56-layer network can trivially represent any 20-layer one — set the extra 36 layers to the identity. The deeper model's solution space *contains* the shallower model's solution. So the deeper network should never be worse.

**Therefore this was an optimization failure, not a capacity failure.** Gradient descent could not find a solution that provably existed. That framing is what made the fix findable.

### The fix

```
y = F(x) + x
```

Instead of learning the mapping `H(x)` directly, learn the **residual** `F(x) = H(x) − x`, and add the input back.

**Why this makes optimization easier.** If the identity is the right answer, a plain stack has to learn a nonlinear composition that happens to equal the identity — hard. A residual block just needs `F(x) → 0`, which means driving weights toward zero — which is exactly what gradient descent and weight decay do naturally.

**Why gradients flow.** Differentiate:

```
∂y/∂x = 1 + ∂F/∂x
```

There is a path with derivative **exactly 1**. Gradient reaches every layer without being multiplied by a chain of small numbers. §7.4's exponential decay is bypassed entirely.

**A third framing worth knowing:** unrolled, a network of `n` residual blocks is an implicit ensemble of `2ⁿ` paths of differing depths — you can go through or around each block. Short paths carry gradient effectively even when long ones don't.

```python
class ResidualBlock(nn.Module):
    def __init__(self, c_in, c_out, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, 3, stride, 1, bias=False)
        self.bn1   = nn.BatchNorm2d(c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, 1, 1, bias=False)
        self.bn2   = nn.BatchNorm2d(c_out)

        # the shortcut must match shape when stride or channels change
        self.shortcut = nn.Sequential()
        if stride != 1 or c_in != c_out:
            self.shortcut = nn.Sequential(
                nn.Conv2d(c_in, c_out, 1, stride, bias=False),
                nn.BatchNorm2d(c_out),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)         # ← the whole idea
        return F.relu(out)
```

Two details that matter: `bias=False` on convolutions followed by BatchNorm (BN has its own `β`, so the conv bias is redundant), and the shortcut needs a 1×1 projection whenever shape changes.

### Bottleneck blocks

For deep ResNets (50+), a cheaper block: `1×1` reduce → `3×3` → `1×1` expand. The expensive 3×3 operates in a low-dimensional space.

### The connection to Chapter 12

**Every transformer is a residual network.** Each attention block and each MLP block is wrapped in `x + Sublayer(x)`. The residual stream running through a 96-layer GPT is the same idea invented here for image classification.

Remember §7.6 on pre-norm versus post-norm — that's a question about *where to put normalization relative to this residual path*, and it's why modern transformers train stably at depth.

---

## 10.10 Data augmentation

Regularization that works by encoding invariances you know are true. If a horizontally flipped cat is still a cat, tell the model.

```python
from torchvision import transforms

train_tf = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.2, 0.2, 0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),        # training statistics only — §4.7
])

test_tf = transforms.Compose([              # NO augmentation at test time
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])
```

Stronger methods: **Mixup** (train on convex combinations of two images and their labels), **CutMix** (paste a patch of one image into another), **RandAugment** (sample from a pool of operations).

**Two rules:**

1. **Never augment the validation or test set.** You'd be measuring performance on data that doesn't match deployment.
2. **Only apply augmentations that preserve the label.** Horizontal flip is fine for cats, wrong for handwritten digits — a flipped `2` isn't a `2`.

Augmentation is often the single highest-return change on small datasets. Try it before adding capacity.

---

## 10.11 Transfer learning

Almost nobody trains an image model from scratch. Take one pretrained on a large dataset and adapt it.

**Why it works:** early layers learn generic features — edges, textures, colour blobs — that are useful for essentially any visual task. Only the later layers are task-specific.

```python
import torchvision.models as models

model = models.resnet50(weights="IMAGENET1K_V2")

for p in model.parameters():             # freeze the backbone
    p.requires_grad = False

model.fc = nn.Linear(2048, n_classes)    # new head, trainable by default
```

**Strategy by dataset size:**

| Your data | Approach |
|---|---|
| Very small (<1k) | Freeze everything, train only the new head |
| Small (1k–10k) | Train the head, then unfreeze the last block at a low LR |
| Medium (10k–100k) | Fine-tune everything, LR ~10× lower than from-scratch |
| Large (>100k) | Fine-tune everything, or consider training from scratch |

**Discriminative learning rates** — lower for early layers, higher for late ones — usually beat a single rate, since early layers need less adjustment.

---

## 10.12 Exercises

**1.** **The permutation experiment.** Apply one fixed random pixel permutation to every MNIST image. Train an MLP and a CNN on both the original and permuted data. Report all four accuracies. Explain what the result proves.

**2.** Derive the output-size formula. Verify it for ten configurations against `nn.Conv2d`.

**3.** Compute by hand, then verify: parameters in a `Conv2d(64, 128, kernel_size=3, padding=1)`. Then the same for a fully connected layer between the equivalent flattened sizes on a 32×32 input.

**4.** Implement `conv2d_naive`. Verify against `torch.nn.functional.conv2d` for random inputs across several strides and paddings.

**5.** Implement `im2col` and `conv2d_fast`. Verify correctness against the naive version, then benchmark both on `(32, 3, 32, 32)` input with 64 filters. Report the speedup.

**6.** Implement `col2im` and the full backward pass for convolution. Gradient-check `dW`, `dX` and `db`. Explain why `col2im` must accumulate.

**7.** Implement `maxpool2d` forward and backward. Verify the backward routes gradient only to the argmax positions.

**8.** Compute the receptive field of every layer in a VGG-11. Write a general function that does this from a layer spec.

**9.** Verify the VGG argument numerically: compare parameters and receptive field for one 7×7 conv versus three 3×3 convs at `C=64`.

**10.** Implement a 1×1 convolution and show it's equivalent to a per-pixel linear layer across channels (reshape and use `nn.Linear`, assert equality).

**11.** Implement a depthwise separable convolution. Compare parameters and FLOPs against a standard convolution at `k=3, C=C_out=256`.

**12.** **The degradation problem.** Train plain CNNs of depth 8, 20 and 56 on CIFAR-10. Plot **training** error for all three. Reproduce the finding that deeper is worse. Then add residual connections and show it reverses.

**13.** Implement `ResidualBlock` from scratch. Build an 18-layer ResNet. Train on CIFAR-10 to >92% test accuracy.

**14.** Ablate augmentation on CIFAR-10: none, flip only, flip+crop, flip+crop+jitter. Plot train and val accuracy for each. Which gap closes most?

**15.** **Chapter project.** Take a small image dataset (a few thousand images across a handful of classes — Oxford Flowers, Food-101 subset, or your own photos). Compare three approaches: (a) a small CNN trained from scratch, (b) a frozen pretrained ResNet with a new head, (c) full fine-tuning. Report accuracy, training time, and data efficiency by also training each on 10% of the data. Write it up.

---

## 10.13 Solutions

<details>
<summary>Open only after attempting</summary>

```python
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, time


# --- 1 ---
# MLP: identical accuracy on original and permuted (a fixed permutation is just
# a relabelling of input units — the first weight matrix absorbs it exactly).
# CNN: collapses, because locality is destroyed and shared 3x3 filters no
# longer see meaningful neighbourhoods. This PROVES the MLP was using no
# spatial structure at all, and the CNN's advantage comes entirely from
# assuming it.


# --- 2, 3 ---
def out_size(i, k, p=0, s=1, d=1): return (i + 2*p - d*(k-1) - 1)//s + 1
for cfg in [(32,3,1,1),(32,3,1,2),(32,5,0,1),(28,7,3,2),(224,11,2,4)]:
    i,k,p,s = cfg
    ref = nn.Conv2d(1,1,k,s,p)(torch.zeros(1,1,i,i)).shape[-1]
    assert out_size(i,k,p,s) == ref, cfg
print("output formula verified")

print(3*3*64*128 + 128)                    # 73,856 conv params
print(64*32*32 * 128*32*32)                # ~8.6e9 for the FC equivalent


# --- 4, 5 ---
def conv2d_naive(X, W, b, stride=1, pad=0):
    N, C, H, Wd = X.shape; C_out, _, kh, kw = W.shape
    Xp = np.pad(X, ((0,0),(0,0),(pad,pad),(pad,pad)))
    Ho = (H+2*pad-kh)//stride + 1; Wo = (Wd+2*pad-kw)//stride + 1
    out = np.zeros((N, C_out, Ho, Wo))
    for n in range(N):
        for co in range(C_out):
            for i in range(Ho):
                for j in range(Wo):
                    p_ = Xp[n,:,i*stride:i*stride+kh, j*stride:j*stride+kw]
                    out[n,co,i,j] = (p_*W[co]).sum() + b[co]
    return out

def im2col(X, kh, kw, stride=1, pad=0):
    N,C,H,Wd = X.shape
    Xp = np.pad(X, ((0,0),(0,0),(pad,pad),(pad,pad)))
    Ho = (H+2*pad-kh)//stride+1; Wo = (Wd+2*pad-kw)//stride+1
    cols = np.zeros((N,C,kh,kw,Ho,Wo))
    for i in range(kh):
        for j in range(kw):
            cols[:,:,i,j,:,:] = Xp[:,:, i:i+stride*Ho:stride, j:j+stride*Wo:stride]
    return cols.transpose(0,4,5,1,2,3).reshape(N*Ho*Wo,-1), Ho, Wo

def conv2d_fast(X, W, b, stride=1, pad=0):
    N = X.shape[0]; C_out, C_in, kh, kw = W.shape
    cols, Ho, Wo = im2col(X, kh, kw, stride, pad)
    out = cols @ W.reshape(C_out,-1).T + b
    return out.reshape(N,Ho,Wo,C_out).transpose(0,3,1,2)

rng = np.random.default_rng(0)
X = rng.standard_normal((4,3,16,16)); W = rng.standard_normal((8,3,3,3)); b = rng.standard_normal(8)
ref = F.conv2d(torch.tensor(X), torch.tensor(W), torch.tensor(b), stride=1, padding=1).numpy()
assert np.allclose(conv2d_naive(X,W,b,1,1), ref, atol=1e-10)
assert np.allclose(conv2d_fast (X,W,b,1,1), ref, atol=1e-10)

Xb = rng.standard_normal((32,3,32,32)); Wb = rng.standard_normal((64,3,3,3)); bb = np.zeros(64)
t=time.perf_counter(); conv2d_naive(Xb,Wb,bb,1,1); t1=time.perf_counter()-t
t=time.perf_counter(); conv2d_fast (Xb,Wb,bb,1,1); t2=time.perf_counter()-t
print(f"naive {t1:.3f}s  im2col {t2:.4f}s  speedup {t1/t2:.0f}x")


# --- 6 ---
# col2im must ACCUMULATE (+=) into overlapping positions: with stride < k a
# single input pixel appears in several patches, so it influenced the loss
# through several routes. That is §3.5 — multiple paths, gradients add.


# --- 9 ---
C = 64
print("one 7x7 :", 7*7*C*C, "params, RF 7, 1 nonlinearity")
print("three3x3:", 3*(3*3*C*C), "params, RF 7, 3 nonlinearities")
# 200,704 vs 110,592 — 45% fewer, same receptive field, 3x the nonlinearity.


# --- 11 ---
k, C, Co = 3, 256, 256
print("standard :", k*k*C*Co)          # 589,824
print("separable:", k*k*C + C*Co)      #  68,608  -> 8.6x fewer


# --- 12: THE KEY EXPERIMENT ---
# Plain nets: train error at depth 56 > depth 20 > depth 8 is WRONG-way-round,
# and it is TRAINING error, so it is not overfitting. A 56-layer net can
# represent any 20-layer net by setting the extra layers to identity, so the
# solution exists and SGD simply fails to find it. Adding residuals reverses
# the ordering immediately: deeper becomes better again, because ∂y/∂x = 1 +
# ∂F/∂x gives gradient a path of derivative exactly 1 to every layer.


# --- 13 ---
class ResidualBlock(nn.Module):
    def __init__(self, ci, co, stride=1):
        super().__init__()
        self.c1 = nn.Conv2d(ci, co, 3, stride, 1, bias=False); self.b1 = nn.BatchNorm2d(co)
        self.c2 = nn.Conv2d(co, co, 3, 1, 1, bias=False);      self.b2 = nn.BatchNorm2d(co)
        self.sc = nn.Sequential()
        if stride != 1 or ci != co:
            self.sc = nn.Sequential(nn.Conv2d(ci, co, 1, stride, bias=False),
                                    nn.BatchNorm2d(co))
    def forward(self, x):
        o = F.relu(self.b1(self.c1(x)))
        o = self.b2(self.c2(o))
        return F.relu(o + self.sc(x))

class ResNet18(nn.Module):
    def __init__(self, n_classes=10):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3,64,3,1,1,bias=False),
                                  nn.BatchNorm2d(64), nn.ReLU())
        cfg = [(64,64,1),(64,64,1),(64,128,2),(128,128,1),
               (128,256,2),(256,256,1),(256,512,2),(512,512,1)]
        self.blocks = nn.Sequential(*[ResidualBlock(a,b,s) for a,b,s in cfg])
        self.head = nn.Linear(512, n_classes)
    def forward(self, x):
        x = self.blocks(self.stem(x))
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)     # global average pooling
        return self.head(x)

# Recipe for >92% on CIFAR-10: SGD momentum 0.9, lr 0.1, wd 5e-4,
# cosine schedule, 100 epochs, random crop + horizontal flip.
```

</details>

---

## 10.14 Chapter 10 checkpoint

Cold — blank file, no notes.

- [ ] State the three ideas behind convolution and what each one buys.
- [ ] **Write the output size formula** and apply it to five configurations. **From memory.**
- [ ] Compute the parameter count of a conv layer given `k, C_in, C_out`.
- [ ] Explain how channels work — why each filter spans all input channels.
- [ ] **Implement `conv2d_naive` from scratch.** **Target: 20 minutes.**
- [ ] Explain the im2col trick and why the backward pass comes free from it.
- [ ] Explain why three 3×3 convs beat one 7×7, with both numbers.
- [ ] **State the degradation problem and explain why it proves an optimization failure**, not overfitting.
- [ ] Write `∂y/∂x` for a residual block and explain why it fixes gradient flow.
- [ ] Explain what a 1×1 convolution does.

Item 8 is the one that matters most conceptually — the reasoning pattern there (a solution provably exists, so failure must be optimization) is how good researchers localize problems.

### Anki cards

- The three ideas behind convolution
- Output size formula
- Conv parameter count formula
- How do channels work in a conv layer?
- What is im2col and why does it help?
- Receptive field formula
- Three 3×3 vs one 7×7 — the three comparisons
- Max pool backward — where does gradient go?
- The degradation problem and what it proved
- Residual block: `∂y/∂x = ?`
- Why is learning `F(x) → 0` easier than learning the identity?
- What does a 1×1 conv do?
- Depthwise separable — cost comparison
- Two rules for data augmentation
- Transfer learning strategy by dataset size

### Deliverables

```
conv/conv2d.py        naive + im2col + full backward, gradient-checked
conv/pool.py          maxpool forward and backward
conv/receptive.py     receptive field calculator
models/resnet.py      ResidualBlock + ResNet18
experiments/permute.py       exercise 1
experiments/degradation.py   exercise 12, with plots
reports/transfer.md          exercise 15
```

```bash
git add .
git commit -m "Chapter 10: convolutions from scratch, ResNet, CIFAR-10 >92%"
git push
```

### Write-up

700 words: **"The degradation problem, and why ResNets fixed it."** Lead with your exercise 12 plots showing deeper-is-worse on *training* error, make the argument that this rules out overfitting and proves an optimization failure, then the identity-mapping argument and `∂y/∂x = 1 + ∂F/∂x`, then your reversal after adding residuals.

Most explanations say "residual connections help gradients flow." Yours will show the diagnosis that led to them — and that diagnostic move is more valuable than the fix.

**Residual connections carry straight into Chapter 12.** Every transformer block is `x + Attention(LN(x))` followed by `x + MLP(LN(x))`. You've now built the pattern that a 96-layer GPT depends on.

---

*Next: Chapter 11 — Sequences and Attention*
