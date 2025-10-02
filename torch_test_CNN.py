# torch_test_NN.py
"""
Draw all the output of NN (PyTorch checkpoint)
  1) Training/Test loss curves
  2) 10x10 confusion matrix (probability)
  3) Top-3 scored images with probability per class
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from src.dataloader import Dataloader
from src.utils import confusion_matrix, confusion_matrix_prob
from model.torch_ThreeLayerCNN import ThreeLayerCNN 

# ---------------- Paths ----------------
CKPT_DIR = "checkpoints"
OUT_DIR  = "outputs_torch_CNN"
os.makedirs(OUT_DIR, exist_ok=True)

CKPT_PATH = os.path.join(CKPT_DIR, "torch_cnn.pt")
HIST_PATH = os.path.join(CKPT_DIR, "torch_cnn_loss_history.json")

# ---------------- Data loaders ----------------
batch_size   = 128
train_loader = Dataloader("dataset", is_train=True,  shuffle=False, batch_size=batch_size)
test_loader  = Dataloader("dataset", is_train=False, shuffle=False, batch_size=batch_size)

# ---------------- Utilities ----------------
def np_batch_to_tensors(images, labels, device):
    """Convert numpy (NHW or NCHW) + labels (one-hot or ids) to torch tensors on device."""
    if isinstance(images, np.ndarray):
        x = torch.from_numpy(images).float()
    else:
        x = images.float()
    # if grayscale NHW, add channel dim -> NCHW
    if x.ndim == 3:
        x = x.unsqueeze(1)
    if isinstance(labels, np.ndarray):
        y_np = labels.argmax(axis=1) if labels.ndim == 2 else labels
        y = torch.from_numpy(y_np).long()
    else:
        y = labels.long()
    return x.to(device), y.to(device)

@torch.no_grad()
def evaluate_full(model, dataload, device):
    """Return avg_loss, acc, y_true(np), y_pred(np), probs_all(np)."""
    model.eval()
    total = 0
    total_loss = 0.0
    y_true = []
    y_pred = []
    probs_all = []

    for images, labels in dataload:
        x, t = np_batch_to_tensors(images, labels, device)
        logits = model(x)
        loss = F.cross_entropy(logits, t)
        total_loss += loss.item() * x.size(0)

        probs = F.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)
        y_pred.append(preds.cpu().numpy())
        if isinstance(labels, np.ndarray):
            t_ids = labels.argmax(axis=1) if labels.ndim == 2 else labels
        else:
            t_ids = t.cpu().numpy()
        y_true.append(np.asarray(t_ids))
        probs_all.append(probs.cpu().numpy())

        total += x.size(0)

    y_true   = np.concatenate(y_true)
    y_pred   = np.concatenate(y_pred)
    probs_all = np.concatenate(probs_all)
    avg_loss = total_loss / total
    acc      = float((y_true == y_pred).mean())
    return avg_loss, acc, y_true, y_pred, probs_all

# ---------------- Load model ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ThreeLayerCNN().to(device)
state = torch.load(CKPT_PATH, map_location=device)
model.load_state_dict(state)

# ---------------- 1) Loss curves (train/test) ----------------
if not os.path.exists(HIST_PATH):
    raise FileNotFoundError(f"Missing {HIST_PATH}. Re-run torch training to save loss history.")
with open(HIST_PATH, "r") as f:
    hist = json.load(f)
train_losses = hist.get("train_losses", [])
test_losses  = hist.get("test_losses", [])

plt.figure()
plt.plot(range(1, len(train_losses)+1), train_losses, label="train")
plt.plot(range(1, len(test_losses)+1),  test_losses,  label="test")
plt.xlabel("Epoch")
plt.ylabel("Cross-Entropy Loss")
plt.title("Torch NN Loss (Train/Test)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "loss_curves.png"), dpi=150)
plt.close()

# ---------------- 2) Confusion matrix on test set ----------------
_, test_acc, y_true, y_pred, probs_all = evaluate_full(model, test_loader, device)
print(f"Test Accuracy: {test_acc:.4f}")
cm_counts = confusion_matrix(y_true, y_pred, num_classes=10)
cm_prob   = confusion_matrix_prob(cm_counts)

# np.savetxt(os.path.join(OUT_DIR, "confusion_matrix_counts.csv"), cm_counts, fmt="%d", delimiter=",")
# np.savetxt(os.path.join(OUT_DIR, "confusion_matrix_prob.csv"),   cm_prob,   fmt="%.4f", delimiter=",")

plt.figure(figsize=(6,5))
plt.imshow(cm_prob, vmin=0.0, vmax=1.0, interpolation="nearest", cmap="Blues")
plt.title("Torch CNN Confusion Matrix (Probability)")
plt.xticks(np.arange(10), [str(i) for i in range(10)])
plt.yticks(np.arange(10), [str(i) for i in range(10)])
cb = plt.colorbar()
cb.set_label("P(pred | true)")
for i in range(cm_prob.shape[0]):
    for j in range(cm_prob.shape[1]):
        plt.text(j, i, f"{cm_prob[i, j]:.3f}", ha="center", va="center", fontsize=6)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "confusion_matrix_prob.png"), dpi=150)
plt.close()

# ---------------- 3) Top-3 images with probabilities (per class) ----------------
# 수집(원본 이미지는 numpy로 묶어둠)
imgs_list, labels_list = [], []
for images, labels in test_loader:
    imgs_list.append(images)
    labels_list.append(labels)

imgs   = np.concatenate(imgs_list, axis=0)
labels = np.concatenate(labels_list, axis=0)

if labels.ndim == 2:
    labels_ids = np.argmax(labels, axis=1)
else:
    labels_ids = labels

summary_lines = []
for c in range(10):
    idx_c = np.where(labels_ids == c)[0]
    if len(idx_c) == 0:
        summary_lines.append(f"{c}: (no samples)")
        continue
    class_probs = probs_all[idx_c, c]
    topk_idx = np.argsort(-class_probs)[:3]

    probs_pct = [f"{p*100:.1f}%" for p in class_probs[topk_idx]]
    summary_lines.append(f"{c}: " + ", ".join(probs_pct))

    chosen = idx_c[topk_idx]
    chosen_probs = class_probs[topk_idx]

    fig = plt.figure(figsize=(9, 3))
    for i_img, (idx_img, p) in enumerate(zip(chosen, chosen_probs), start=1):
        ax = fig.add_subplot(1, 3, i_img)
        img = imgs[idx_img]
        if img.ndim == 3:
            if img.shape[0] == 1:
                img = img[0]
        ax.imshow(img.reshape(28, 28), cmap="gray")
        ax.axis("off")
        ax.set_title(f"{p*100:.1f}%", fontsize=10)
    plt.suptitle(f"Class {c}: Top-3 images")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(OUT_DIR, f"top3_class_{c}.png"), dpi=150)
    plt.close()

with open(os.path.join(OUT_DIR, "top3_summary.txt"), "w") as f:
    f.write("\n".join(summary_lines))