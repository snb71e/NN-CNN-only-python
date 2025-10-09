import os
import json
import numpy as np
import matplotlib.pyplot as plt
from src.dataloader import Dataloader
from src.utils import load_model_cnn, confusion_matrix, confusion_matrix_prob

CKPT_DIR = "checkpoints" 
OUT_DIR  = "epoch_5_outputs_CNN" 
os.makedirs(OUT_DIR, exist_ok=True)

model = load_model_cnn(os.path.join(CKPT_DIR, 'epoch_5_cnn.npz'))
hist_path = os.path.join(CKPT_DIR, 'epoch_5_cnn_loss_history.json')

batch_size = 128
train_loader = Dataloader("dataset", is_train=True,  shuffle=False, batch_size=batch_size)
test_loader  = Dataloader("dataset", is_train=False, shuffle=False, batch_size=batch_size)

def evaluate_full(model, dataload):
    total = 0
    total_loss = 0.0
    y_true = []
    y_pred = []
    probs_all = []
    for images, labels in dataload:
        N = images.shape[0]
        x = images
        t = labels
        loss  = model.forward(x, t)
        total_loss += float(loss) * N
        probs = model.forward(x, t=None)
        preds = np.argmax(probs, axis=1)
        t_ids = np.argmax(t, axis=1) if t.ndim == 2 else t
        y_true.append(t_ids)
        y_pred.append(preds)
        probs_all.append(probs)
        total += N
    y_true   = np.concatenate(y_true)
    y_pred   = np.concatenate(y_pred)
    probs_all = np.concatenate(probs_all)
    avg_loss = total_loss / total
    acc      = float((y_true == y_pred).mean())
    return avg_loss, acc, y_true, y_pred, probs_all

# Loss Curve
if not os.path.exists(hist_path):
    raise FileNotFoundError(f"Missing {hist_path}. Re-run training so it saves loss history.")
with open(hist_path, 'r') as f:
    hist = json.load(f)
train_losses = hist.get('train_losses', [])
test_losses  = hist.get('test_losses',  [])

plt.figure()
plt.plot(range(1, len(train_losses)+1), train_losses, label='train')
plt.plot(range(1, len(test_losses)+1),  test_losses,  label='test')
plt.xlabel('Epoch')
plt.ylabel('Cross-Entropy Loss')
plt.title('CNN Loss (Train/Test)')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'loss_curves.png'), dpi=150)
plt.close()

# Confusion matrix

train_loss, train_acc, _, _, _ = evaluate_full(model, train_loader)
test_loss, test_acc, y_true, y_pred, probs_all = evaluate_full(model, test_loader)
print(f"Final Results:")
print(f"  Train Loss: {train_loss:.4f}")
print(f"  Train Acc: {train_acc:.4f}")
print(f"  Test  Loss: {test_loss:.4f}")
print(f"  Test  Acc : {test_acc:.4f}")


cm_counts = confusion_matrix(y_true, y_pred, num_classes=10)
cm_prob   = confusion_matrix_prob(cm_counts)

plt.figure(figsize=(6,5))
plt.imshow(cm_prob, vmin=0.0, vmax=1.0, interpolation='nearest', cmap='Blues')
plt.title('CNN Confusion Matrix')
plt.xticks(np.arange(10), [str(i) for i in range(10)])
plt.yticks(np.arange(10), [str(i) for i in range(10)])
cb = plt.colorbar()
cb.set_label('P(pred | true)')
for i in range(cm_prob.shape[0]):
    for j in range(cm_prob.shape[1]):
        plt.text(j, i, f"{cm_prob[i, j]:.3f}", ha='center', va='center', fontsize=6)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'confusion_matrix_prob.png'), dpi=150)
plt.close()

# Top-3 images with probabilities
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
    summary_lines.append(f"{c}: "+", ".join(probs_pct))
    chosen   = idx_c[topk_idx]
    chosen_probs = class_probs[topk_idx]

    fig = plt.figure(figsize=(9, 3))
    for i_img, (idx_img, p) in enumerate(zip(chosen, chosen_probs), start=1):
        ax = fig.add_subplot(1, 3, i_img)
        img = imgs[idx_img]
        if img.ndim == 3 and img.shape[0] == 1:
            img = img[0]
        ax.imshow(img.reshape(28,28), cmap='gray')
        ax.axis('off')
        ax.set_title(f'{p*100:.1f}%', fontsize=10)
    plt.suptitle(f'Class {c}: Top-3 images')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(OUT_DIR, f'top3_class_{c}.png'), dpi=150)
    plt.close()

with open(os.path.join(OUT_DIR, 'top3_summary.txt'), 'w') as f:
    f.write("\n".join(summary_lines))
