# ─── Imports ────────────────────────────────────────────
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.cm as colormap
from sklearn.metrics import (classification_report, confusion_matrix,
                             ConfusionMatrixDisplay, accuracy_score, roc_auc_score)

from src.data.data_loader import IMAGENET_MEAN, IMAGENET_STD


# ─── Visualisation courbes ───────────────────────────────
def plot_history(history, title):
    epochs = range(1, len(history['train_loss']) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, history['train_loss'], 'b-o', label='Train loss', markersize=4)
    ax1.plot(epochs, history['val_loss'],   'r-o', label='Val loss',   markersize=4)
    ax1.set_title('Loss (lower = better)')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history['train_acc'], 'b-o', label='Train accuracy', markersize=4)
    ax2.plot(epochs, history['val_acc'],   'r-o', label='Val accuracy',   markersize=4)
    ax2.set_title('Accuracy (higher = better)')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_ylim([0, 1])
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=13)
    plt.tight_layout()
    plt.show()


def plot_curves(loss_hist, metric_hist, title="Model A — From Scratch"):
    epochs = range(1, len(loss_hist["train"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title, fontsize=14, fontweight='bold')

    ax1.plot(epochs, loss_hist["train"], label="Train Loss", color="steelblue")
    ax1.plot(epochs, loss_hist["val"],   label="Val Loss",   color="tomato")
    ax1.set_title("Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(epochs, metric_hist["train"], label="Train Acc", color="steelblue")
    ax2.plot(epochs, metric_hist["val"],   label="Val Acc",   color="tomato")
    ax2.set_title("Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()


# ─── Confusion matrix ────────────────────────────────────
def plot_confusion_matrix(y_true, y_pred, class_names, title="Confusion Matrix"):
    cm   = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(title, fontweight='bold')
    plt.tight_layout()
    plt.show()


# ─── Métriques ───────────────────────────────────────────
def print_metrics(y_true, y_pred, class_names, title="Evaluation Metrics"):
    acc = accuracy_score(y_true, y_pred)
    print("=" * 55)
    print(f"  {title}")
    print("=" * 55)
    print(f"  Overall Accuracy : {acc*100:.2f}%")
    if len(class_names) == 2:
        auc = roc_auc_score(y_true, y_pred)
        print(f"  AUC Score        : {auc:.4f}")
    print()
    print(classification_report(y_true, y_pred, target_names=class_names))


# ─── Evaluation transfer learning ────────────────────────
def evaluate_model(model, loader, class_names, model_name, DEVICE, MODEL_A_PATH, MODEL_B_PATH):
    path = MODEL_A_PATH if 'A' in model_name else MODEL_B_PATH
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()

    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            logits = model(images)
            preds  = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    print(f'\n{"="*55}')
    print(f'  {model_name}  — Test Results')
    print(f'{"="*55}')
    print(classification_report(all_labels, all_preds, target_names=class_names))

    cm   = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, colorbar=False, cmap='Blues')
    ax.set_title(f'Confusion Matrix — {model_name}')
    plt.tight_layout()
    plt.show()


# ─── Evaluation scratch ───────────────────────────────────
def evaluate_model_scratch(model, loader, class_names, model_name, device, threshold=0.4):
    model.load_state_dict(torch.load("weights_scratch_a.pt", map_location=device))
    model = model.to(device)
    model.eval()

    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            probs  = torch.exp(model(images))
            preds  = (probs[:, 1] > threshold).long().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    print(f'\n{"="*55}')
    print(f'  {model_name} — Threshold={threshold}')
    print(f'{"="*55}')
    print(classification_report(all_labels, all_preds, target_names=class_names))

    cm   = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, colorbar=False, cmap='Blues')
    ax.set_title(f'Confusion Matrix — {model_name}')
    plt.tight_layout()
    plt.show()


# ─── GradCAM visualisation ───────────────────────────────
def show_gradcam_grid(model, target_layer, loader, class_names,
                      title, device, GradCAM, n=4, threshold=0.4, is_binary=True):
    from src.model.model import GradCAM
    gc  = GradCAM(model, target_layer)
    fig, axes = plt.subplots(2, n, figsize=(4 * n, 9))
    shown = 0

    for images, labels in loader:
        for i in range(images.size(0)):
            if shown >= n:
                break
            cam, pred = gc.generate(images[i].unsqueeze(0), device,
                                    threshold=threshold, is_binary=is_binary)

            cam_up = F.interpolate(
                torch.tensor(cam).unsqueeze(0).unsqueeze(0),
                size=(224, 224), mode='bilinear', align_corners=False
            ).squeeze().numpy()

            img_np = images[i].permute(1, 2, 0).numpy()
            img_np = (img_np * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN)).clip(0, 1)

            heat    = colormap.jet(cam_up)[:, :, :3]
            overlay = (0.55 * img_np + 0.45 * heat).clip(0, 1)

            axes[0, shown].imshow(img_np, cmap='gray')
            axes[0, shown].set_title(f'True: {class_names[labels[i]]}', fontsize=10)
            axes[0, shown].axis('off')
            axes[1, shown].imshow(overlay)
            axes[1, shown].set_title(f'Pred: {class_names[pred]}', fontsize=10)
            axes[1, shown].axis('off')
            shown += 1
        if shown >= n:
            break

    axes[0, 0].set_ylabel('Original MRI', fontsize=11)
    axes[1, 0].set_ylabel('Grad-CAM heatmap', fontsize=11)
    plt.suptitle(title, fontsize=13)
    plt.tight_layout()
    plt.show()