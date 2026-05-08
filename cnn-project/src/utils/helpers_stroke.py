# ─── Imports ────────────────────────────────────────────
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.cm as colormap
from sklearn.metrics import (classification_report, confusion_matrix,
                             ConfusionMatrixDisplay, accuracy_score,
                             roc_auc_score)

from src.data.data_loader import IMAGENET_MEAN, IMAGENET_STD


# ─── Courbes d'entraînement ──────────────────────────────
def plot_history(history, title):
    epochs = range(1, len(history['train_loss']) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    ax1.plot(epochs, history['train_loss'], 'b-o', label='Train', markersize=4)
    ax1.plot(epochs, history['val_loss'],   'r-o', label='Val',   markersize=4)
    ax1.set_title('Loss (lower = better)')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy
    ax2.plot(epochs, history['train_acc'], 'b-o', label='Train', markersize=4)
    ax2.plot(epochs, history['val_acc'],   'r-o', label='Val',   markersize=4)
    ax2.set_title('Accuracy (higher = better)')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_ylim([0, 1])
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{title.replace(" ", "_")}.png')
    plt.show()


# ─── Matrice de confusion ────────────────────────────────
def plot_confusion_matrix(y_true, y_pred, class_names, title="Confusion Matrix"):
    cm   = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, colorbar=False, cmap='Blues')
    ax.set_title(title, fontweight='bold')
    plt.tight_layout()
    plt.show()


# ─── Métriques ───────────────────────────────────────────
def print_metrics(y_true, y_pred, class_names, title="Résultats"):
    acc = accuracy_score(y_true, y_pred)
    print("=" * 55)
    print(f"  {title}")
    print("=" * 55)
    print(f"  Accuracy : {acc*100:.2f}%")
    if len(class_names) == 2:
        auc = roc_auc_score(y_true, y_pred)
        print(f"  AUC Score : {auc:.4f}")
    print()
    print(classification_report(y_true, y_pred, target_names=class_names))


# ─── Evaluation ──────────────────────────────────────────
def evaluate_model(model, loader, class_names, model_name, DEVICE):
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

    print_metrics(all_labels, all_preds, class_names, title=model_name)
    plot_confusion_matrix(all_labels, all_preds, class_names,
                          title=f'Confusion Matrix — {model_name}')
    return np.array(all_labels), np.array(all_preds)


# ─── GradCAM visualisation ───────────────────────────────
def show_gradcam_grid(model, target_layer, loader, class_names,
                      title, device, GradCAM, n=4):
    from src.model.model import GradCAM

    gc  = GradCAM(model, target_layer)
    fig, axes = plt.subplots(2, n, figsize=(4 * n, 9))
    shown = 0

    for images, labels in loader:
        for i in range(images.size(0)):
            if shown >= n:
                break

            cam, pred = gc.generate(images[i].unsqueeze(0), device)

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

    axes[0, 0].set_ylabel('Image originale', fontsize=11)
    axes[1, 0].set_ylabel('Grad-CAM', fontsize=11)
    plt.suptitle(title, fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show()


# ─── Prédiction single image ─────────────────────────────
def predict_single(image_path, model_a, model_b, val_transform,
                   DEVICE, BINARY_CLASSES, STROKE_CLASSES,
                   CONFIDENCE_THRESHOLD=0.85, threshold=0.5):
    from PIL import Image

    image  = Image.open(image_path).convert('RGB')
    tensor = val_transform(image).unsqueeze(0).to(DEVICE)

    # ── Stage 1 : Model A ────────────────────────────────
    with torch.no_grad():
        logits_a = model_a(tensor)
        probs_a  = torch.softmax(logits_a, dim=1)[0]

    conf_normal = probs_a[0].item()
    conf_stroke = probs_a[1].item()
    max_conf    = max(conf_normal, conf_stroke)

    # Low confidence
    if max_conf < CONFIDENCE_THRESHOLD:
        return {
            'status'    : 'low_confidence',
            'type'      : None,
            'conf_a'    : max_conf,
            'conf_b'    : None,
            'message'   : f'⚠️ Confiance faible ({max_conf:.1%}) — consulter un radiologue'
        }

    # Normal
    if conf_stroke <= threshold:
        return {
            'status'    : 'normal',
            'type'      : None,
            'conf_a'    : conf_normal,
            'conf_b'    : None,
            'message'   : f'✅ Cerveau Normal (confiance: {conf_normal:.1%})'
        }

    # ── Stage 2 : Model B ────────────────────────────────
    with torch.no_grad():
        logits_b = model_b(tensor)
        probs_b  = torch.softmax(logits_b, dim=1)[0]

    pred_b     = int(probs_b.argmax())
    conf_b     = probs_b[pred_b].item()
    stroke_type = STROKE_CLASSES[pred_b]

    return {
        'status'    : 'stroke',
        'type'      : stroke_type,
        'conf_a'    : conf_stroke,
        'conf_b'    : conf_b,
        'message'   : (f'🚨 AVC détecté : {stroke_type} '
                       f'(détection: {conf_stroke:.1%}, '
                       f'classification: {conf_b:.1%})')
    }