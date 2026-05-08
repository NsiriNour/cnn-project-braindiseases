"""Train brain disease models from JSON configuration files.

Usage:
    python train.py Alzheimer
    python train.py Stroke
    python train.py Brain_Tumor

This script does not modify model architectures or training logic.
It only orchestrates configuration, result folder creation, and automatic output saving.
"""

import argparse
from pathlib import Path
import numpy as np
import torch
from torch import nn

from src.config import get_config, get_paths
from src.utils.output_manager import get_output_manager

from src.data.data_loader_alzheimer import get_dataloaders as get_alzheimer_loaders
from src.model.model_alzheimer import build_efficientnet_b0
from src.model.train_utils_alzheimer import train_model as train_alzheimer, evaluate as evaluate_alzheimer

import src.data.data_loader_stroke as stroke_data
from src.data.data_loader_stroke import get_loaders as get_stroke_loaders
from src.model.model_stroke import build_model_a as build_stroke_model_a, build_model_b as build_stroke_model_b
from src.model.train_utils_stroke import train_model as train_stroke, set_seed as set_stroke_seed

import src.data.data_loader as tumor_data
from src.data.data_loader import get_loaders as get_tumor_loaders
from src.model.model import build_model_a as build_tumor_model_a, build_model_b as build_tumor_model_b, count_params
from src.model.train_utils import train_model as train_tumor


def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = Path.cwd() / path
    return path


def ensure_dirs(paths):
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)


def predict_labels(model, loader, device):
    model.eval()
    labels = []
    preds = []
    with torch.no_grad():
        for imgs, labs in loader:
            imgs = imgs.to(device)
            out = model(imgs)
            pred = out.argmax(dim=1).cpu().numpy()
            preds.extend(pred.tolist())
            labels.extend(labs.cpu().numpy().tolist())
    return np.array(labels), np.array(preds)


def train_alzheimer_pipeline(config, paths, manager):
    data_root = resolve_path(config['paths']['data_root'])
    train_loader, val_loader, test_loader, class_names = get_alzheimer_loaders(
        data_root=str(data_root),
        img_size=int(config['data']['img_size']),
        batch_size=int(config['data']['batch_size']),
        val_split=float(config['data']['val_split']),
        num_workers=int(config['data'].get('num_workers', 2)),
        seed=int(config['data'].get('seed', 42)),
    )

    model = build_efficientnet_b0(
        num_classes=int(config['model']['num_classes']),
        dropout=float(config['model']['dropout'])
    )

    history, best_model = train_alzheimer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=torch.device(config.get('device', 'cpu')),
        epochs=int(config['model']['epochs']),
        lr=float(config['model']['learning_rate']),
        weight_decay=float(config['model'].get('weight_decay', 1e-4)),
        patience=int(config['model'].get('patience', 7)),
    )

    manager.save_model_weights(best_model)
    manager.save_training_history(history)
    manager.save_training_curves(history, 'Alzheimer MRI — EfficientNet-B0')

    criterion = nn.CrossEntropyLoss()
    loss, acc, preds, labels, probs, macro_f1 = evaluate_alzheimer(
        best_model, test_loader, criterion, torch.device(config.get('device', 'cpu'))
    )

    manager.save_confusion_matrix(labels, preds, class_names,
                                  title='Alzheimer Confusion Matrix')
    manager.save_metrics_report({
        'loss': float(loss),
        'accuracy': float(acc),
        'macro_f1': float(macro_f1),
        'classes': class_names
    })


def train_stroke_pipeline(config, paths, manager):
    stroke_data.DATA_ROOT = resolve_path(config['paths']['data_root'])
    stroke_data.BATCH_SIZE = int(config['data']['batch_size'])
    stroke_data.VAL_SPLIT = float(config['data']['val_split'])

    set_stroke_seed(int(config['data'].get('seed', 42)))
    device = torch.device(config.get('device', 'cpu'))

    for model_type, build_fn in [('a', build_stroke_model_a), ('b', build_stroke_model_b)]:
        model_config = config[f'model_{model_type}']
        model = build_fn()

        train_loader, val_loader, test_loader = get_stroke_loaders(
            'binary' if model_type == 'a' else 'multiclass'
        )

        best_model, history = train_stroke(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=int(model_config['epochs']),
            save_path=str(manager.paths['weights'] / (config['paths'][f'weights_{model_type}'].split('/')[-1])),
            model_name=model_config['name'],
            DEVICE=device,
            LR=float(model_config['learning_rate']),
            PATIENCE=int(model_config['patience']),
        )

        manager.save_training_history(history, model_type=model_type)
        manager.save_training_curves(history, model_config['name'], model_type=model_type)

        labels, preds = predict_labels(best_model, test_loader, device)
        manager.save_confusion_matrix(labels, preds,
                                      config['model_a']['classes'] if model_type == 'a' else config['model_b']['classes'],
                                      title=model_config['name'],
                                      model_type=model_type)
        manager.save_metrics_report({
            'model_name': model_config['name'],
            'accuracy': float((labels == preds).mean()),
            'classes': config['model_a']['classes'] if model_type == 'a' else config['model_b']['classes']
        }, model_type=model_type)


def train_brain_tumor_pipeline(config, paths, manager):
    tumor_data.DATASET_ROOT = resolve_path(config['paths']['data_root'])
    tumor_data.TEST_ROOT = resolve_path(config['paths']['test_root'])
    tumor_data.BATCH_SIZE = int(config['data']['batch_size'])
    tumor_data.VAL_SPLIT = float(config['data']['val_split'])

    set_stroke_seed(int(config['data'].get('seed', 42)))
    device = torch.device(config.get('device', 'cpu'))

    for model_type, build_fn in [('a', build_tumor_model_a), ('b', build_tumor_model_b)]:
        model_config = config[f'model_{model_type}']
        model = build_fn()

        train_loader, val_loader, test_loader = get_tumor_loaders(
            'binary' if model_type == 'a' else '3class'
        )

        best_model, history = train_tumor(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=int(model_config['epochs']),
            lr=float(model_config['learning_rate']),
            weight_decay=float(model_config.get('weight_decay', 1e-4)),
            patience=int(model_config.get('patience', 5)),
        )

        manager.save_training_history(history, model_type=model_type)
        manager.save_training_curves(history, model_config['name'], model_type=model_type)

        labels, preds = predict_labels(best_model, test_loader, device)
        manager.save_confusion_matrix(labels, preds,
                                      config['model_a']['classes'] if model_type == 'a' else config['model_b']['classes'],
                                      title=model_config['name'],
                                      model_type=model_type)
        manager.save_metrics_report({
            'model_name': model_config['name'],
            'accuracy': float((labels == preds).mean()),
            'classes': config['model_a']['classes'] if model_type == 'a' else config['model_b']['classes']
        }, model_type=model_type)


def main():
    parser = argparse.ArgumentParser(description='Train brain disease models from JSON configs')
    parser.add_argument('disease', choices=['Alzheimer', 'Stroke', 'Brain_Tumor'], help='Disease model to train')
    args = parser.parse_args()

    config = get_config(args.disease)
    paths = get_paths(args.disease)
    manager = get_output_manager(args.disease)

    print(f"Training {args.disease} with config: {paths['config']}")
    manager.print_structure()

    if args.disease == 'Alzheimer':
        train_alzheimer_pipeline(config, paths, manager)
    elif args.disease == 'Stroke':
        train_stroke_pipeline(config, paths, manager)
    elif args.disease == 'Brain_Tumor':
        train_brain_tumor_pipeline(config, paths, manager)

    print('\nDone. All outputs were saved inside the results directory.')


if __name__ == '__main__':
    main()
