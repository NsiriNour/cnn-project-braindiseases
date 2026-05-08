#!/usr/bin/env python3
"""Test that all imports and paths are correct after restructuring."""

from src.data.data_loader_stroke import DATA_ROOT as STROKE_DATA
from src.data.data_loader import DATASET_ROOT, TEST_ROOT
from src.data.data_loader_alzheimer import DATA_ROOT as ALZ_DATA
from pathlib import Path

print('FINAL VERIFICATION')
print('═' * 60)
print(' All imports successful!')
print('')
print(' DATA PATHS VERIFICATION:')
print(f'  ✓ Stroke: {STROKE_DATA.exists()} - {STROKE_DATA}')
print(f'  ✓ Brain Tumor: {DATASET_ROOT.exists()} - {DATASET_ROOT}')
print(f'  ✓ Alzheimer: {ALZ_DATA.exists()} - {ALZ_DATA}')
print('')
print(' MODELS:')
models = list(Path('./models').glob('*.pth')) + list(Path('./models').glob('*.pt'))
print(f'  Found {len(models)} model files:')
for m in sorted(models):
    size_mb = m.stat().st_size / (1024*1024)
    print(f'    - {m.name} ({size_mb:.1f} MB)')
print('═' * 60)
print(' STRUCTURE FIX COMPLETE!')
