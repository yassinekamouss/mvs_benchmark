"""
Génère le notebook Jupyter 'notebooks/remote_gpu_worker.ipynb' prêt à être ouvert dans Google Colab
pour exécuter VGGT (CVPR 2025) et COLMAP (SfM + PatchMatch MVS) sur GPU NVIDIA T4/L4.
"""

import json
from pathlib import Path

NOTEBOOK_CONTENT = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "colab": {"provenance": [], "gpuType": "T4"},
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "accelerator": "GPU",
    },
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# 🚀 Worker GPU Distant : Reconstruction 3D Dense (VGGT vs COLMAP)\n",
                "**Projet Vision par Ordinateur — Groupe G11 (Yahya AHMANE & Yassine KAMOUSS)**\n\n",
                "Ce notebook exécute sur GPU NVIDIA (Google Colab) :\n",
                "1. Le téléchargement et l'échantillonnage d'une scène **Tanks and Temples** (`Ignatius`, `Truck`, `Barn`).\n",
                "2. La reconstruction dense *feed-forward* via **VGGT (`facebookresearch/vggt`)**.\n",
                "3. La reconstruction dense géométrique via **COLMAP (`SfM + PatchMatch MVS`)**.\n",
                "4. L'exportation d'une archive `<SCENE>_bundle.zip` à importer dans l'application locale via :\n",
                "   `python scripts/setup_data.py --import-bundle Ignatius_bundle.zip --scene Ignatius`"
            ],
        },
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": [
                "# 1. Vérification du GPU et installation des dépendances système et Python\n",
                "!nvidia-smi\n",
                "!apt-get update -qq && apt-get install -y -qq colmap xvfb\n",
                "!pip install -q open3d plyfile trimesh tqdm huggingface_hub\n",
                "!git clone --depth 1 https://github.com/facebookresearch/vggt.git /content/vggt\n",
                "!pip install -q -e /content/vggt"
            ],
        },
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": [
                "# 2. Configuration de la scène et préparation des dossiers\n",
                "import os, time, json, shutil, zipfile, glob\n",
                "from pathlib import Path\n",
                "import numpy as np\n",
                "import torch\n",
                "import open3d as o3d\n\n",
                "SCENE_NAME = 'Ignatius'  # Choix : 'Ignatius', 'Truck', 'Barn'\n",
                "NUM_VIEWS = 30           # Nombre de vues échantillonnées uniformément\n",
                "WORK_DIR = Path(f'/content/workspace/{SCENE_NAME}')\n",
                "IMG_DIR = WORK_DIR / 'images'\n",
                "OUT_DIR = WORK_DIR / 'outputs'\n",
                "IMG_DIR.mkdir(parents=True, exist_ok=True)\n",
                "OUT_DIR.mkdir(parents=True, exist_ok=True)\n",
                "print(f'Workspace initialisé dans {WORK_DIR} sur {torch.cuda.get_device_name(0)}')"
            ],
        },
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": [
                "# 3. Inférence VGGT (Visual Geometry Grounded Transformer - CVPR 2025)\n",
                "import sys\n",
                "sys.path.append('/content/vggt')\n",
                "from vggt.models.vggt import VGGT\n",
                "from vggt.utils.load_fn import load_and_preprocess_images\n\n",
                "image_paths = sorted( [str(p) for p in IMG_DIR.glob('*.jpg')] + [str(p) for p in IMG_DIR.glob('*.png')] )\n",
                "if len(image_paths) > NUM_VIEWS:\n",
                "    idx = np.linspace(0, len(image_paths) - 1, NUM_VIEWS, dtype=int)\n",
                "    image_paths = [image_paths[i] for i in idx]\n\n",
                "device = 'cuda' if torch.cuda.is_available() else 'cpu'\n",
                "dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16\n\n",
                "print(f'[*] Chargement du modèle VGGT-1B sur {device}...')\n",
                "model = VGGT.from_pretrained('facebook/VGGT-1B').to(device)\n",
                "images = load_and_preprocess_images(image_paths).to(device)\n\n",
                "t0 = time.time()\n",
                "with torch.no_grad():\n",
                "    with torch.cuda.amp.autocast(dtype=dtype):\n",
                "        predictions = model(images)\n",
                "torch.cuda.synchronize()\n",
                "vggt_time = round(time.time() - t0, 2)\n\n",
                "# Extraction des cartes de points 3D (world_points) et filtrage par confiance\n",
                "pts = predictions['world_points'].squeeze(0).reshape(-1, 3).cpu().numpy()\n",
                "conf = predictions['world_points_conf'].squeeze(0).reshape(-1).cpu().numpy()\n",
                "colors = images.squeeze(0).permute(0, 2, 3, 1).reshape(-1, 3).cpu().numpy()\n\n",
                "mask = conf > np.percentile(conf, 25)\n",
                "pcd_vggt = o3d.geometry.PointCloud()\n",
                "pcd_vggt.points = o3d.utility.Vector3dVector(pts[mask])\n",
                "pcd_vggt.colors = o3d.utility.Vector3dVector(colors[mask])\n",
                "pcd_vggt = pcd_vggt.voxel_down_sample(voxel_size=0.005)\n",
                "o3d.io.write_point_cloud(str(OUT_DIR / 'vggt_dense.ply'), pcd_vggt)\n",
                "print(f'[OK] VGGT terminé en {vggt_time}s ({len(pcd_vggt.points):,} points)')"
            ],
        },
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": [
                "# 4. Pipeline Géométrique Classique COLMAP (SfM + PatchMatch MVS Dense)\n",
                "COLMAP_WS = WORK_DIR / 'colmap_ws'\n",
                "SPARSE_DIR = COLMAP_WS / 'sparse'\n",
                "DENSE_DIR = COLMAP_WS / 'dense'\n",
                "SPARSE_DIR.mkdir(parents=True, exist_ok=True)\n",
                "DENSE_DIR.mkdir(parents=True, exist_ok=True)\n",
                "DB_PATH = COLMAP_WS / 'database.db'\n\n",
                "t0 = time.time()\n",
                "!colmap feature_extractor --database_path {DB_PATH} --image_path {IMG_DIR} --SiftExtraction.use_gpu 1\n",
                "!colmap exhaustive_matcher --database_path {DB_PATH} --SiftMatching.use_gpu 1\n",
                "!colmap mapper --database_path {DB_PATH} --image_path {IMG_DIR} --output_path {SPARSE_DIR}\n",
                "!colmap image_undistorter --image_path {IMG_DIR} --input_path {SPARSE_DIR}/0 --output_path {DENSE_DIR} --output_type COLMAP --max_image_size 1600\n",
                "!colmap patch_match_stereo --workspace_path {DENSE_DIR} --workspace_format COLMAP --PatchMatchStereo.geom_consistency true\n",
                "!colmap stereo_fusion --workspace_path {DENSE_DIR} --workspace_format COLMAP --input_type geometric --output_path {OUT_DIR}/colmap_dense.ply\n",
                "colmap_time = round(time.time() - t0, 2)\n",
                "print(f'[OK] COLMAP MVS terminé en {colmap_time}s')"
            ],
        },
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": [
                "# 5. Création de l'archive Bundle (.zip) pour l'application locale\n",
                "bundle_path = f'/content/{SCENE_NAME}_bundle.zip'\n",
                "with zipfile.ZipFile(bundle_path, 'w', zipfile.ZIP_DEFLATED) as zf:\n",
                "    for img_p in image_paths:\n",
                "        zf.write(img_p, arcname=f'images/{Path(img_p).name}')\n",
                "    for ply_name in ['vggt_dense.ply', 'colmap_dense.ply', 'timings.json']:\n",
                "        p = OUT_DIR / ply_name\n",
                "        if p.exists():\n",
                "            zf.write(p, arcname=ply_name)\n",
                "print(f'Bundle prêt à télécharger : {bundle_path}')"
            ],
        },
    ],
}

if __name__ == "__main__":
    out_path = Path(__file__).resolve().parent.parent / "notebooks" / "remote_gpu_worker.ipynb"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(NOTEBOOK_CONTENT, f, indent=2, ensure_ascii=False)
    print(f"[OK] Notebook GPU généré : {out_path}")
