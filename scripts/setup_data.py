"""
Script de préparation, de vérification des datasets et de génération de données de test calibrées.
Usage :
    python scripts/setup_data.py --generate-mock
    python scripts/setup_data.py --check
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import open3d as o3d
from PIL import Image, ImageDraw

# Ajout de la racine du projet au PYTHONPATH
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.loader import SceneDataLoader


def generate_synthetic_benchmark(loader: SceneDataLoader, scene_name: str = "Ignatius") -> None:
    """
    Génère un jeu de données synthétique calibré (Images + Ground Truth + Nuages VGGT/COLMAP simulés)
    pour tester de bout en bout le pipeline d'alignement (Sim3/ICP) et les métriques (Chamfer/F-Score)
    avant ou pendant le téléchargement des archives lourdes de Tanks and Temples.
    """
    print(f"[*] Génération des données de calibration pour la scène '{scene_name}'...")
    paths = loader.get_scene_paths(scene_name)
    paths["images_dir"].mkdir(parents=True, exist_ok=True)
    paths["output_dir"].mkdir(parents=True, exist_ok=True)

    # 1. Génération de 20 vues synthétiques numérotées (simulation de caméras à 360°)
    for i in range(1, 21):
        img = Image.new("RGB", (640, 360), color=(25, 32, 48))
        draw = ImageDraw.Draw(img)
        angle = int((i - 1) * (360 / 20))
        # Dessin d'une mire géométrique simulant le changement de point de vue
        cx = 320 + int(80 * np.cos(np.radians(angle)))
        cy = 180 + int(40 * np.sin(np.radians(angle)))
        draw.ellipse([cx - 60, cy - 60, cx + 60, cy + 60], fill=(70, 140, 220), outline=(255, 255, 255), width=3)
        draw.text((20, 20), f"Scene: {scene_name} | View #{i:03d} | Angle: {angle} deg", fill=(230, 230, 230))
        img.save(paths["images_dir"] / f"frame_{i:04d}.jpg", quality=92)

    # 2. Génération du Ground Truth 3D (Tore 3D haute densité de 50 000 points, rayon ~ 1.0 m)
    mesh_gt = o3d.geometry.TriangleMesh.create_torus(torus_radius=1.0, tube_radius=0.35, radial_resolution=120, tubular_resolution=80)
    mesh_gt.compute_vertex_normals()
    pcd_gt = mesh_gt.sample_points_uniformly(number_of_points=50000)
    pcd_gt.paint_uniform_color([0.2, 0.7, 0.3])  # Vert référence
    o3d.io.write_point_cloud(str(paths["ground_truth_ply"]), pcd_gt)

    gt_points = np.asarray(pcd_gt.points)
    rng = np.random.default_rng(seed=42)

    # 3. Simulation d'une sortie VGGT (Facteur d'échelle arbitraire s=0.45, rotation 15°, translation, bruit léger ~2.5mm)
    scale_vggt = 0.45
    R_vggt = pcd_gt.get_rotation_matrix_from_xyz((0.15, -0.25, 0.10))
    t_vggt = np.array([1.2, -0.8, 2.5])
    noise_vggt = rng.normal(loc=0.0, scale=0.0025, size=gt_points.shape)

    vggt_points = scale_vggt * ((gt_points + noise_vggt) @ R_vggt.T) + t_vggt
    pcd_vggt = o3d.geometry.PointCloud()
    pcd_vggt.points = o3d.utility.Vector3dVector(vggt_points)
    pcd_vggt.paint_uniform_color([0.2, 0.5, 0.9])
    o3d.io.write_point_cloud(str(paths["vggt_ply"]), pcd_vggt)

    # 4. Simulation d'une sortie COLMAP (Précise à 1.5mm sur les zones texturées, mais trou/occultation sur x > 0.8)
    mask_visible = gt_points[:, 0] < 0.85  # Simule une zone sans texture / occultée non reconstruite
    colmap_base = gt_points[mask_visible]
    noise_colmap = rng.normal(loc=0.0, scale=0.0015, size=colmap_base.shape)
    scale_colmap = 0.80
    R_colmap = pcd_gt.get_rotation_matrix_from_xyz((-0.05, 0.12, -0.08))
    t_colmap = np.array([-0.5, 0.4, -1.1])

    colmap_points = scale_colmap * ((colmap_base + noise_colmap) @ R_colmap.T) + t_colmap
    pcd_colmap = o3d.geometry.PointCloud()
    pcd_colmap.points = o3d.utility.Vector3dVector(colmap_points)
    pcd_colmap.paint_uniform_color([0.9, 0.5, 0.2])
    o3d.io.write_point_cloud(str(paths["colmap_ply"]), pcd_colmap)

    print(f"[OK] Données générées avec succès dans '{paths['scene_dir']}' et '{paths['output_dir']}'.")


def check_environment(loader: SceneDataLoader) -> None:
    """Affiche un rapport d'audit complet sur les scènes et fichiers détectés."""
    print("=" * 75)
    print(" AUDIT DU DATA LOADER - MVS BENCHMARK APP")
    print("=" * 75)
    for scene in loader.list_available_scenes():
        meta = loader.get_scene_metadata(scene)
        paths = loader.get_scene_paths(scene)
        imgs_all = loader.get_all_images(scene)
        imgs_sampled = loader.get_sampled_images(scene, max_views=10)

        print(f"\n[Scène : {scene}] ({meta['dataset']})")
        print(f"  ├─ Description       : {meta['description']}")
        print(f"  ├─ Seuil F-Score (τ) : {meta['tau_mm']} mm | Voxel : {meta['voxel_downsample_m']*1000:.1f} mm")
        print(f"  ├─ Images détectées  : {len(imgs_all)} vues (Échantillonnage test N=10 -> {len(imgs_sampled)} vues)")

        for label, key in [
            ("Ground Truth (.ply)", "ground_truth_ply"),
            ("Sortie VGGT (.ply) ", "vggt_ply"),
            ("Sortie COLMAP (.ply)", "colmap_ply"),
        ]:
            ply_path = paths[key]
            if ply_path.exists():
                _, stats = loader.load_point_cloud(ply_path)
                print(
                    f"  ├─ {label} : PRÉSENT ({stats.num_points:,} pts | "
                    f"Diagonale = {stats.diagonal_length:.3f} m)"
                )
            else:
                print(f"  ├─ {label} : MANQUANT ({ply_path})")
    print("\n" + "=" * 75)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Utilitaire DataOps pour le projet MVS 3D.")
    parser.add_argument("--generate-mock", action="store_true", help="Génère des données 3D synthétiques calibrées.")
    parser.add_argument("--check", action="store_true", help="Vérifie l'état des dossiers et teste le DataLoader.")
    args = parser.parse_args()

    data_loader = SceneDataLoader()

    if args.generate_mock:
        generate_synthetic_benchmark(data_loader, scene_name="Ignatius")
        check_environment(data_loader)
    elif args.check:
        check_environment(data_loader)
    else:
        parser.print_help()
