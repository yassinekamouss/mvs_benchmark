"""
Script de préparation, de vérification des datasets et de génération de données de test calibrées.
Usage :
    python scripts/setup_data.py --generate-mock
    python scripts/setup_data.py --import-bundle chemin/vers/Ignatius_bundle.zip --scene Ignatius
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
from src.reconstructor import ReconstructionManager, ReconstructionResult


def _create_base_geometry(scene_name: str, num_points: int = 50000) -> o3d.geometry.PointCloud:
    """Crée une géométrie 3D synthétique représentative de chaque scène de Tanks and Temples."""
    if scene_name == "Ignatius":
        # Statue complexe -> Tore haute résolution
        mesh = o3d.geometry.TriangleMesh.create_torus(
            torus_radius=1.0, tube_radius=0.35, radial_resolution=120, tubular_resolution=80
        )
    elif scene_name == "Truck":
        # Camion industriel -> Combinaison d'un châssis (boîte) et d'un cylindre
        box = o3d.geometry.TriangleMesh.create_box(width=2.4, height=1.2, depth=1.1)
        box.translate([-1.2, -0.6, -0.55])
        cyl = o3d.geometry.TriangleMesh.create_cylinder(radius=0.5, height=1.2, resolution=60, split=10)
        cyl.translate([0.8, 0.0, 0.0])
        mesh = box + cyl
    else:
        # Barn (Bâtiment grande échelle) -> Structure architecturale large
        mesh = o3d.geometry.TriangleMesh.create_box(width=4.5, height=2.8, depth=2.2)
        mesh.translate([-2.25, -1.4, -1.1])

    mesh.compute_vertex_normals()
    pcd = mesh.sample_points_uniformly(number_of_points=num_points)
    pcd.paint_uniform_color([0.2, 0.7, 0.3])
    return pcd


def generate_synthetic_benchmark(
    loader: SceneDataLoader, recon_mgr: ReconstructionManager, scene_name: str
) -> None:
    """
    Génère un jeu de données synthétique calibré (Images + Ground Truth + VGGT + COLMAP + timings.json)
    reflétant fidèlement les propriétés géométriques étudiées dans le Devoir 1 :
      - VGGT : Ultra-rapide (~3-5s), complet à 100%, mais échelle arbitraire et léger bruit métrique.
      - COLMAP : Très précis sur les zones texturées, mais lent (~450-650s) et incomplet sur zones lisses/occultées.
    """
    print(f"[*] Génération du benchmark calibré pour la scène '{scene_name}'...")
    paths = loader.get_scene_paths(scene_name)
    meta = loader.get_scene_metadata(scene_name)
    paths["images_dir"].mkdir(parents=True, exist_ok=True)
    paths["output_dir"].mkdir(parents=True, exist_ok=True)

    num_views = int(meta.get("default_num_views", 30))

    # 1. Génération des vues multi-angles
    for i in range(1, num_views + 1):
        img = Image.new("RGB", (640, 360), color=(22, 28, 42))
        draw = ImageDraw.Draw(img)
        angle = int((i - 1) * (360 / num_views))
        cx = 320 + int(90 * np.cos(np.radians(angle)))
        cy = 180 + int(45 * np.sin(np.radians(angle)))
        draw.rectangle([cx - 70, cy - 50, cx + 70, cy + 50], fill=(55, 125, 205), outline=(240, 240, 240), width=2)
        draw.text((20, 20), f"Tanks & Temples: {scene_name} | View {i:02d}/{num_views} ({angle} deg)", fill=(235, 235, 235))
        img.save(paths["images_dir"] / f"frame_{i:04d}.jpg", quality=90)

    # 2. Ground Truth LiDAR (50 000 points)
    pcd_gt = _create_base_geometry(scene_name, num_points=50000)
    o3d.io.write_point_cloud(str(paths["ground_truth_ply"]), pcd_gt)
    gt_points = np.asarray(pcd_gt.points)
    rng = np.random.default_rng(seed=abs(hash(scene_name)) % 10000)

    # Bruit proportionnel au seuil officiel tau_mm de la scène
    tau_m = float(meta["tau_mm"]) / 1000.0

    # 3. Sortie VGGT (Feed-Forward : complet, échelle arbitraire s=0.45, bruit ~ 0.85 * tau)
    scale_vggt = 0.45
    R_vggt = pcd_gt.get_rotation_matrix_from_xyz((0.15, -0.25, 0.10))
    t_vggt = np.array([1.2, -0.8, 2.5])
    noise_vggt = rng.normal(loc=0.0, scale=0.85 * tau_m, size=gt_points.shape)

    vggt_points = scale_vggt * ((gt_points + noise_vggt) @ R_vggt.T) + t_vggt
    pcd_vggt = o3d.geometry.PointCloud()
    pcd_vggt.points = o3d.utility.Vector3dVector(vggt_points)
    pcd_vggt.paint_uniform_color([0.2, 0.5, 0.9])
    o3d.io.write_point_cloud(str(paths["vggt_ply"]), pcd_vggt)

    # 4. Sortie COLMAP (MVS Classique : bruit très faible ~ 0.45 * tau, mais trou sur ~15% de la surface)
    x_limit = np.percentile(gt_points[:, 0], 85)  # Simule 15% de surface spéculaire/sans texture non reconstruite
    mask_visible = gt_points[:, 0] <= x_limit
    colmap_base = gt_points[mask_visible]
    noise_colmap = rng.normal(loc=0.0, scale=0.45 * tau_m, size=colmap_base.shape)
    scale_colmap = 0.80
    R_colmap = pcd_gt.get_rotation_matrix_from_xyz((-0.05, 0.12, -0.08))
    t_colmap = np.array([-0.5, 0.4, -1.1])

    colmap_points = scale_colmap * ((colmap_base + noise_colmap) @ R_colmap.T) + t_colmap
    pcd_colmap = o3d.geometry.PointCloud()
    pcd_colmap.points = o3d.utility.Vector3dVector(colmap_points)
    pcd_colmap.paint_uniform_color([0.9, 0.5, 0.2])
    o3d.io.write_point_cloud(str(paths["colmap_ply"]), pcd_colmap)

    # 5. Enregistrement des métadonnées de performance (timings.json)
    recon_mgr.save_metadata(
        scene_name,
        "vggt",
        ReconstructionResult(
            scene_name=scene_name,
            method="VGGT",
            status="cached",
            ply_path=str(paths["vggt_ply"]),
            num_views_used=num_views,
            num_points=len(vggt_points),
            execution_time_sec=round(3.2 + 0.08 * num_views, 2),
            device_name="NVIDIA Tesla T4 (16GB VRAM) - Feed-Forward",
            notes="Inférence directe en une passe avant sans calibrage préalable.",
        ),
    )
    recon_mgr.save_metadata(
        scene_name,
        "colmap",
        ReconstructionResult(
            scene_name=scene_name,
            method="COLMAP",
            status="cached",
            ply_path=str(paths["colmap_ply"]),
            num_views_used=num_views,
            num_points=len(colmap_points),
            execution_time_sec=round(14.5 * num_views, 1),
            device_name="NVIDIA Tesla T4 (16GB VRAM) - SfM + PatchMatch MVS",
            notes="Extraction SIFT + Bundle Adjustment + PatchMatch Stereo Fusion.",
        ),
    )
    print(f"[OK] Scène '{scene_name}' prête ({num_views} vues, GT, VGGT, COLMAP & timings.json).")


def check_environment(loader: SceneDataLoader, recon_mgr: ReconstructionManager) -> None:
    """Affiche un rapport d'audit complet sur les scènes, nuages et temps d'exécution."""
    print("=" * 80)
    print(" AUDIT DU DATA LOADER & RECONSTRUCTION ENGINE - MVS BENCHMARK APP")
    print("=" * 80)
    for scene in loader.list_available_scenes():
        meta = loader.get_scene_metadata(scene)
        paths = loader.get_scene_paths(scene)
        imgs_all = loader.get_all_images(scene)

        print(f"\n[Scène : {scene}] ({meta['dataset']})")
        print(f"  ├─ Description       : {meta['description']}")
        print(f"  ├─ Seuil F-Score (τ) : {meta['tau_mm']} mm | Voxel : {meta['voxel_downsample_m']*1000:.1f} mm")
        print(f"  ├─ Images détectées  : {len(imgs_all)} vues")

        gt_path = paths["ground_truth_ply"]
        if gt_path.exists():
            _, stats = loader.load_point_cloud(gt_path)
            print(f"  ├─ Ground Truth      : PRÉSENT ({stats.num_points:,} pts | Diag = {stats.diagonal_length:.3f} m)")
        else:
            print(f"  ├─ Ground Truth      : MANQUANT")

        for method in ("vggt", "colmap"):
            res = recon_mgr.run_or_load(scene, method)
            if res.status != "error":
                _, stats = loader.load_point_cloud(res.ply_path)
                print(
                    f"  ├─ Sortie {res.method:<6}     : PRÉSENT ({res.num_points:,} pts | "
                    f"Diag = {stats.diagonal_length:.3f} m | Temps = {res.execution_time_sec}s)"
                )
            else:
                print(f"  ├─ Sortie {method.upper():<6}     : MANQUANT")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Utilitaire DataOps & Reconstruction pour le projet MVS 3D.")
    parser.add_argument("--generate-mock", action="store_true", help="Génère les données calibrées pour les 3 scènes.")
    parser.add_argument("--import-bundle", type=str, help="Chemin vers un fichier .zip généré sur Google Colab.")
    parser.add_argument("--scene", type=str, default="Ignatius", help="Nom de la scène cible pour l'import.")
    parser.add_argument("--check", action="store_true", help="Vérifie l'état des dossiers et des reconstructions.")
    args = parser.parse_args()

    data_loader = SceneDataLoader()
    recon_manager = ReconstructionManager(data_loader)

    if args.generate_mock:
        for s in data_loader.list_available_scenes():
            generate_synthetic_benchmark(data_loader, recon_manager, scene_name=s)
        check_environment(data_loader, recon_manager)
    elif args.import_bundle:
        info = recon_manager.import_colab_bundle(args.import_bundle, args.scene)
        print(f"[OK] Bundle importé pour {info['scene']} ({info['extracted_count']} fichiers extraits).")
        check_environment(data_loader, recon_manager)
    elif args.check:
        check_environment(data_loader, recon_manager)
    else:
        parser.print_help()
