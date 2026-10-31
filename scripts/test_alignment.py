"""
Script de validation et de test unitaire du moteur de recalage 3D (Sim(3) + Multi-Scale ICP).
Aligne les reconstructions VGGT et COLMAP sur le Ground Truth LiDAR pour toutes les scènes
et sauvegarde les nuages recalés dans data/outputs/<Scene>/*_aligned.ply.
"""

import sys
from pathlib import Path
import numpy as np
import open3d as o3d

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.loader import SceneDataLoader
from src.alignment import PointCloudAligner


def run_alignment_benchmark() -> None:
    loader = SceneDataLoader()
    aligner = PointCloudAligner(loader.config.get("alignment", {}))

    print("=" * 88)
    print(" BENCHMARK DU MOTEUR DE RECALAGE 3D (Sim(3) + Multi-Scale ICP)")
    print("=" * 88)

    for scene in loader.list_available_scenes():
        paths = loader.get_scene_paths(scene)
        meta = loader.get_scene_metadata(scene)
        voxel_m = float(meta["voxel_downsample_m"])

        if not paths["ground_truth_ply"].exists():
            continue

        pcd_gt, stats_gt = loader.load_point_cloud(paths["ground_truth_ply"])
        print(f"\n[Scène : {scene}] | Voxel = {voxel_m*1000:.1f} mm | Diag GT = {stats_gt.diagonal_length:.3f} m")
        print("-" * 88)

        for method, ply_key in [("VGGT", "vggt_ply"), ("COLMAP", "colmap_ply")]:
            src_path = paths[ply_key]
            if not src_path.exists():
                continue

            pcd_src, _ = loader.load_point_cloud(src_path)
            res = aligner.align(pcd_src, pcd_gt, voxel_size=voxel_m)

            # Sauvegarde du nuage aligné pour les phases suivantes
            aligned_out_path = paths["output_dir"] / f"{method.lower()}_aligned.ply"
            o3d.io.write_point_cloud(str(aligned_out_path), res.aligned_cloud)

            # Vérification de la diagonale après recalage
            pts_aligned = np.asarray(res.aligned_cloud.points)
            diag_aligned = float(np.linalg.norm(pts_aligned.max(axis=0) - pts_aligned.min(axis=0)))

            print(
                f"  ► {method:<6} | Échelle estimée s = {res.estimated_scale:.4f} | "
                f"Fitness : {res.fitness_before*100:5.1f}% -> {res.fitness_after*100:5.1f}% | "
                f"RMSE : {res.inlier_rmse_before_mm:6.2f} mm -> {res.inlier_rmse_after_mm:5.2f} mm | "
                f"Diag recalée = {diag_aligned:.3f} m ({res.execution_time_sec}s)"
            )

    print("\n" + "=" * 88)
    print("[OK] Tous les nuages recalés (*_aligned.ply) ont été générés avec succès.")


if __name__ == "__main__":
    run_alignment_benchmark()
