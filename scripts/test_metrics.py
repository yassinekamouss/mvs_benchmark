"""
Script de validation du moteur d'évaluation géométrique (Accuracy, Completeness, Chamfer, F-Score).
Exécute le benchmark complet sur les 3 scènes et sauvegarde les résultats dans metrics_summary.json.
"""

import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.loader import SceneDataLoader
from src.alignment import PointCloudAligner
from src.metrics import GeometryEvaluator


def run_metrics_benchmark() -> None:
    loader = SceneDataLoader()
    aligner = PointCloudAligner(loader.config.get("alignment", {}))
    evaluator = GeometryEvaluator(loader.config.get("evaluation", {}))

    print("=" * 100)
    print(" BENCHMARK QUANTITATIF COMPLET : PRÉCISION, COMPLÉTUDE, CHAMFER & F-SCORE (TANKS & TEMPLES)")
    print("=" * 100)

    for scene in loader.list_available_scenes():
        paths = loader.get_scene_paths(scene)
        meta = loader.get_scene_metadata(scene)
        tau_mm = float(meta["tau_mm"])
        voxel_m = float(meta["voxel_downsample_m"])

        if not paths["ground_truth_ply"].exists():
            continue

        pcd_gt, _ = loader.load_point_cloud(paths["ground_truth_ply"])
        summary_results = {}

        print(f"\n[Scène : {scene}] | Seuil officiel τ = {tau_mm} mm | Description : {meta['description']}")
        print("-" * 100)
        print(
            f"  {'Méthode / Comparaison':<24} | {'Acc (mm)':>9} | {'Comp (mm)':>9} | "
            f"{'Chamfer':>9} | {'Précision':>9} | {'Rappel':>8} | {'F-Score':>8}"
        )
        print("-" * 100)

        aligned_clouds = {}

        for method, ply_key in [("VGGT", "vggt_ply"), ("COLMAP", "colmap_ply")]:
            aligned_path = paths["output_dir"] / f"{method.lower()}_aligned.ply"
            if aligned_path.exists():
                pcd_aligned, _ = loader.load_point_cloud(aligned_path)
            else:
                pcd_raw, _ = loader.load_point_cloud(paths[ply_key])
                align_res = aligner.align(pcd_raw, pcd_gt, voxel_size=voxel_m)
                pcd_aligned = align_res.aligned_cloud

            aligned_clouds[method] = pcd_aligned

            metrics = evaluator.evaluate(
                recon_pcd=pcd_aligned,
                gt_pcd=pcd_gt,
                tau_mm=tau_mm,
                scene_name=scene,
                method_name=f"{method} vs GT LiDAR",
            )
            summary_results[method.lower()] = metrics.to_summary_dict()

            print(
                f"  {metrics.method_name:<24} | {metrics.accuracy_mean_mm:8.2f}  | "
                f"{metrics.completeness_mean_mm:8.2f}  | {metrics.chamfer_distance_mm:8.2f}  | "
                f"{metrics.precision_pct:8.1f}% | {metrics.recall_pct:7.1f}% | {metrics.fscore_pct:7.1f}%"
            )

        # Comparaison croisée directe : COLMAP évalué par rapport à VGGT comme référence
        if "COLMAP" in aligned_clouds and "VGGT" in aligned_clouds:
            cross_metrics = evaluator.evaluate(
                recon_pcd=aligned_clouds["COLMAP"],
                gt_pcd=aligned_clouds["VGGT"],
                tau_mm=tau_mm,
                scene_name=scene,
                method_name="COLMAP vs VGGT (Réf)",
            )
            summary_results["colmap_vs_vggt"] = cross_metrics.to_summary_dict()
            print(
                f"  {cross_metrics.method_name:<24} | {cross_metrics.accuracy_mean_mm:8.2f}  | "
                f"{cross_metrics.completeness_mean_mm:8.2f}  | {cross_metrics.chamfer_distance_mm:8.2f}  | "
                f"{cross_metrics.precision_pct:8.1f}% | {cross_metrics.recall_pct:7.1f}% | {cross_metrics.fscore_pct:7.1f}%"
            )

        # Sauvegarde JSON pour chargement instantané dans Streamlit et export vers le rapport
        out_json = paths["output_dir"] / "metrics_summary.json"
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(summary_results, f, indent=4, ensure_ascii=False)

    print("\n" + "=" * 100)
    print("[OK] Évaluations terminées et sauvegardées dans data/outputs/<Scene>/metrics_summary.json.")


if __name__ == "__main__":
    run_metrics_benchmark()
