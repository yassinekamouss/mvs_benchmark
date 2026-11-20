"""
Module d'évaluation mathématique et géométrique des nuages de points 3D.
Implémente les métriques standards des benchmarks DTU et Tanks and Temples :
Accuracy, Completeness, Distance de Chamfer, Précision, Rappel et F-Score multi-seuils.
Auteurs : Yahya AHMANE & Yassine KAMOUSS
"""

from dataclasses import dataclass
from typing import Dict, List, Any, Optional
import time
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree


@dataclass
class EvaluationMetrics:
    """Structure contenant toutes les métriques quantitatives et distributions d'erreurs."""
    scene_name: str
    method_name: str
    tau_mm: float                          # Seuil officiel F-Score de la scène (en mm)
    num_points_eval_recon: int             # Nombre de points évalués (nuage reconstruit)
    num_points_eval_gt: int                # Nombre de points évalués (vérité terrain)

    # Métriques de distance (DTU Benchmark - plus bas est meilleur, en mm)
    accuracy_mean_mm: float                # Distance moyenne Reconstruction -> GT
    accuracy_median_mm: float              # Distance médiane Reconstruction -> GT
    accuracy_std_mm: float                 # Écart-type de l'erreur de précision
    completeness_mean_mm: float            # Distance moyenne GT -> Reconstruction
    completeness_median_mm: float          # Distance médiane GT -> Reconstruction
    completeness_std_mm: float             # Écart-type de l'erreur de complétude
    chamfer_distance_mm: float             # Moyenne (Accuracy + Completeness) / 2
    hausdorff_95_mm: float                 # 95ème centile de l'erreur globale (robuste aux extrêmes)

    # Métriques de classification géométrique au seuil tau (Tanks & Temples - plus haut est meilleur, en %)
    precision_pct: float                   # % de points reconstruits à distance < tau_mm du GT
    recall_pct: float                      # % de points du GT couverts à distance < tau_mm
    fscore_pct: float                      # Moyenne harmonique de Precision et Recall

    # Courbes multi-seuils pour les graphiques analytiques (Plotly)
    curve_thresholds_mm: List[float]
    curve_precision_pct: List[float]
    curve_recall_pct: List[float]
    curve_fscore_pct: List[float]

    # Vecteurs bruts des distances par point (en mm) pour la génération de Heatmap 3D et d'histogrammes
    dist_recon_to_gt_mm: np.ndarray
    dist_gt_to_recon_mm: np.ndarray

    execution_time_sec: float

    def to_summary_dict(self) -> Dict[str, Any]:
        """Retourne un dictionnaire léger (sans les tableaux NumPy lourds) pour export JSON/CSV."""
        return {
            "scene_name": self.scene_name,
            "method_name": self.method_name,
            "tau_mm": self.tau_mm,
            "num_points_eval_recon": self.num_points_eval_recon,
            "num_points_eval_gt": self.num_points_eval_gt,
            "accuracy_mean_mm": self.accuracy_mean_mm,
            "accuracy_median_mm": self.accuracy_median_mm,
            "completeness_mean_mm": self.completeness_mean_mm,
            "completeness_median_mm": self.completeness_median_mm,
            "chamfer_distance_mm": self.chamfer_distance_mm,
            "hausdorff_95_mm": self.hausdorff_95_mm,
            "precision_pct": self.precision_pct,
            "recall_pct": self.recall_pct,
            "fscore_pct": self.fscore_pct,
            "curve_thresholds_mm": self.curve_thresholds_mm,
            "curve_precision_pct": self.curve_precision_pct,
            "curve_recall_pct": self.curve_recall_pct,
            "curve_fscore_pct": self.curve_fscore_pct,
            "execution_time_sec": self.execution_time_sec,
        }


class GeometryEvaluator:
    """
    Moteur de calcul des métriques géométriques 3D par indexation spatiale KD-Tree (cKDTree).
    """

    def __init__(self, eval_cfg: Optional[Dict[str, Any]] = None) -> None:
        cfg = eval_cfg or {}
        self.max_points_eval: int = int(cfg.get("max_points_eval", 250000))
        self.default_thresholds_mm: List[float] = [
            float(x) for x in cfg.get(
                "fscore_thresholds_mm",
                [1.0, 2.0, 3.0, 5.0, 7.5, 10.0, 15.0, 20.0, 30.0, 50.0],
            )
        ]

    @staticmethod
    def _subsample_points_if_needed(points: np.ndarray, max_pts: int, seed: int = 42) -> np.ndarray:
        """Sous-échantillonne aléatoirement si le nuage dépasse la limite max_pts."""
        if len(points) <= max_pts:
            return points
        rng = np.random.default_rng(seed=seed)
        indices = rng.choice(len(points), size=max_pts, replace=False)
        return points[indices]

    @staticmethod
    def _compute_fscore_at_threshold(
        dist_p2g_mm: np.ndarray, dist_g2p_mm: np.ndarray, tau_mm: float
    ) -> tuple[float, float, float]:
        """Calcule Précision (%), Rappel (%) et F-Score (%) pour un seuil tau_mm donné."""
        precision = float(np.mean(dist_p2g_mm < tau_mm) * 100.0)
        recall = float(np.mean(dist_g2p_mm < tau_mm) * 100.0)
        if precision + recall > 1e-9:
            fscore = (2.0 * precision * recall) / (precision + recall)
        else:
            fscore = 0.0
        return round(precision, 2), round(recall, 2), round(fscore, 2)

    def evaluate(
        self,
        recon_pcd: o3d.geometry.PointCloud,
        gt_pcd: o3d.geometry.PointCloud,
        tau_mm: float = 5.0,
        scene_name: str = "Unknown",
        method_name: str = "Unknown",
    ) -> EvaluationMetrics:
        """
        Compare quantitativement un nuage reconstruit aligné (`recon_pcd`) à la vérité terrain (`gt_pcd`).

        Args:
            recon_pcd (PointCloud): Nuage reconstruit déjà recalé dans le repère du Ground Truth.
            gt_pcd (PointCloud): Nuage de référence Ground Truth (LiDAR ou VGGT de référence).
            tau_mm (float): Seuil officiel d'évaluation du F-Score en millimètres.
            scene_name (str): Nom de la scène évaluée.
            method_name (str): Nom de la méthode évaluée ('VGGT', 'COLMAP', etc.).

        Returns:
            EvaluationMetrics: Objet complet contenant scores globaux, courbes et erreurs par point.
        """
        t0 = time.time()

        pts_recon_raw = np.asarray(recon_pcd.points)
        pts_gt_raw = np.asarray(gt_pcd.points)

        if len(pts_recon_raw) == 0 or len(pts_gt_raw) == 0:
            raise ValueError("Impossible d'évaluer un nuage de points vide.")

        pts_recon = self._subsample_points_if_needed(pts_recon_raw, self.max_points_eval)
        pts_gt = self._subsample_points_if_needed(pts_gt_raw, self.max_points_eval)

        # 1. Construction des KD-Trees C++ multi-threadés
        tree_gt = cKDTree(pts_gt)
        tree_recon = cKDTree(pts_recon)

        # 2. Recherche du plus proche voisin dans les deux directions (workers=-1 utilise tous les coeurs CPU)
        dist_p2g_m, _ = tree_gt.query(pts_recon, k=1, workers=-1)
        dist_g2p_m, _ = tree_recon.query(pts_gt, k=1, workers=-1)

        # Conversion mètres -> millimètres
        dist_p2g_mm = dist_p2g_m * 1000.0
        dist_g2p_mm = dist_g2p_m * 1000.0

        # 3. Calcul des métriques de distance (Précision, Complétude, Chamfer, Hausdorff 95%)
        acc_mean = float(np.mean(dist_p2g_mm))
        acc_median = float(np.median(dist_p2g_mm))
        acc_std = float(np.std(dist_p2g_mm))

        comp_mean = float(np.mean(dist_g2p_mm))
        comp_median = float(np.median(dist_g2p_mm))
        comp_std = float(np.std(dist_g2p_mm))

        chamfer_mm = 0.5 * (acc_mean + comp_mean)
        combined_dist = np.concatenate([dist_p2g_mm, dist_g2p_mm])
        hausdorff_95 = float(np.percentile(combined_dist, 95))

        # 4. Calcul de la Précision, du Rappel et du F-Score au seuil nominal tau_mm
        prec_nom, rec_nom, f1_nom = self._compute_fscore_at_threshold(
            dist_p2g_mm, dist_g2p_mm, tau_mm
        )

        # 5. Calcul vectorisé des courbes sur toute la plage de seuils tau
        thresholds = sorted(set(self.default_thresholds_mm + [float(tau_mm)]))
        c_prec, c_rec, c_f1 = [], [], []
        for th in thresholds:
            p_val, r_val, f_val = self._compute_fscore_at_threshold(dist_p2g_mm, dist_g2p_mm, th)
            c_prec.append(p_val)
            c_rec.append(r_val)
            c_f1.append(f_val)

        # Pour la Heatmap 3D : si `pts_recon_raw` n'a pas été sous-échantillonné, on garde dist_p2g_mm,
        # sinon on calcule les distances exactes sur tous les points de recon_pcd pour la colorisation.
        if len(pts_recon_raw) != len(pts_recon):
            full_dist_m, _ = tree_gt.query(pts_recon_raw, k=1, workers=-1)
            full_dist_p2g_mm = full_dist_m * 1000.0
        else:
            full_dist_p2g_mm = dist_p2g_mm

        elapsed = round(time.time() - t0, 3)

        return EvaluationMetrics(
            scene_name=scene_name,
            method_name=method_name,
            tau_mm=float(tau_mm),
            num_points_eval_recon=len(pts_recon),
            num_points_eval_gt=len(pts_gt),
            accuracy_mean_mm=round(acc_mean, 3),
            accuracy_median_mm=round(acc_median, 3),
            accuracy_std_mm=round(acc_std, 3),
            completeness_mean_mm=round(comp_mean, 3),
            completeness_median_mm=round(comp_median, 3),
            completeness_std_mm=round(comp_std, 3),
            chamfer_distance_mm=round(chamfer_mm, 3),
            hausdorff_95_mm=round(hausdorff_95, 3),
            precision_pct=prec_nom,
            recall_pct=rec_nom,
            fscore_pct=f1_nom,
            curve_thresholds_mm=thresholds,
            curve_precision_pct=c_prec,
            curve_recall_pct=c_rec,
            curve_fscore_pct=c_f1,
            dist_recon_to_gt_mm=full_dist_p2g_mm,
            dist_gt_to_recon_mm=dist_g2p_mm,
            execution_time_sec=elapsed,
        )
