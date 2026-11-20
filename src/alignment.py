"""
Module de prétraitement et de recalage géométrique 3D (Sim(3) Global Init + Multi-Scale ICP).
Permet d'aligner un nuage reconstruit (VGGT ou COLMAP, sans échelle absolue) sur une vérité terrain LiDAR.
Auteurs : Yahya AHMANE & Yassine KAMOUSS
"""

from dataclasses import dataclass
from typing import Dict, Any, Tuple, List
import copy
import time
import numpy as np
import open3d as o3d


@dataclass
class AlignmentResult:
    """Structure encapsulant les résultats mathématiques et géométriques du recalage 3D."""
    aligned_cloud: o3d.geometry.PointCloud
    transformation_matrix: np.ndarray      # Matrice homogène 4x4 Sim(3) finale
    estimated_scale: float                 # Facteur d'échelle s extrait de la matrice
    rotation_matrix: np.ndarray            # Matrice de rotation pure 3x3 (SO(3))
    translation_vector: np.ndarray         # Vecteur de translation 3x1
    fitness_before: float                  # Taux de recouvrement avant ICP fin
    fitness_after: float                   # Taux de recouvrement après ICP fin (0.0 à 1.0)
    inlier_rmse_before_mm: float           # Erreur RMSE des inliers avant ICP (en mm)
    inlier_rmse_after_mm: float            # Erreur RMSE des inliers après ICP (en mm)
    num_points_clean: int                  # Nombre de points après filtrage des outliers
    execution_time_sec: float              # Temps de calcul du recalage en secondes


class PointCloudAligner:
    """
    Moteur de recalage 3D robuste combinant :
      1. Filtrage statistique du bruit (Statistical Outlier Removal) et Voxel Downsampling.
      2. Alignement initial Sim(3) (Échelle robuste + Hypothèses d'orientation PCA/SVD avec garde-fou d'échelle).
      3. Affinement multi-échelles par ICP (Coarse-to-Fine Point-to-Point avec estimation d'échelle).
    """

    def __init__(self, alignment_cfg: Dict[str, Any]) -> None:
        self.remove_outliers: bool = bool(alignment_cfg.get("remove_outliers", True))
        self.nb_neighbors: int = int(alignment_cfg.get("nb_neighbors", 20))
        self.std_ratio: int = float(alignment_cfg.get("std_ratio", 2.0))
        self.max_iterations: int = int(alignment_cfg.get("icp_max_iterations", 60))
        self.dist_multiplier: float = float(alignment_cfg.get("icp_distance_multiplier", 5.0))
        self.estimate_scale: bool = bool(alignment_cfg.get("estimate_scale", True))

    def preprocess(
        self, pcd: o3d.geometry.PointCloud, voxel_size: float, clean_outliers: bool = True
    ) -> Tuple[o3d.geometry.PointCloud, o3d.geometry.PointCloud]:
        """Nettoie le nuage de points et génère une version sous-échantillonnée rapide pour le solveur Sim(3)+ICP."""
        pcd_clean = copy.deepcopy(pcd)

        # Sous-échantillonnage rapide initial si le nuage brut est massif (> 120 000 pts, ex: sortie brute VGGT 640k pts)
        if len(pcd_clean.points) > 120000:
            rng = np.random.default_rng(seed=42)
            idx = rng.choice(len(pcd_clean.points), size=120000, replace=False)
            pcd_clean = pcd_clean.select_by_index(idx)

        if clean_outliers and self.remove_outliers and len(pcd_clean.points) > self.nb_neighbors:
            pcd_clean, _ = pcd_clean.remove_statistical_outlier(
                nb_neighbors=self.nb_neighbors, std_ratio=self.std_ratio
            )

        if voxel_size > 0:
            pcd_down = pcd_clean.voxel_down_sample(voxel_size=voxel_size)
        else:
            pcd_down = copy.deepcopy(pcd_clean)

        # Plafonnement à 15 000 points pour l'estimation de la matrice 4x4 Sim(3)+ICP (garantit un recalage < 1s)
        if len(pcd_down.points) > 15000:
            rng = np.random.default_rng(seed=42)
            idx = rng.choice(len(pcd_down.points), size=15000, replace=False)
            pcd_down = pcd_down.select_by_index(idx)

        return pcd_clean, pcd_down

    @staticmethod
    def _decompose_sim3(T: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
        """Décompose une matrice 4x4 Sim(3) en (scale, R, t)."""
        upper_3x3 = T[:3, :3]
        scale = float(np.mean(np.linalg.norm(upper_3x3, axis=0)))
        rotation = upper_3x3 / max(scale, 1e-12)
        translation = T[:3, 3].copy()
        return scale, rotation, translation

    @staticmethod
    def _build_sim3_matrix(scale: float, R: np.ndarray, mu_src: np.ndarray, mu_tgt: np.ndarray) -> np.ndarray:
        """Construit la matrice 4x4 qui applique : p' = scale * R * (p - mu_src) + mu_tgt."""
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = scale * R
        T[:3, 3] = mu_tgt - (scale * (R @ mu_src))
        return T

    def compute_initial_sim3(
        self,
        source_down: o3d.geometry.PointCloud,
        target_down: o3d.geometry.PointCloud,
        eval_threshold: float,
    ) -> np.ndarray:
        """
        Calcule la transformation initiale globale Sim(3) par rapport d'étendue robuste
        et alignement multi-hypothèses des axes principaux (PCA/SVD), avec garde-fou anti-dégénérescence.
        """
        src_pts = np.asarray(source_down.points)
        tgt_pts = np.asarray(target_down.points)

        # Utilisation conjointe du centroïde et du centre de boîte englobante robuste (5%-95%)
        mu_src = np.mean(src_pts, axis=0)
        mu_tgt = np.mean(tgt_pts, axis=0)
        bbox_center_src = 0.5 * (np.percentile(src_pts, 98, axis=0) + np.percentile(src_pts, 2, axis=0))
        bbox_center_tgt = 0.5 * (np.percentile(tgt_pts, 98, axis=0) + np.percentile(tgt_pts, 2, axis=0))

        if self.estimate_scale:
            dist_src = np.median(np.linalg.norm(src_pts - mu_src, axis=1))
            dist_tgt = np.median(np.linalg.norm(tgt_pts - mu_tgt, axis=1))
            scale_med = float(dist_tgt / max(dist_src, 1e-9))

            span_src = np.linalg.norm(np.percentile(src_pts, 98, axis=0) - np.percentile(src_pts, 2, axis=0))
            span_tgt = np.linalg.norm(np.percentile(tgt_pts, 98, axis=0) - np.percentile(tgt_pts, 2, axis=0))
            scale_span = float(span_tgt / max(span_src, 1e-9))
            candidate_scales = [scale_med, scale_span]
            ref_scale = scale_span
        else:
            candidate_scales = [1.0]
            ref_scale = 1.0

        candidate_rotations: List[np.ndarray] = [np.eye(3)]
        try:
            _, _, Vt_src = np.linalg.svd(src_pts - mu_src, full_matrices=False)
            _, _, Vt_tgt = np.linalg.svd(tgt_pts - mu_tgt, full_matrices=False)
            for s_vec in [[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]]:
                S = np.diag(s_vec)
                R_pca = Vt_tgt.T @ S @ Vt_src
                if np.linalg.det(R_pca) < 0:
                    S[2, 2] *= -1
                    R_pca = Vt_tgt.T @ S @ Vt_src
                candidate_rotations.append(R_pca)
        except np.linalg.LinAlgError:
            pass

        best_T = self._build_sim3_matrix(ref_scale, np.eye(3), mu_src, mu_tgt)
        best_score = -np.inf

        # Sous-échantillonnage léger (5 000 pts) pour tester les 20 hypothèses globales PCA instantanément
        src_coarse = (
            source_down.select_by_index(np.linspace(0, len(src_pts) - 1, 5000, dtype=int))
            if len(src_pts) > 5000
            else source_down
        )
        tgt_coarse = (
            target_down.select_by_index(np.linspace(0, len(tgt_pts) - 1, 5000, dtype=int))
            if len(tgt_pts) > 5000
            else target_down
        )

        for sc in candidate_scales:
            for c_src, c_tgt in [(mu_src, mu_tgt), (bbox_center_src, bbox_center_tgt)]:
                for R_cand in candidate_rotations:
                    T_cand = self._build_sim3_matrix(sc, R_cand, c_src, c_tgt)
                    ev = o3d.pipelines.registration.evaluate_registration(
                        src_coarse, tgt_coarse, eval_threshold * 3.0, T_cand
                    )
                    score = ev.fitness - (ev.inlier_rmse / max(eval_threshold * 3.0, 1e-6)) * 0.25
                    if score > best_score:
                        best_score = score
                        best_T = T_cand

        return best_T

    def align(
        self,
        source_pcd: o3d.geometry.PointCloud,
        target_pcd: o3d.geometry.PointCloud,
        voxel_size: float = 0.005,
    ) -> AlignmentResult:
        """Exécute le pipeline complet de recalage 3D Sim(3) + Multi-Scale ICP."""
        t0 = time.time()

        source_clean, source_down = self.preprocess(source_pcd, voxel_size=voxel_size, clean_outliers=True)
        _, target_down = self.preprocess(target_pcd, voxel_size=voxel_size, clean_outliers=False)

        base_dist = voxel_size * self.dist_multiplier

        # 1. Initialisation globale Sim(3)
        init_T = self.compute_initial_sim3(source_down, target_down, eval_threshold=base_dist)
        init_scale, _, _ = self._decompose_sim3(init_T)
        current_T = init_T.copy()

        eval_before = o3d.pipelines.registration.evaluate_registration(
            source_down, target_down, base_dist, current_T
        )

        # 2. Pyramide ICP Multi-Échelles avec garde-fou sur le facteur d'échelle
        estimation_method = o3d.pipelines.registration.TransformationEstimationPointToPoint(
            with_scaling=self.estimate_scale
        )

        for mult in [3.5, 2.0, 1.0]:
            threshold = base_dist * mult
            criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
                relative_fitness=1e-7,
                relative_rmse=1e-7,
                max_iteration=self.max_iterations,
            )
            reg_res = o3d.pipelines.registration.registration_icp(
                source_down,
                target_down,
                threshold,
                current_T,
                estimation_method,
                criteria,
            )
            cand_scale, _, _ = self._decompose_sim3(reg_res.transformation)
            # Garde-fou anti-effondrement : l'échelle ne peut pas dériver de plus de ±35% de l'échelle initiale
            if 0.65 * init_scale <= cand_scale <= 1.35 * init_scale:
                current_T = reg_res.transformation

        eval_after = o3d.pipelines.registration.evaluate_registration(
            source_down, target_down, base_dist, current_T
        )

        aligned_dense = copy.deepcopy(source_clean)
        aligned_dense.transform(current_T)

        est_scale, rot_mat, trans_vec = self._decompose_sim3(current_T)
        elapsed = round(time.time() - t0, 3)

        return AlignmentResult(
            aligned_cloud=aligned_dense,
            transformation_matrix=current_T,
            estimated_scale=est_scale,
            rotation_matrix=rot_mat,
            translation_vector=trans_vec,
            fitness_before=round(float(eval_before.fitness), 4),
            fitness_after=round(float(eval_after.fitness), 4),
            inlier_rmse_before_mm=round(float(eval_before.inlier_rmse * 1000.0), 3),
            inlier_rmse_after_mm=round(float(eval_after.inlier_rmse * 1000.0), 3),
            num_points_clean=len(aligned_dense.points),
            execution_time_sec=elapsed,
        )