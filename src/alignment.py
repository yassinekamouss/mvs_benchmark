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
      2. Alignement initial Sim(3) (mise à l'échelle robuste + alignement centroïde/PCA).
      3. Affinement multi-échelles par ICP (Coarse-to-Fine Point-to-Point avec estimation d'échelle).
    """

    def __init__(self, alignment_cfg: Dict[str, Any]) -> None:
        """
        Initialise le moteur de recalage avec les hyperparamètres de config.yaml.

        Args:
            alignment_cfg (Dict[str, Any]): Section 'alignment' du fichier config.yaml.
        """
        self.remove_outliers: bool = bool(alignment_cfg.get("remove_outliers", True))
        self.nb_neighbors: int = int(alignment_cfg.get("nb_neighbors", 20))
        self.std_ratio: float = float(alignment_cfg.get("std_ratio", 2.0))
        self.max_iterations: int = int(alignment_cfg.get("icp_max_iterations", 50))
        self.dist_multiplier: float = float(alignment_cfg.get("icp_distance_multiplier", 5.0))
        self.estimate_scale: bool = bool(alignment_cfg.get("estimate_scale", True))

    def preprocess(
        self, pcd: o3d.geometry.PointCloud, voxel_size: float, clean_outliers: bool = True
    ) -> Tuple[o3d.geometry.PointCloud, o3d.geometry.PointCloud]:
        """
        Nettoie le nuage de points et génère une version sous-échantillonnée par grille de voxels.

        Args:
            pcd (o3d.geometry.PointCloud): Nuage de points brut.
            voxel_size (float): Taille de voxel en mètres (ex: 0.005 pour 5 mm).
            clean_outliers (bool): Active le filtrage statistique du bruit.

        Returns:
            Tuple[PointCloud, PointCloud]: (Nuage dense nettoyé, Nuage sous-échantillonné avec normales).
        """
        pcd_clean = copy.deepcopy(pcd)

        if clean_outliers and self.remove_outliers and len(pcd_clean.points) > self.nb_neighbors:
            pcd_clean, _ = pcd_clean.remove_statistical_outlier(
                nb_neighbors=self.nb_neighbors, std_ratio=self.std_ratio
            )

        if voxel_size > 0:
            pcd_down = pcd_clean.voxel_down_sample(voxel_size=voxel_size)
        else:
            pcd_down = copy.deepcopy(pcd_clean)

        # Calcul des normales locales pour l'analyse géométrique
        radius_normal = max(voxel_size * 2.5, 0.01)
        pcd_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
        )
        return pcd_clean, pcd_down

    @staticmethod
    def _decompose_sim3(T: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Décompose une matrice homogène 4x4 de similarité Sim(3) en :
          - facteur d'échelle scalaire s > 0
          - matrice de rotation pure R (3x3)
          - vecteur de translation t (3,)
        """
        upper_3x3 = T[:3, :3]
        # La norme moyenne des colonnes donne le facteur d'échelle isotrope s
        scale = float(np.mean(np.linalg.norm(upper_3x3, axis=0)))
        rotation = upper_3x3 / max(scale, 1e-12)
        translation = T[:3, 3].copy()
        return scale, rotation, translation

    @staticmethod
    def _build_sim3_matrix(scale: float, R: np.ndarray, mu_src: np.ndarray, mu_tgt: np.ndarray) -> np.ndarray:
        """
        Construit la matrice 4x4 qui applique : p' = scale * R * (p - mu_src) + mu_tgt.
        """
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
        Calcule une transformation initiale globale Sim(3) (Échelle + Rotation + Translation)
        pour amener le nuage reconstruit dans le bassin de convergence du Ground Truth.
        Teste à la fois l'orientation directe et les alignements par axes principaux (PCA/SVD)
        afin de résister aux rotations initiales arbitraires et aux occultations partielles.
        """
        src_pts = np.asarray(source_down.points)
        tgt_pts = np.asarray(target_down.points)

        mu_src = np.mean(src_pts, axis=0)
        mu_tgt = np.mean(tgt_pts, axis=0)

        src_centered = src_pts - mu_src
        tgt_centered = tgt_pts - mu_tgt

        # Estimation robuste du facteur d'échelle via la médiane des distances au centroïde
        if self.estimate_scale:
            dist_src = np.median(np.linalg.norm(src_centered, axis=1))
            dist_tgt = np.median(np.linalg.norm(tgt_centered, axis=1))
            scale_init = float(dist_tgt / max(dist_src, 1e-9))
        else:
            scale_init = 1.0

        # Candidats de rotation initiale : Identité + Alignements des axes principaux (PCA)
        candidate_rotations: List[np.ndarray] = [np.eye(3)]

        try:
            _, _, Vt_src = np.linalg.svd(src_centered, full_matrices=False)
            _, _, Vt_tgt = np.linalg.svd(tgt_centered, full_matrices=False)

            # Les vecteurs propres définissent un repère à un signe près sur chaque axe (4 rotations paires dans SO(3))
            signs_list = [
                [1, 1, 1],
                [1, -1, -1],
                [-1, 1, -1],
                [-1, -1, 1],
            ]
            for s_vec in signs_list:
                S = np.diag(s_vec)
                R_pca = Vt_tgt.T @ S @ Vt_src
                if np.linalg.det(R_pca) < 0:
                    S[2, 2] *= -1
                    R_pca = Vt_tgt.T @ S @ Vt_src
                candidate_rotations.append(R_pca)
        except np.linalg.LinAlgError:
            pass

        # Sélection du meilleur candidat initial (celui qui minimise l'erreur et maximise le recouvrement)
        best_T = self._build_sim3_matrix(scale_init, np.eye(3), mu_src, mu_tgt)
        best_score = -np.inf

        for R_cand in candidate_rotations:
            T_cand = self._build_sim3_matrix(scale_init, R_cand, mu_src, mu_tgt)
            eval_res = o3d.pipelines.registration.evaluate_registration(
                source_down, target_down, eval_threshold * 3.0, T_cand
            )
            # Score combinant fort recouvrement (fitness) et faible RMSE
            score = eval_res.fitness - (eval_res.inlier_rmse / max(eval_threshold * 3.0, 1e-6)) * 0.25
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
        """
        Exécute le pipeline complet de recalage 3D entre `source_pcd` (reconstruction) et `target_pcd` (référence).

        Args:
            source_pcd (PointCloud): Nuage reconstruit à aligner (VGGT ou COLMAP).
            target_pcd (PointCloud): Nuage de vérité terrain (LiDAR Tanks and Temples).
            voxel_size (float): Taille de voxel de référence en mètres.

        Returns:
            AlignmentResult: Nuage aligné et métriques complètes de transformation.
        """
        t0 = time.time()

        # 1. Prétraitement et sous-échantillonnage
        source_clean, source_down = self.preprocess(source_pcd, voxel_size=voxel_size, clean_outliers=True)
        _, target_down = self.preprocess(target_pcd, voxel_size=voxel_size, clean_outliers=False)

        base_dist = voxel_size * self.dist_multiplier

        # 2. Initialisation Sim(3) globale (échelle + centroïde + orientation)
        current_T = self.compute_initial_sim3(source_down, target_down, eval_threshold=base_dist)

        # Évaluation juste après la mise à l'échelle/centroïde initial (avant ICP fin)
        eval_before = o3d.pipelines.registration.evaluate_registration(
            source_down, target_down, base_dist, current_T
        )

        # 3. Pyramide ICP Multi-Échelles (Coarse -> Medium -> Fine) avec estimation d'échelle Sim(3)
        estimation_method = o3d.pipelines.registration.TransformationEstimationPointToPoint(
            with_scaling=self.estimate_scale
        )

        # 3 niveaux de raffinement : grossier (3x), moyen (1.5x), fin (1.0x)
        pyramid_multipliers = [3.0, 1.5, 1.0]
        pyramid_iters = [
            max(self.max_iterations, 40),
            max(self.max_iterations, 40),
            self.max_iterations,
        ]

        for mult, max_iter in zip(pyramid_multipliers, pyramid_iters):
            threshold = base_dist * mult
            criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
                relative_fitness=1e-7,
                relative_rmse=1e-7,
                max_iteration=max_iter,
            )
            reg_res = o3d.pipelines.registration.registration_icp(
                source_down,
                target_down,
                threshold,
                current_T,
                estimation_method,
                criteria,
            )
            current_T = reg_res.transformation

        # 4. Évaluation finale au rayon nominal
        eval_after = o3d.pipelines.registration.evaluate_registration(
            source_down, target_down, base_dist, current_T
        )

        # 5. Application de la transformation 4x4 finale au nuage dense nettoyé
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
