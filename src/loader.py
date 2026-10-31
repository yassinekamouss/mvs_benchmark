"""
Module d'ingestion et de gestion des données (Images Multi-Vues & Nuages de Points 3D).
Fait partie du projet de Reconstruction 3D par Stéréo Multi-Vues (MVS).
Auteurs : Yahya AHMANE & Yassine KAMOUSS
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import yaml
import numpy as np
import open3d as o3d
from PIL import Image


@dataclass
class PointCloudStats:
    """Structure de données résumant les propriétés géométriques d'un nuage de points."""
    file_path: str
    num_points: int
    has_colors: bool
    has_normals: bool
    bbox_min: np.ndarray
    bbox_max: np.ndarray
    extent: np.ndarray
    centroid: np.ndarray

    @property
    def diagonal_length(self) -> float:
        """Retourne la longueur de la diagonale de la boîte englobante (échelle de la scène)."""
        return float(np.linalg.norm(self.extent))


class SceneDataLoader:
    """
    Gestionnaire de chargement des jeux d'images multi-vues et des nuages de points (.ply)
    pour l'évaluation de reconstructions 3D denses (COLMAP, VGGT, Tanks and Temples).
    """

    SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

    def __init__(self, config_path: str = "configs/config.yaml") -> None:
        """
        Initialise le chargeur de données à partir du fichier de configuration YAML.

        Args:
            config_path (str): Chemin vers le fichier config.yaml.
        """
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(f"Fichier de configuration introuvable : {self.config_path}")

        with open(self.config_path, "r", encoding="utf-8") as f:
            self.config: Dict[str, Any] = yaml.safe_load(f)

        self.data_root = Path(self.config["project"]["data_root"])
        self.output_root = Path(self.config["project"]["output_root"])

    def list_available_scenes(self) -> List[str]:
        """Retourne la liste des scènes configurées dans config.yaml."""
        return list(self.config.get("scenes", {}).keys())

    def get_scene_metadata(self, scene_name: str) -> Dict[str, Any]:
        """
        Récupère les métadonnées et seuils officiels (tau_mm, voxel_size) d'une scène.

        Args:
            scene_name (str): Nom de la scène (ex: 'Ignatius', 'Truck', 'Barn').
        """
        scenes = self.config.get("scenes", {})
        if scene_name not in scenes:
            raise ValueError(f"Scène '{scene_name}' non reconnue. Disponibles : {list(scenes.keys())}")
        return scenes[scene_name]

    def get_scene_paths(self, scene_name: str) -> Dict[str, Path]:
        """
        Retourne les chemins normalisés (dossier images, vérité terrain, sorties) pour une scène.
        """
        scene_dir = self.data_root / scene_name
        out_dir = self.output_root / scene_name
        out_dir.mkdir(parents=True, exist_ok=True)

        return {
            "scene_dir": scene_dir,
            "images_dir": scene_dir / "images",
            "ground_truth_ply": scene_dir / "ground_truth.ply",
            "output_dir": out_dir,
            "vggt_ply": out_dir / "vggt_dense.ply",
            "colmap_ply": out_dir / "colmap_dense.ply",
        }

    def get_all_images(self, scene_name: str) -> List[Path]:
        """
        Liste toutes les images valides triées par ordre alphabétique dans le dossier de la scène.
        """
        images_dir = self.get_scene_paths(scene_name)["images_dir"]
        if not images_dir.exists():
            return []

        images = [
            p for p in images_dir.iterdir()
            if p.is_file() and p.suffix.lower() in self.SUPPORTED_IMAGE_EXTS
        ]
        return sorted(images)

    def get_sampled_images(self, scene_name: str, max_views: Optional[int] = None) -> List[Path]:
        """
        Sélectionne un sous-ensemble d'images espacées uniformément (Uniform Keyframe Sampling).
        Idéal pour tester l'effet du nombre de vues (ex: 15, 30, 50) sur la qualité de reconstruction.

        Args:
            scene_name (str): Nom de la scène.
            max_views (Optional[int]): Nombre maximal de vues souhaitées. Si None, retourne tout.

        Returns:
            List[Path]: Liste ordonnée des chemins des images sélectionnées.
        """
        all_images = self.get_all_images(scene_name)
        total = len(all_images)

        if total == 0 or max_views is None or max_views >= total:
            return all_images

        # Échantillonnage linéaire uniforme entre 0 et total - 1
        indices = np.linspace(0, total - 1, num=max_views, dtype=int)
        return [all_images[i] for i in indices]

    @staticmethod
    def inspect_image(image_path: Path) -> Dict[str, Any]:
        """Retourne la résolution (largeur, hauteur) et le mode couleur d'une image."""
        with Image.open(image_path) as img:
            return {
                "filename": image_path.name,
                "width": img.width,
                "height": img.height,
                "mode": img.mode,
            }

    @staticmethod
    def load_point_cloud(ply_path: Path | str) -> Tuple[o3d.geometry.PointCloud, PointCloudStats]:
        """
        Charge un nuage de points 3D (.ply) via Open3D et extrait ses statistiques géométriques.

        Args:
            ply_path (Path | str): Chemin vers le fichier .ply.

        Returns:
            Tuple[o3d.geometry.PointCloud, PointCloudStats]: L'objet PointCloud et ses métadonnées.
        """
        path = Path(ply_path)
        if not path.exists():
            raise FileNotFoundError(f"Nuage de points introuvable : {path}")

        pcd = o3d.io.read_point_cloud(str(path))
        if pcd.is_empty():
            raise ValueError(f"Le nuage de points chargé est vide : {path}")

        points = np.asarray(pcd.points)
        bbox_min = points.min(axis=0)
        bbox_max = points.max(axis=0)
        extent = bbox_max - bbox_min
        centroid = points.mean(axis=0)

        stats = PointCloudStats(
            file_path=str(path),
            num_points=len(points),
            has_colors=pcd.has_colors(),
            has_normals=pcd.has_normals(),
            bbox_min=bbox_min,
            bbox_max=bbox_max,
            extent=extent,
            centroid=centroid,
        )
        return pcd, stats
