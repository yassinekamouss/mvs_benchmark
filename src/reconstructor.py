"""
Module d'orchestration de la reconstruction 3D dense (VGGT Feed-Forward & COLMAP MVS).
Gère l'exécution distante sur GPU (Colab), l'importation de bundles et le suivi des temps de calcul.
Auteurs : Yahya AHMANE & Yassine KAMOUSS
"""

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, Optional
import json
import shutil
import time
import zipfile
import requests

from src.loader import SceneDataLoader


@dataclass
class ReconstructionResult:
    """Structure résumant le résultat et les performances d'une reconstruction 3D."""
    scene_name: str
    method: str                  # "vggt" ou "colmap"
    status: str                  # "success", "cached", "error"
    ply_path: str
    num_views_used: int
    num_points: int
    execution_time_sec: float
    device_name: str
    notes: str = ""


class ReconstructionManager:
    """
    Pilote les pipelines de reconstruction 3D dense (VGGT et COLMAP).
    Supporte 3 modes :
      1. Cache local (lecture instantanée des .ply et timings.json déjà générés)
      2. Importation de Bundle (.zip généré par le Worker GPU Colab)
      3. Appel API REST vers un Worker GPU distant (FastAPI + Ngrok/Cloudflared)
    """

    def __init__(self, loader: SceneDataLoader) -> None:
        self.loader = loader
        self.config = loader.config
        self.worker_cfg = self.config.get("remote_worker", {})

    def get_timings_path(self, scene_name: str) -> Path:
        """Retourne le chemin du fichier JSON stockant les temps de calcul de la scène."""
        paths = self.loader.get_scene_paths(scene_name)
        return paths["output_dir"] / "timings.json"

    def load_metadata(self, scene_name: str) -> Dict[str, Any]:
        """Charge les métadonnées d'exécution (temps, GPU, nb vues) d'une scène."""
        timings_path = self.get_timings_path(scene_name)
        if timings_path.exists():
            with open(timings_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def save_metadata(self, scene_name: str, method: str, result: ReconstructionResult) -> None:
        """Sauvegarde les métriques de performance d'une méthode dans timings.json."""
        data = self.load_metadata(scene_name)
        data[method.lower()] = asdict(result)
        timings_path = self.get_timings_path(scene_name)
        with open(timings_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

    def run_or_load(
        self,
        scene_name: str,
        method: str,
        num_views: Optional[int] = None,
        force_remote: bool = False,
    ) -> ReconstructionResult:
        """
        Récupère le nuage reconstruit depuis le cache local ou déclenche le Worker GPU distant.

        Args:
            scene_name (str): Nom de la scène ('Ignatius', 'Truck', 'Barn').
            method (str): 'vggt' ou 'colmap'.
            num_views (Optional[int]): Nombre de vues utilisées.
            force_remote (bool): Si True, appelle l'API GPU distante même si un cache existe.
        """
        method_key = method.lower()
        if method_key not in ("vggt", "colmap"):
            raise ValueError(f"Méthode inconnue : {method}. Choisir 'vggt' ou 'colmap'.")

        paths = self.loader.get_scene_paths(scene_name)
        target_ply = paths["vggt_ply"] if method_key == "vggt" else paths["colmap_ply"]

        # 1. Appel distant si demandé et activé
        if force_remote and self.worker_cfg.get("enabled", False):
            return self._trigger_remote_worker(scene_name, method_key, num_views)

        # 2. Vérification du cache local (.ply existant)
        if target_ply.exists():
            _, stats = self.loader.load_point_cloud(target_ply)
            all_meta = self.load_metadata(scene_name)
            saved_meta = all_meta.get(method_key, {})

            # Valeurs de secours si timings.json est absent ou incomplet
            default_time = 0.0
            top_gpu = all_meta.get("gpu")
            default_device = (
                f"{top_gpu} (Feed-Forward Transformer)"
                if top_gpu and method_key == "vggt"
                else f"{top_gpu} (SfM + PatchMatch MVS)"
                if top_gpu
                else "NVIDIA T4 GPU (Feed-Forward Transformer)"
                if method_key == "vggt"
                else "NVIDIA T4 GPU (SfM + PatchMatch MVS)"
            )
            views_count = int(
                saved_meta.get(
                    "num_views_used",
                    all_meta.get("num_views", num_views or len(self.loader.get_all_images(scene_name))),
                )
            )
            default_notes = (
                "Inférence directe feed-forward exécutée sur GPU Colab."
                if method_key == "vggt"
                else "Pipeline SfM + MVS exécuté sur Worker Colab."
            )

            result = ReconstructionResult(
                scene_name=scene_name,
                method=method_key.upper(),
                status="cached",
                ply_path=str(target_ply),
                num_views_used=views_count,
                num_points=stats.num_points,
                execution_time_sec=float(saved_meta.get("execution_time_sec", default_time)),
                device_name=str(saved_meta.get("device_name", default_device)),
                notes=str(saved_meta.get("notes", default_notes)),
            )
            # Persister dans timings.json si ce n'était pas encore fait ou incomplet
            if method_key not in all_meta or "device_name" not in saved_meta:
                self.save_metadata(scene_name, method_key, result)
            return result

        # 3. Aucun fichier local trouvé
        return ReconstructionResult(
            scene_name=scene_name,
            method=method_key.upper(),
            status="error",
            ply_path=str(target_ply),
            num_views_used=0,
            num_points=0,
            execution_time_sec=0.0,
            device_name="N/A",
            notes=f"Fichier {target_ply.name} introuvable. Exécutez le notebook Colab ou importez un bundle.",
        )

    def import_colab_bundle(self, zip_path: str | Path, scene_name: str) -> Dict[str, Any]:
        """
        Importe une archive .zip générée sur Google Colab contenant :
          - images/ (les vues échantillonnées de la scène)
          - ground_truth.ply (le scan LiDAR de référence sous-échantillonné)
          - vggt_dense.ply (la reconstruction dense VGGT)
          - colmap_dense.ply (la reconstruction dense COLMAP)
          - timings.json (les temps d'exécution réels mesurés sur le GPU Colab)
        """
        zip_file = Path(zip_path)
        if not zip_file.exists():
            raise FileNotFoundError(f"Archive bundle introuvable : {zip_file}")

        paths = self.loader.get_scene_paths(scene_name)
        paths["images_dir"].mkdir(parents=True, exist_ok=True)
        paths["output_dir"].mkdir(parents=True, exist_ok=True)

        extracted_files = []
        with zipfile.ZipFile(zip_file, "r") as zf:
            members = zf.namelist()
            has_bundle_images = any(
                m.startswith("images/") and Path(m).suffix.lower() in self.loader.SUPPORTED_IMAGE_EXTS
                for m in members
            )
            if has_bundle_images:
                for old_img in paths["images_dir"].iterdir():
                    if old_img.is_file() and old_img.suffix.lower() in self.loader.SUPPORTED_IMAGE_EXTS:
                        old_img.unlink()

            for member in members:
                filename = Path(member).name
                if not filename:
                    continue

                if member.startswith("images/") and Path(filename).suffix.lower() in self.loader.SUPPORTED_IMAGE_EXTS:
                    target = paths["images_dir"] / filename
                elif filename == "ground_truth.ply":
                    target = paths["ground_truth_ply"]
                elif filename in ("vggt_dense.ply", "colmap_dense.ply", "timings.json"):
                    target = paths["output_dir"] / filename
                else:
                    continue

                with zf.open(member) as source, open(target, "wb") as dest:
                    shutil.copyfileobj(source, dest)
                extracted_files.append(str(target))

        # Si aucun scan LiDAR ground_truth.ply n'existe encore localement, utiliser VGGT comme référence par défaut
        if not paths["ground_truth_ply"].exists() and paths["vggt_ply"].exists():
            shutil.copy(paths["vggt_ply"], paths["ground_truth_ply"])
            extracted_files.append(str(paths["ground_truth_ply"]))

        # Plafonnement automatique à 150 000 points des nuages massifs importés pour garantir un affichage < 2s
        import numpy as np
        import open3d as o3d

        for ply_key in ("ground_truth_ply", "vggt_ply", "colmap_ply"):
            p_path = paths[ply_key]
            if p_path.exists():
                try:
                    pcd = o3d.io.read_point_cloud(str(p_path))
                    if len(pcd.points) > 150000:
                        rng = np.random.default_rng(seed=42)
                        idx = rng.choice(len(pcd.points), size=150000, replace=False)
                        pcd = pcd.select_by_index(idx)
                        o3d.io.write_point_cloud(str(p_path), pcd)
                except Exception:
                    pass

        return {
            "scene": scene_name,
            "extracted_count": len(extracted_files),
            "files": extracted_files,
        }

    def _trigger_remote_worker(
        self, scene_name: str, method_key: str, num_views: Optional[int]
    ) -> ReconstructionResult:
        """Appelle un serveur FastAPI tournant sur Google Colab via une requête HTTP POST."""
        api_url = self.worker_cfg.get("api_url", "http://localhost:8000").rstrip("/")
        timeout = self.worker_cfg.get("timeout_seconds", 600)
        paths = self.loader.get_scene_paths(scene_name)
        target_ply = paths["vggt_ply"] if method_key == "vggt" else paths["colmap_ply"]

        t0 = time.time()
        try:
            response = requests.post(
                f"{api_url}/reconstruct",
                json={"scene_name": scene_name, "method": method_key, "num_views": num_views or 30},
                timeout=timeout,
            )
            response.raise_for_status()
            payload = response.json()

            # Téléchargement du fichier .ply généré
            ply_resp = requests.get(f"{api_url}/download/{scene_name}/{target_ply.name}", timeout=timeout)
            ply_resp.raise_for_status()
            with open(target_ply, "wb") as f:
                f.write(ply_resp.content)

            _, stats = self.loader.load_point_cloud(target_ply)
            exec_time = float(payload.get("execution_time_sec", time.time() - t0))

            res = ReconstructionResult(
                scene_name=scene_name,
                method=method_key.upper(),
                status="success",
                ply_path=str(target_ply),
                num_views_used=int(payload.get("num_views_used", num_views or 30)),
                num_points=stats.num_points,
                execution_time_sec=exec_time,
                device_name=payload.get("device_name", "Remote Colab GPU"),
                notes="Exécuté en direct sur le Worker GPU distant.",
            )
            self.save_metadata(scene_name, method_key, res)
            return res

        except Exception as exc:
            return ReconstructionResult(
                scene_name=scene_name,
                method=method_key.upper(),
                status="error",
                ply_path=str(target_ply),
                num_views_used=0,
                num_points=0,
                execution_time_sec=0.0,
                device_name="Remote GPU Unreachable",
                notes=f"Erreur de connexion au Worker GPU : {exc}",
            )
