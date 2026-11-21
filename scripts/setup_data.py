"""
Script de préparation, d'audit et d'importation des datasets Tanks and Temples et des bundles Colab.
Usage :
    python scripts/setup_data.py --import-all
    python scripts/setup_data.py --import-bundle data/bundles/Ignatius_bundle.zip --scene Ignatius
    python scripts/setup_data.py --check
    python scripts/setup_data.py --prepare-colab chemin/vers/Ignatius.zip --ply chemin/vers/Ignatius.ply --scene Ignatius
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import open3d as o3d

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.loader import SceneDataLoader
from src.reconstructor import ReconstructionManager


def check_environment(loader: SceneDataLoader, recon_mgr: ReconstructionManager) -> None:
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

        for method in ("vggt", "colmap"):
            res = recon_mgr.run_or_load(scene, method)
            if res.status != "error":
                _, stats = loader.load_point_cloud(res.ply_path)
                print(
                    f"  ├─ Sortie {res.method:<6}     : PRÉSENT ({res.num_points:,} pts | "
                    f"Diag = {stats.diagonal_length:.3f} m | Temps = {res.execution_time_sec}s)"
                )
    print("\n" + "=" * 80)


def prepare_light_colab_input(
    raw_zip_path: str, raw_ply_path: str | None, scene_name: str, num_views: int = 30
) -> Path:
    """
    Prépare sur le PC local une archive ultra-légère (~8-12 Mo au lieu de 320 Mo) contenant :
      - Les N vraies photos HD sous-échantillonnées uniformément à 360° depuis <Scene>.zip
      - Le vrai scan LiDAR <Scene>.ply sous-échantillonné proprement à ~180 000 points
    Cette archive légère s'uploade en 3 secondes sur Google Colab sans risque de coupure réseau.
    """
    import tempfile
    import zipfile

    zip_p = Path(raw_zip_path)
    if not zip_p.exists():
        raise FileNotFoundError(f"Fichier ZIP introuvable : {zip_p}")

    out_zip = ROOT_DIR / "data" / "colab_inputs" / f"{scene_name}_light_colab.zip"
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        with zipfile.ZipFile(zip_p, "r") as zf:
            img_members = sorted(
                [
                    m
                    for m in zf.namelist()
                    if Path(m).suffix.lower() in (".jpg", ".jpeg", ".png")
                    and not Path(m).name.startswith(".")
                ]
            )
            if not img_members:
                raise ValueError(f"Aucune image (.jpg/.png) trouvée dans {zip_p}")
            indices = np.linspace(0, len(img_members) - 1, min(num_views, len(img_members)), dtype=int)
            selected_members = [img_members[i] for i in indices]

            with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as out_zf:
                for i, member in enumerate(selected_members):
                    ext = Path(member).suffix.lower()
                    data = zf.read(member)
                    out_zf.writestr(f"images/frame_{i+1:04d}{ext}", data)

                if raw_ply_path and Path(raw_ply_path).exists():
                    print(f"[*] Sous-échantillonnage local du LiDAR {raw_ply_path}...")
                    pcd = o3d.io.read_point_cloud(str(raw_ply_path))
                    raw_cnt = len(pcd.points)
                    if raw_cnt > 180000:
                        pcd = pcd.voxel_down_sample(voxel_size=0.008)
                    if len(pcd.points) > 200000:
                        rng = np.random.default_rng(42)
                        pcd = pcd.select_by_index(rng.choice(len(pcd.points), size=180000, replace=False))
                    gt_tmp = tmp / f"{scene_name}_gt.ply"
                    o3d.io.write_point_cloud(str(gt_tmp), pcd)
                    out_zf.write(gt_tmp, arcname=f"{scene_name}_gt.ply")
                    print(f"[OK] LiDAR compressé : {raw_cnt:,} pts -> {len(pcd.points):,} pts")

    size_mb = out_zip.stat().st_size / (1024 * 1024)
    print(f"[OK] Archive légère pour Colab créée : {out_zip} ({size_mb:.2f} MB pour {len(selected_members)} photos HD)")
    return out_zip


def import_all_bundles(
    loader: SceneDataLoader,
    recon_mgr: ReconstructionManager,
    bundles_dir: Path = ROOT_DIR / "data" / "bundles",
) -> int:
    """Extrait automatiquement tous les bundles Colab (<Scene>_bundle.zip) présents dans data/bundles/."""
    if not bundles_dir.exists():
        raise FileNotFoundError(f"Dossier des bundles introuvable : {bundles_dir}")

    imported = 0
    for scene in loader.list_available_scenes():
        bundle_path = bundles_dir / f"{scene}_bundle.zip"
        if bundle_path.exists():
            info = recon_mgr.import_colab_bundle(bundle_path, scene)
            print(f"[OK] {bundle_path.name} -> scène '{scene}' ({info['extracted_count']} fichiers extraits).")
            imported += 1
        else:
            print(f"[!] Bundle absent pour '{scene}' ({bundle_path.name}) — ignoré.")

    if imported == 0:
        raise FileNotFoundError(f"Aucun fichier *_bundle.zip trouvé dans {bundles_dir}.")
    return imported


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Utilitaire DataOps & Reconstruction pour le projet MVS 3D.")
    parser.add_argument("--import-all", action="store_true", help="Extrait automatiquement tous les bundles de data/bundles/.")
    parser.add_argument("--import-bundle", type=str, help="Chemin vers un fichier .zip généré sur Google Colab.")
    parser.add_argument("--prepare-colab", type=str, help="Chemin vers le gros <Scene>.zip local pour créer un zip léger dans data/colab_inputs/.")
    parser.add_argument("--ply", type=str, default=None, help="Chemin optionnel vers le gros <Scene>.ply local.")
    parser.add_argument("--scene", type=str, default="Ignatius", help="Nom de la scène cible.")
    parser.add_argument("--check", action="store_true", help="Vérifie l'état des dossiers et des reconstructions.")
    args = parser.parse_args()

    data_loader = SceneDataLoader()
    recon_manager = ReconstructionManager(data_loader)

    if args.import_all:
        import_all_bundles(data_loader, recon_manager)
        check_environment(data_loader, recon_manager)
    elif args.prepare_colab:
        prepare_light_colab_input(args.prepare_colab, args.ply, args.scene)
    elif args.import_bundle:
        info = recon_manager.import_colab_bundle(args.import_bundle, args.scene)
        print(f"[OK] Bundle importé pour {info['scene']} ({info['extracted_count']} fichiers extraits).")
        check_environment(data_loader, recon_manager)
    elif args.check:
        check_environment(data_loader, recon_manager)
    else:
        parser.print_help()
