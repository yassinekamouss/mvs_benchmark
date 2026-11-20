"""
Application Principale Streamlit : Plateforme de Reconstruction 3D Dense & Benchmarking Géométrique.
Compare les reconstructions Multi-Vues (VGGT Feed-Forward vs COLMAP MVS) aux références LiDAR (Tanks and Temples).
Module : Vision par Ordinateur — FST Tanger (Cycle Ingénieur LSI 3)
Réalisé par : Yahya AHMANE & Yassine KAMOUSS — Encadrant : Pr. M'hamed AIT KBIR
"""

import json
import pandas as pd
import numpy as np
import streamlit as st
from PIL import Image

from src.loader import SceneDataLoader
from src.reconstructor import ReconstructionManager
from src.alignment import PointCloudAligner
from src.metrics import GeometryEvaluator
from src.visualizer import PointCloudVisualizer


# ============================================================================
# CONFIGURATION DE LA PAGE & STYLE CSS DARK PRO
# ============================================================================
st.set_page_config(
    page_title="MVS 3D Benchmark | VGGT vs COLMAP",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 1.2rem; padding-bottom: 1.5rem; }
    div[data-testid="stMetric"] {
        background-color: #161B22;
        border: 1px solid #30363D;
        padding: 12px 16px;
        border-radius: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================================
# INITIALISATION DES MOTEURS & CACHE STREAMLIT
# ============================================================================
@st.cache_resource
def init_engines():
    loader = SceneDataLoader("configs/config.yaml")
    recon_mgr = ReconstructionManager(loader)
    visualizer = PointCloudVisualizer(
        max_points_webgl=int(loader.config["evaluation"].get("max_points_viz_webgl", 45000))
    )
    return loader, recon_mgr, visualizer


loader, recon_mgr, visualizer = init_engines()


@st.cache_resource(show_spinner=False)
def run_cached_alignment_and_eval(
    scene_name: str,
    voxel_m: float,
    tau_mm: float,
    max_icp_iter: int,
    estimate_scale: bool,
    use_vggt_as_ref: bool = False,
):
    """Exécute et met en cache le recalage Sim(3)+ICP et l'évaluation KD-Tree pour une scène."""
    paths = loader.get_scene_paths(scene_name)
    align_cfg = dict(loader.config.get("alignment", {}))
    align_cfg["icp_max_iterations"] = max_icp_iter
    align_cfg["estimate_scale"] = estimate_scale

    aligner = PointCloudAligner(align_cfg)
    evaluator = GeometryEvaluator(loader.config.get("evaluation", {}))

    pcd_vggt_raw, stats_vggt_raw = loader.load_point_cloud(paths["vggt_ply"])
    pcd_colmap_raw, stats_colmap_raw = loader.load_point_cloud(paths["colmap_ply"])

    if use_vggt_as_ref or not paths["ground_truth_ply"].exists():
        pcd_gt, stats_gt = pcd_vggt_raw, stats_vggt_raw
    else:
        pcd_gt, stats_gt = loader.load_point_cloud(paths["ground_truth_ply"])

    # Recalage 3D sur la Référence choisie (Ground Truth LiDAR ou VGGT)
    align_vggt = aligner.align(pcd_vggt_raw, pcd_gt, voxel_size=voxel_m)
    align_colmap = aligner.align(pcd_colmap_raw, pcd_gt, voxel_size=voxel_m)

    # Évaluation quantitative : VGGT vs GT, COLMAP vs GT, et COLMAP vs VGGT (Référence)
    eval_vggt = evaluator.evaluate(
        align_vggt.aligned_cloud, pcd_gt, tau_mm=tau_mm, scene_name=scene_name, method_name="VGGT vs GT LiDAR"
    )
    eval_colmap = evaluator.evaluate(
        align_colmap.aligned_cloud, pcd_gt, tau_mm=tau_mm, scene_name=scene_name, method_name="COLMAP vs GT LiDAR"
    )
    eval_cross = evaluator.evaluate(
        align_colmap.aligned_cloud,
        align_vggt.aligned_cloud,
        tau_mm=tau_mm,
        scene_name=scene_name,
        method_name="COLMAP vs VGGT (Réf)",
    )

    return {
        "pcd_gt": pcd_gt,
        "stats_gt": stats_gt,
        "pcd_vggt_raw": pcd_vggt_raw,
        "stats_vggt_raw": stats_vggt_raw,
        "pcd_colmap_raw": pcd_colmap_raw,
        "stats_colmap_raw": stats_colmap_raw,
        "align_vggt": align_vggt,
        "align_colmap": align_colmap,
        "eval_vggt": eval_vggt,
        "eval_colmap": eval_colmap,
        "eval_cross": eval_cross,
    }


# ============================================================================
# BARRE LATÉRALE (CONTRÔLE DE LA SCÈNE & HYPERPARAMÈTRES)
# ============================================================================
with st.sidebar:
    st.markdown("## 🧊 MVS 3D Benchmark")
    st.caption("FST Tanger — Module Vision par Ordinateur\n**Yahya AHMANE & Yassine KAMOUSS**")
    st.divider()

    available_scenes = loader.list_available_scenes()
    selected_scene = st.selectbox("📂 Scène d'évaluation (Tanks & Temples)", available_scenes, index=0)
    scene_meta = loader.get_scene_metadata(selected_scene)

    reference_mode = st.selectbox(
        "🎯 Référence Géométrique Cible",
        [
            "Scan LiDAR Officiel (Tanks and Temples)",
            "VGGT pris comme Référence (Feed-Forward)",
        ],
        index=0,
        help="Permet de comparer les nuages reconstruits soit au LiDAR Tanks & Temples, soit à VGGT pris comme référence.",
    )

    st.info(f"**Description :** {scene_meta['description']}")

    st.markdown("### ⚙️ Paramètres Géométriques")
    num_views_sampled = st.slider(
        "Nombre de vues multi-angles (N)",
        min_value=5,
        max_value=50,
        value=int(scene_meta["default_num_views"]),
        step=5,
        help="Échantillonnage uniforme des images d'entrée pour la reconstruction.",
    )
    tau_mm = st.slider(
        "Seuil de tolérance F-Score τ (mm)",
        min_value=1.0,
        max_value=25.0,
        value=float(scene_meta["tau_mm"]),
        step=0.5,
        help="Seuil officiel de distance pour considérer un point comme valide.",
    )
    voxel_mm = st.slider(
        "Taille de Voxel ICP (mm)",
        min_value=2.0,
        max_value=25.0,
        value=float(scene_meta["voxel_downsample_m"] * 1000.0),
        step=1.0,
    )
    max_icp_iter = st.slider("Itérations max ICP par niveau", 20, 100, 60, 10)
    estimate_scale = st.checkbox("Activer l'estimation d'échelle Sim(3)", value=True)

    st.divider()
    st.markdown("### ☁️ Import Bundle Colab (.zip)")
    uploaded_zip = st.file_uploader("Importer une archive GPU Colab", type=["zip"])
    if uploaded_zip is not None:
        tmp_zip = loader.output_root / f"temp_{selected_scene}.zip"
        with open(tmp_zip, "wb") as f:
            f.write(uploaded_zip.getbuffer())
        info = recon_mgr.import_colab_bundle(tmp_zip, selected_scene)
        tmp_zip.unlink(missing_ok=True)
        st.success(f"Bundle importé ({info['extracted_count']} fichiers) !")
        st.cache_resource.clear()


# ============================================================================
# EN-TÊTE PRINCIPAL & EXÉCUTION DU PIPELINE
# ============================================================================
st.markdown(
    f"## 📐 Reconstruction 3D Multi-Vues & Évaluation Géométrique — Scène : `{selected_scene}`"
)

paths = loader.get_scene_paths(selected_scene)
if not paths["ground_truth_ply"].exists() and not paths["vggt_ply"].exists():
    st.error(
        f"Données manquantes pour `{selected_scene}`. Importez un bundle Colab (`<Scene>_bundle.zip`) via la barre latérale ou `python scripts/setup_data.py --import-bundle ...`."
    )
    st.stop()

use_vggt_ref = reference_mode.startswith("VGGT") or not paths["ground_truth_ply"].exists()

with st.spinner(f"Recalage Sim(3)+ICP et évaluation KD-Tree en cours sur '{selected_scene}'..."):
    bundle = run_cached_alignment_and_eval(
        scene_name=selected_scene,
        voxel_m=voxel_mm / 1000.0,
        tau_mm=tau_mm,
        max_icp_iter=max_icp_iter,
        estimate_scale=estimate_scale,
        use_vggt_as_ref=use_vggt_ref,
    )

# Définition des 4 onglets d'ingénierie
tab1, tab2, tab3, tab4 = st.tabs(
    [
        "🖼️ 1. Jeu d'Images Multi-Vues & GT",
        "🚀 2. Reconstruction 3D Dense (VGGT vs COLMAP)",
        "🧭 3. Recalage Géométrique Sim(3) + ICP",
        "📊 4. Analyse Quantitative & Heatmaps 3D",
    ]
)

# ============================================================================
# ONGLET 1 : EXPLORATION DU JEU D'IMAGES MULTI-VUES & GROUND TRUTH
# ============================================================================
with tab1:
    all_imgs = loader.get_all_images(selected_scene)
    sampled_imgs = loader.get_sampled_images(selected_scene, max_views=num_views_sampled)
    stats_gt = bundle["stats_gt"]

    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Vues Totales Disponibles", f"{len(all_imgs)} images")
    col_b.metric("Vues Échantillonnées (N)", f"{len(sampled_imgs)} vues")
    col_c.metric("Points Ground Truth (LiDAR)", f"{stats_gt.num_points:,} pts")
    col_d.metric("Échelle Scène (Diagonale GT)", f"{stats_gt.diagonal_length:.3f} m")

    st.markdown("#### 📸 Galerie d'Acquisition Multi-Vues (Échantillonnage Angulaire Uniforme)")
    if sampled_imgs:
        cols = st.columns(5)
        for idx, img_path in enumerate(sampled_imgs[:15]):
            with cols[idx % 5]:
                st.image(Image.open(img_path), caption=img_path.name, use_container_width=True)
    else:
        st.warning("Aucune image trouvée dans le dossier de la scène.")

    st.markdown("#### 📏 Propriétés Métriques du Nuage de Référence (Tanks and Temples LiDAR)")
    gt_df = pd.DataFrame(
        [
            {
                "Fichier Référence": paths["ground_truth_ply"].name,
                "Nombre de Points": f"{stats_gt.num_points:,}",
                "Dimensions X × Y × Z (m)": f"{stats_gt.extent[0]:.2f} × {stats_gt.extent[1]:.2f} × {stats_gt.extent[2]:.2f}",
                "Centroïde (m)": f"({stats_gt.centroid[0]:.2f}, {stats_gt.centroid[1]:.2f}, {stats_gt.centroid[2]:.2f})",
                "Seuil Officiel τ (mm)": f"{tau_mm:.1f} mm",
            }
        ]
    )
    st.dataframe(gt_df, use_container_width=True, hide_index=True)


# ============================================================================
# ONGLET 2 : MOTEUR DE RECONSTRUCTION 3D DENSE (VGGT vs COLMAP)
# ============================================================================
with tab2:
    res_vggt = recon_mgr.run_or_load(selected_scene, "vggt", num_views=len(sampled_imgs))
    res_colmap = recon_mgr.run_or_load(selected_scene, "colmap", num_views=len(sampled_imgs))

    st.markdown("#### ⚡ Comparaison des Paradigmes de Reconstruction 3D Dense")
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("### 🧠 Backend 1 : VGGT (Feed-Forward Transformer)")
        st.caption("Prédiction directe en une passe avant sans calibrage préalable (CVPR 2025).")
        m1, m2, m3 = st.columns(3)
        m1.metric("Temps d'Inférence", f"{res_vggt.execution_time_sec:.2f} s")
        m2.metric("Points Générés", f"{res_vggt.num_points:,}")
        m3.metric("Diagonale Brute", f"{bundle['stats_vggt_raw'].diagonal_length:.3f} m")
        st.info(f"**Matériel :** {res_vggt.device_name}\n\n**Note :** {res_vggt.notes}")

    with c2:
        st.markdown("### 📐 Backend 2 : COLMAP (SfM + PatchMatch MVS)")
        st.caption("Extraction SIFT, Ajustement de Faisceaux et Stéréo Multi-Vues par PatchMatch.")
        m4, m5, m6 = st.columns(3)
        m4.metric(
            "Temps de Calcul",
            f"{res_colmap.execution_time_sec:.1f} s",
            delta=f"{res_colmap.execution_time_sec / max(res_vggt.execution_time_sec, 0.1):.0f}x plus lent",
            delta_color="inverse",
        )
        m5.metric("Points Fusionnés", f"{res_colmap.num_points:,}")
        m6.metric("Diagonale Brute", f"{bundle['stats_colmap_raw'].diagonal_length:.3f} m")
        st.info(f"**Matériel :** {res_colmap.device_name}\n\n**Note :** {res_colmap.notes}")


# ============================================================================
# ONGLET 3 : LABORATOIRE DE RECALAGE GÉOMÉTRIQUE (Sim(3) + Multi-Scale ICP)
# ============================================================================
with tab3:
    st.markdown("#### 🧭 Recalage Spatial & Mise à l'Échelle Métrique ($Sim(3)$ + Pyramide ICP)")
    method_align_view = st.radio(
        "Sélectionner le nuage reconstruit à inspecter :",
        ["VGGT (Feed-Forward)", "COLMAP (PatchMatch MVS)"],
        horizontal=True,
    )
    show_raw_cloud = st.checkbox("Afficher également le nuage brut avant recalage (en rouge)", value=False)

    is_vggt = method_align_view.startswith("VGGT")
    align_res = bundle["align_vggt"] if is_vggt else bundle["align_colmap"]
    raw_cloud = bundle["pcd_vggt_raw"] if is_vggt else bundle["pcd_colmap_raw"]

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Facteur d'Échelle Estimé (s)", f"{align_res.estimated_scale:.4f}")
    k2.metric(
        "Recouvrement (Fitness ICP)",
        f"{align_res.fitness_after * 100:.1f} %",
        delta=f"+{(align_res.fitness_after - align_res.fitness_before) * 100:.1f} %",
    )
    k3.metric(
        "Erreur Inlier RMSE Finale",
        f"{align_res.inlier_rmse_after_mm:.2f} mm",
        delta=f"{align_res.inlier_rmse_after_mm - align_res.inlier_rmse_before_mm:.2f} mm",
        delta_color="inverse",
    )
    k4.metric("Temps de Recalage CPU", f"{align_res.execution_time_sec:.3f} s")

    col_viz, col_mat = st.columns([2.2, 1.0])
    with col_viz:
        fig_align = visualizer.create_alignment_comparison_3d(
            raw_pcd=raw_cloud,
            aligned_pcd=align_res.aligned_cloud,
            gt_pcd=bundle["pcd_gt"],
            show_raw=show_raw_cloud,
        )
        st.plotly_chart(fig_align, use_container_width=True)

    with col_mat:
        st.markdown(r"##### Matrice Homogène $T \in Sim(3)$ ($4 \times 4$)")
        mat_df = pd.DataFrame(
            np.round(align_res.transformation_matrix, 4),
            columns=["X", "Y", "Z", "Translation (m)"],
            index=["Ligne 1", "Ligne 2", "Ligne 3", "Homogène"],
        )
        st.dataframe(mat_df, use_container_width=True)
        st.caption(
            "La sous-matrice $3 \\times 3$ supérieure gauche encode $s \\cdot R$ "
            "(facteur d'échelle isotrope et rotation orthogonale $SO(3)$). "
            "La 4ème colonne encode le vecteur de translation $t$ en mètres."
        )


# ============================================================================
# ONGLET 4 : ANALYSE QUANTITATIVE, HEATMAPS 3D & EXPORT RAPPORT
# ============================================================================
with tab4:
    eval_vggt = bundle["eval_vggt"]
    eval_colmap = bundle["eval_colmap"]
    eval_cross = bundle["eval_cross"]

    st.markdown(f"#### 🏆 Tableau de Bord Comparatif des Métriques (Seuil officiel $\\tau = {tau_mm}\\text{{ mm}}$)")

    comp_df = pd.DataFrame(
        [
            {
                "Comparaison": eval_vggt.method_name,
                "Précision / Accuracy (mm) ↓": eval_vggt.accuracy_mean_mm,
                "Complétude / Completeness (mm) ↓": eval_vggt.completeness_mean_mm,
                "Distance de Chamfer (mm) ↓": eval_vggt.chamfer_distance_mm,
                "Hausdorff 95% (mm) ↓": eval_vggt.hausdorff_95_mm,
                "Précision @ τ (%) ↑": eval_vggt.precision_pct,
                "Rappel @ τ (%) ↑": eval_vggt.recall_pct,
                "F-Score @ τ (%) ↑": eval_vggt.fscore_pct,
            },
            {
                "Comparaison": eval_colmap.method_name,
                "Précision / Accuracy (mm) ↓": eval_colmap.accuracy_mean_mm,
                "Complétude / Completeness (mm) ↓": eval_colmap.completeness_mean_mm,
                "Distance de Chamfer (mm) ↓": eval_colmap.chamfer_distance_mm,
                "Hausdorff 95% (mm) ↓": eval_colmap.hausdorff_95_mm,
                "Précision @ τ (%) ↑": eval_colmap.precision_pct,
                "Rappel @ τ (%) ↑": eval_colmap.recall_pct,
                "F-Score @ τ (%) ↑": eval_colmap.fscore_pct,
            },
            {
                "Comparaison": eval_cross.method_name,
                "Précision / Accuracy (mm) ↓": eval_cross.accuracy_mean_mm,
                "Complétude / Completeness (mm) ↓": eval_cross.completeness_mean_mm,
                "Distance de Chamfer (mm) ↓": eval_cross.chamfer_distance_mm,
                "Hausdorff 95% (mm) ↓": eval_cross.hausdorff_95_mm,
                "Précision @ τ (%) ↑": eval_cross.precision_pct,
                "Rappel @ τ (%) ↑": eval_cross.recall_pct,
                "F-Score @ τ (%) ↑": eval_cross.fscore_pct,
            },
        ]
    )
    st.dataframe(comp_df, use_container_width=True, hide_index=True)

    st.divider()
    st.markdown("#### 🔥 Visualisation Scientifique : Carte Thermique 3D des Erreurs Locales")

    hm_col1, hm_col2, hm_col3 = st.columns([1.5, 1.5, 1.0])
    with hm_col1:
        heatmap_choice = st.selectbox(
            "Nuage à inspecter en Heatmap 3D :",
            [
                "VGGT évalué vs Ground Truth LiDAR",
                "COLMAP évalué vs Ground Truth LiDAR",
                "COLMAP évalué vs VGGT (Référence)",
            ],
        )
    with hm_col2:
        show_ghost_gt = st.checkbox("Superposer la référence en filigrane (Fantôme)", value=True)

    if heatmap_choice.startswith("VGGT"):
        active_pcd = bundle["align_vggt"].aligned_cloud
        active_err = eval_vggt.dist_recon_to_gt_mm
        ref_pcd = bundle["pcd_gt"]
    elif "Ground Truth" in heatmap_choice:
        active_pcd = bundle["align_colmap"].aligned_cloud
        active_err = eval_colmap.dist_recon_to_gt_mm
        ref_pcd = bundle["pcd_gt"]
    else:
        active_pcd = bundle["align_colmap"].aligned_cloud
        active_err = eval_cross.dist_recon_to_gt_mm
        ref_pcd = bundle["align_vggt"].aligned_cloud

    with hm_col3:
        st.write("")
        if st.button("🖥️ Ouvrir dans Open3D Natif (Desktop)"):
            visualizer.launch_native_open3d_viewer(
                active_pcd, active_err, tau_mm=tau_mm, window_title=f"{selected_scene} - {heatmap_choice}"
            )

    fig_hm = visualizer.create_error_heatmap_3d(
        aligned_pcd=active_pcd,
        errors_mm=active_err,
        tau_mm=tau_mm,
        gt_pcd=ref_pcd if show_ghost_gt else None,
        title=f"Scène {selected_scene} — {heatmap_choice} (Seuil τ = {tau_mm} mm)",
    )
    st.plotly_chart(fig_hm, use_container_width=True)

    st.divider()
    st.markdown("#### 📈 Courbes Multi-Seuils (F-Score, Précision, Rappel) & Distributions Statistiques")
    metrics_to_plot = [eval_vggt, eval_colmap, eval_cross]
    st.plotly_chart(visualizer.create_fscore_curves_plot(metrics_to_plot, tau_mm), use_container_width=True)
    st.plotly_chart(visualizer.create_error_histogram_plot(metrics_to_plot, tau_mm), use_container_width=True)

    # Export des résultats pour le rapport final
    st.divider()
    exp1, exp2 = st.columns(2)
    csv_bytes = comp_df.to_csv(index=False).encode("utf-8")
    exp1.download_button(
        "📥 Télécharger le Tableau Comparatif (.CSV)",
        data=csv_bytes,
        file_name=f"benchmark_{selected_scene}.csv",
        mime="text/csv",
    )
    json_bytes = json.dumps(
        {
            "vggt_vs_gt": eval_vggt.to_summary_dict(),
            "colmap_vs_gt": eval_colmap.to_summary_dict(),
            "colmap_vs_vggt": eval_cross.to_summary_dict(),
        },
        indent=4,
    ).encode("utf-8")
    exp2.download_button(
        "📥 Télécharger le Rapport Complet des Métriques (.JSON)",
        data=json_bytes,
        file_name=f"metrics_{selected_scene}.json",
        mime="application/json",
    )
