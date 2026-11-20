"""
Module de visualisation scientifique 3D (Plotly WebGL & Open3D) et de graphiques analytiques.
Génère les Heatmaps 3D d'erreur géométrique, les superpositions de recalage et les courbes F-Score.
Auteurs : Yahya AHMANE & Yassine KAMOUSS
"""

from typing import List, Optional
import numpy as np
import open3d as o3d
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.metrics import EvaluationMetrics


class PointCloudVisualizer:
    """
    Générateur de visualisations 3D interactives (WebGL) et de graphiques de performance
    en thème sombre professionnel (Dark Mode).
    """

    def __init__(self, max_points_webgl: int = 45000) -> None:
        self.max_points_webgl = max_points_webgl

    def _subsample_for_viz(
        self, points: np.ndarray, values: Optional[np.ndarray] = None, seed: int = 42
    ) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """Sous-échantillonne un nuage de points pour garantir un rendu fluide à 60 FPS dans le navigateur."""
        n = len(points)
        if n <= self.max_points_webgl:
            return points, values
        rng = np.random.default_rng(seed=seed)
        idx = rng.choice(n, size=self.max_points_webgl, replace=False)
        return points[idx], (values[idx] if values is not None else None)

    def create_alignment_comparison_3d(
        self,
        raw_pcd: o3d.geometry.PointCloud,
        aligned_pcd: o3d.geometry.PointCloud,
        gt_pcd: o3d.geometry.PointCloud,
        show_raw: bool = False,
    ) -> go.Figure:
        """
        Crée une vue 3D interactive superposant la Vérité Terrain (Vert/Cyan),
        le nuage Recalé Sim(3)+ICP (Orange/Or) et optionnellement le nuage Brut non recalé (Rouge/Violet).
        """
        pts_gt, _ = self._subsample_for_viz(np.asarray(gt_pcd.points))
        pts_al, _ = self._subsample_for_viz(np.asarray(aligned_pcd.points))

        traces = [
            go.Scatter3d(
                x=pts_gt[:, 0],
                y=pts_gt[:, 1],
                z=pts_gt[:, 2],
                mode="markers",
                name="Vérité Terrain (LiDAR GT)",
                marker=dict(size=1.8, color="#00E5FF", opacity=0.45),
            ),
            go.Scatter3d(
                x=pts_al[:, 0],
                y=pts_al[:, 1],
                z=pts_al[:, 2],
                mode="markers",
                name="Nuage Recalé (Sim(3) + ICP)",
                marker=dict(size=2.0, color="#FF9100", opacity=0.80),
            ),
        ]

        if show_raw:
            pts_raw, _ = self._subsample_for_viz(np.asarray(raw_pcd.points))
            traces.append(
                go.Scatter3d(
                    x=pts_raw[:, 0],
                    y=pts_raw[:, 1],
                    z=pts_raw[:, 2],
                    mode="markers",
                    name="Nuage Brut Initial (Avant Recalage)",
                    marker=dict(size=1.8, color="#FF1744", opacity=0.55),
                )
            )

        fig = go.Figure(data=traces)
        fig.update_layout(
            template="plotly_dark",
            margin=dict(l=0, r=0, b=0, t=35),
            height=580,
            legend=dict(x=0.02, y=0.98, bgcolor="rgba(15,20,30,0.8)"),
            scene=dict(
                xaxis_title="X (m)",
                yaxis_title="Y (m)",
                zaxis_title="Z (m)",
                aspectmode="data",
                bgcolor="rgb(14, 17, 23)",
            ),
        )
        return fig

    def create_error_heatmap_3d(
        self,
        aligned_pcd: o3d.geometry.PointCloud,
        errors_mm: np.ndarray,
        tau_mm: float,
        gt_pcd: Optional[o3d.geometry.PointCloud] = None,
        title: str = "Heatmap 3D d'Erreur Géométrique",
    ) -> go.Figure:
        """
        Génère une carte thermique 3D où chaque point reconstruit est colorisé selon sa distance
        au plus proche voisin de la référence (échelle bornée entre 0 et 2 * tau_mm).
        """
        pts = np.asarray(aligned_pcd.points)
        pts_sub, err_sub = self._subsample_for_viz(pts, errors_mm)
        cmax = max(float(tau_mm) * 2.5, 5.0)

        traces = [
            go.Scatter3d(
                x=pts_sub[:, 0],
                y=pts_sub[:, 1],
                z=pts_sub[:, 2],
                mode="markers",
                name="Erreur locale d(p, GT)",
                text=[f"Erreur: {e:.2f} mm" for e in err_sub],
                hoverinfo="text+x+y+z",
                marker=dict(
                    size=2.2,
                    color=np.clip(err_sub, 0.0, cmax),
                    colorscale="Turbo",
                    cmin=0.0,
                    cmax=cmax,
                    opacity=0.9,
                    colorbar=dict(
                        title=f"Erreur (mm)<br>[τ={tau_mm}mm]",
                        thickness=18,
                        len=0.75,
                    ),
                ),
            )
        ]

        if gt_pcd is not None:
            pts_gt, _ = self._subsample_for_viz(np.asarray(gt_pcd.points))
            traces.insert(
                0,
                go.Scatter3d(
                    x=pts_gt[:, 0],
                    y=pts_gt[:, 1],
                    z=pts_gt[:, 2],
                    mode="markers",
                    name="Référence (Fantôme)",
                    marker=dict(size=1.3, color="#90A4AE", opacity=0.18),
                ),
            )

        fig = go.Figure(data=traces)
        fig.update_layout(
            title=dict(text=title, x=0.02, font=dict(size=15)),
            template="plotly_dark",
            margin=dict(l=0, r=0, b=0, t=40),
            height=600,
            legend=dict(x=0.02, y=0.96, bgcolor="rgba(15,20,30,0.8)"),
            scene=dict(
                xaxis_title="X (m)",
                yaxis_title="Y (m)",
                zaxis_title="Z (m)",
                aspectmode="data",
                bgcolor="rgb(14, 17, 23)",
            ),
        )
        return fig

    @staticmethod
    def create_fscore_curves_plot(metrics_list: List[EvaluationMetrics], nominal_tau_mm: float) -> go.Figure:
        """
        Trace les courbes comparatives F-Score, Précision et Rappel en fonction du seuil de tolérance τ (mm).
        """
        fig = make_subplots(
            rows=1,
            cols=3,
            subplot_titles=("F-Score (%) vs Seuil τ", "Précision (%) vs Seuil τ", "Rappel / Complétude (%) vs Seuil τ"),
        )
        palette = ["#2979FF", "#FF9100", "#00E676"]

        for idx, m in enumerate(metrics_list):
            color = palette[idx % len(palette)]
            fig.add_trace(
                go.Scatter(
                    x=m.curve_thresholds_mm,
                    y=m.curve_fscore_pct,
                    mode="lines+markers",
                    name=f"{m.method_name} (F-Score)",
                    line=dict(color=color, width=3),
                ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=m.curve_thresholds_mm,
                    y=m.curve_precision_pct,
                    mode="lines+markers",
                    name=f"{m.method_name} (Précision)",
                    line=dict(color=color, width=2, dash="solid"),
                    showlegend=False,
                ),
                row=1,
                col=2,
            )
            fig.add_trace(
                go.Scatter(
                    x=m.curve_thresholds_mm,
                    y=m.curve_recall_pct,
                    mode="lines+markers",
                    name=f"{m.method_name} (Rappel)",
                    line=dict(color=color, width=2, dash="dot"),
                    showlegend=False,
                ),
                row=1,
                col=3,
            )

        # Ligne verticale indiquant le seuil officiel tau_mm de la scène
        for col_idx in (1, 2, 3):
            fig.add_vline(
                x=nominal_tau_mm,
                line_width=1.5,
                line_dash="dash",
                line_color="#FF5252",
                annotation_text=f"τ={nominal_tau_mm}mm",
                row=1,
                col=col_idx,
            )
            fig.update_xaxes(title_text="Seuil de tolérance τ (mm)", row=1, col=col_idx)
            fig.update_yaxes(title_text="Score (%)", range=[0, 105], row=1, col=col_idx)

        fig.update_layout(
            template="plotly_dark",
            height=420,
            margin=dict(l=20, r=20, t=45, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=-0.28, xanchor="center", x=0.5),
        )
        return fig

    @staticmethod
    def create_error_histogram_plot(metrics_list: List[EvaluationMetrics], nominal_tau_mm: float) -> go.Figure:
        """
        Trace la distribution statistique des distances d'erreur (Précision et Complétude) en millimètres.
        """
        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=(
                "Distribution de l'Erreur de Précision d(P → G)",
                "Distribution de l'Erreur de Complétude d(G → P)",
            ),
        )
        palette = ["#2979FF", "#FF9100", "#00E676"]
        max_x = nominal_tau_mm * 4.0

        for idx, m in enumerate(metrics_list):
            color = palette[idx % len(palette)]
            d_acc = m.dist_recon_to_gt_mm[m.dist_recon_to_gt_mm <= max_x]
            d_comp = m.dist_gt_to_recon_mm[m.dist_gt_to_recon_mm <= max_x]

            fig.add_trace(
                go.Histogram(
                    x=d_acc,
                    nbinsx=60,
                    name=m.method_name,
                    marker_color=color,
                    opacity=0.65,
                    histnorm="percent",
                ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Histogram(
                    x=d_comp,
                    nbinsx=60,
                    name=m.method_name,
                    marker_color=color,
                    opacity=0.65,
                    histnorm="percent",
                    showlegend=False,
                ),
                row=1,
                col=2,
            )

        for col_idx in (1, 2):
            fig.add_vline(
                x=nominal_tau_mm,
                line_width=2,
                line_dash="dash",
                line_color="#FF5252",
                annotation_text=f"Seuil τ = {nominal_tau_mm} mm",
                row=1,
                col=col_idx,
            )
            fig.update_xaxes(title_text="Distance d'erreur (mm)", row=1, col=col_idx)
            fig.update_yaxes(title_text="Proportion des points (%)", row=1, col=col_idx)

        fig.update_layout(
            barmode="overlay",
            template="plotly_dark",
            height=400,
            margin=dict(l=20, r=20, t=45, b=20),
        )
        return fig

    @staticmethod
    def launch_native_open3d_viewer(
        aligned_pcd: o3d.geometry.PointCloud, errors_mm: np.ndarray, tau_mm: float, window_title: str = "Open3D Viewer"
    ) -> None:
        """
        Ouvre une fenêtre native Open3D C++ accélérée matériellement sur le bureau Linux
        avec le nuage complet colorisé par erreur thermique.
        """
        import matplotlib

        pcd_viz = o3d.geometry.PointCloud(aligned_pcd)
        norm_err = np.clip(errors_mm / max(tau_mm * 2.5, 1e-6), 0.0, 1.0)
        colormap = matplotlib.colormaps["turbo"]
        colors_rgb = colormap(norm_err)[:, :3]
        pcd_viz.colors = o3d.utility.Vector3dVector(colors_rgb)
        o3d.visualization.draw_geometries([pcd_viz], window_name=window_title, width=1280, height=800)
