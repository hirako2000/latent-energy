import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
from scipy.ndimage import gaussian_filter
import plotly.graph_objects as go
from pathlib import Path
from typing import Optional, cast, Any, List, Tuple
from rich.console import Console
from mpl_toolkits.mplot3d import Axes3D

from grid_energy.core.energy import NonogramEnergy
from grid_energy.core.models import NonogramCNN
from grid_energy.utils.config import settings

console = Console()

class MultiCellEnergyVisualizer:
    
    def __init__(
        self,
        model_path: Optional[Path] = None,
        output_dir: Optional[Path] = None
    ):
        self.model_path = model_path or settings.ROOT_DIR / "data/models/ebm_best.pt"
        self.output_dir = output_dir or settings.ROOT_DIR / "docs" / "energy_atlas"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load_model()
        self.setup_test_puzzle()
    
    def load_model(self):
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found at {self.model_path}")
        
        self.model = NonogramCNN(grid_size=5, hint_dim=20).to(self.device)
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model.eval()
        
        self.energy_fn = NonogramEnergy(self.model).to(self.device)
    
    def setup_test_puzzle(self):
        self.target_grid = torch.tensor([
            [1, 0, 1, 0, 1],
            [0, 1, 0, 1, 0],
            [1, 0, 1, 0, 1],
            [0, 1, 0, 1, 0],
            [1, 0, 1, 0, 1]
        ], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
        
        row_hints = torch.tensor([
            [3, 0], [2, 0], [3, 0], [2, 0], [3, 0]
        ], dtype=torch.float32).unsqueeze(0)
        
        col_hints = torch.tensor([
            [3, 0], [2, 0], [3, 0], [2, 0], [3, 0]
        ], dtype=torch.float32).unsqueeze(0)
        
        self.hints = torch.stack([row_hints, col_hints], dim=1).to(self.device)
        self.energy_fn.set_context(self.hints)
    
    def sample_2d_slice(self, center: torch.Tensor, dim1: int, dim2: int, steps: int = 40):
        x = np.linspace(-2, 2, steps)
        y = np.linspace(-2, 2, steps)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros((steps, steps))
        
        row1, col1 = divmod(dim1, 5)
        row2, col2 = divmod(dim2, 5)
        
        with torch.no_grad():
            for i in range(steps):
                for j in range(steps):
                    point = center.clone()
                    point[0, 0, row1, col1] = X[i, j]
                    point[0, 0, row2, col2] = Y[i, j]
                    point = point.to(self.device)
                    Z[i, j] = self.energy_fn(point).item()
        
        return X, Y, Z
    
    def create_energy_atlas_3x3(self):
        cell_pairs = [
            (0, 1), (0, 5), (0, 6), (1, 2), (5, 6), (0, 12), (6, 7), (12, 13), (18, 19)
        ]
        
        fig = plt.figure(figsize=(20, 15))
        plt.suptitle('Energy Landscape Atlas: Multiple Cell Interactions',
                    fontsize=20, fontweight='bold', y=0.95)
        
        cmap_viridis = colormaps["viridis"]
        
        for idx, (dim1, dim2) in enumerate(cell_pairs[:9]):
            row1, col1 = divmod(dim1, 5)
            row2, col2 = divmod(dim2, 5)
            
            X, Y, Z = self.sample_2d_slice(self.target_grid, dim1, dim2, steps=30)
            Z_smooth = gaussian_filter(Z, sigma=1.0)
            
            ax = cast(Axes3D, fig.add_subplot(3, 3, idx + 1, projection='3d'))
            ax.plot_surface(X, Y, Z_smooth, cmap=cmap_viridis,
                           alpha=0.8, linewidth=0.5, antialiased=True)
            
            ax.set_title(f'({row1},{col1}) vs ({row2},{col2})', fontsize=10, pad=0)
            ax.set_xlabel(f'Cell {dim1}', fontsize=8, labelpad=5)
            ax.set_ylabel(f'Cell {dim2}', fontsize=8, labelpad=5)
            ax.set_zlabel('Energy', fontsize=8, labelpad=5)
            ax.view_init(elev=30, azim=45)
            
            energy_min: float = float(np.min(Z_smooth))
            energy_max: float = float(np.max(Z_smooth))
            ax.text2D(0.05, 0.95, f'{energy_min:.1f}-{energy_max:.1f}',
                     transform=ax.transAxes, fontsize=7,
                     bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.7})
        
        plt.tight_layout(rect=(0, 0, 1, 0.93))
        filepath = self.output_dir / "energy_atlas_3x3.png"
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        return filepath
    
    def create_main_with_satellites(self):
        main_pair = (0, 1)
        satellite_pairs = [(0, 5), (1, 6), (0, 6), (5, 6)]
        cmap_viridis = colormaps["viridis"]
        
        fig = plt.figure(figsize=(18, 12))
        ax_main = cast(Axes3D, fig.add_subplot(2, 3, (1, 4), projection='3d'))
        
        X_main, Y_main, Z_main = self.sample_2d_slice(self.target_grid, *main_pair, steps=40)
        Z_main_smooth = gaussian_filter(Z_main, sigma=1.5)
        
        surf_main = ax_main.plot_surface(X_main, Y_main, Z_main_smooth,
                                        cmap=cmap_viridis, alpha=0.85,
                                        linewidth=0.3, antialiased=True)
        
        row1, col1 = divmod(main_pair[0], 5)
        row2, col2 = divmod(main_pair[1], 5)
        ax_main.set_title(f'MAIN: Cells ({row1},{col1}) vs ({row2},{col2})',
                         fontsize=10, fontweight='normal', pad=15)
        ax_main.set_xlabel(f'Cell {main_pair[0]}', fontsize=11, labelpad=10)
        ax_main.set_ylabel(f'Cell {main_pair[1]}', fontsize=11, labelpad=10)
        ax_main.set_zlabel('Energy', fontsize=11, labelpad=10)
        ax_main.view_init(elev=25, azim=45)
        
        fig.colorbar(surf_main, ax=ax_main, shrink=0.6, aspect=20, pad=0.1)
        
        for idx, (dim1, dim2) in enumerate(satellite_pairs):
            ax = cast(Axes3D, fig.add_subplot(2, 3, idx + 2, projection='3d'))
            X, Y, Z = self.sample_2d_slice(self.target_grid, dim1, dim2, steps=25)
            Z_smooth = gaussian_filter(Z, sigma=1.0)
            
            row1, col1 = divmod(dim1, 5)
            row2, col2 = divmod(dim2, 5)
            
            ax.plot_surface(X, Y, Z_smooth, cmap=cmap_viridis,
                           alpha=0.7, linewidth=0.2, antialiased=True)
            
            ax.set_title(f'({row1},{col1}) vs ({row2},{col2})', fontsize=10, pad=5)
            ax.set_xlabel(f'C{dim1}', fontsize=8, labelpad=5)
            ax.set_ylabel(f'C{dim2}', fontsize=8, labelpad=5)
            ax.set_zlabel('E', fontsize=8, labelpad=5)
            ax.view_init(elev=30, azim=45)
            ax.set_xlim(-1.5, 1.5)
            ax.set_ylim(-1.5, 1.5)
        
        plt.suptitle('Energy Landscape: Main View with Satellite Comparisons',
                    fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout(rect=(0, 0, 1, 0.96))
        filepath = self.output_dir / "main_with_satellites.png"
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        return filepath
    
    def create_cell_neighborhood_map(self):
        fig = plt.figure(figsize=(16, 12))
        center_cells = [6, 7, 8, 11, 12, 13, 16, 17, 18]
        cmap_plasma = colormaps["plasma"]
        
        for idx, cell in enumerate(center_cells):
            row, col = divmod(cell, 5)
            if col == 4:
                continue
                
            neighbor = cell + 1
            ax = cast(Axes3D, fig.add_subplot(3, 3, idx + 1, projection='3d'))
            X, Y, Z = self.sample_2d_slice(self.target_grid, cell, neighbor, steps=25)
            Z_smooth = gaussian_filter(Z, sigma=1.0)
            
            ax.plot_surface(X, Y, Z_smooth, cmap=cmap_plasma,
                           alpha=0.8, linewidth=0.3, antialiased=True)
            
            ax.set_title(f'({row},{col})⇄({row},{col+1})', fontsize=9, pad=2)
            ax.set_xlabel('', fontsize=6)
            ax.set_ylabel('', fontsize=6)
            
            cast(Any, ax).set_xticks([])
            cast(Any, ax).set_yticks([])
            cast(Any, ax).set_zticks([])
            ax.view_init(elev=25, azim=45)
            
            e_min: float = float(np.min(Z_smooth))
            e_max: float = float(np.max(Z_smooth))
            ax.text2D(0.05, 0.95, f'{e_min:.1f}-{e_max:.1f}',
                     transform=ax.transAxes, fontsize=6,
                     bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "alpha": 0.7})
        
        plt.suptitle('Cell Neighborhood Interactions: Horizontal Adjacencies',
                    fontsize=16, fontweight='bold', y=0.95)
        plt.tight_layout(rect=(0, 0, 1, 0.93))
        filepath = self.output_dir / "cell_neighborhood_map.png"
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        return filepath
    
    def create_interactive_comparison_dashboard(self):
        try:
            comparison_pairs = [
                (0, 1, "Horizontal Neighbors (0,0)-(0,1)"),
                (0, 5, "Vertical Neighbors (0,0)-(1,0)"),
                (0, 6, "Diagonal Neighbors (0,0)-(1,1)"),
                (12, 13, "Center Region (2,2)-(2,3)"),
                (0, 12, "Long Distance (0,0)-(2,2)")
            ]
            
            plot_divs = []
            colorscales = ['Viridis', 'Plasma', 'Magma', 'Cividis', 'Inferno']
            energy_ranges: List[Tuple[str, float, float]] = []
            
            for idx, (dim1, dim2, title) in enumerate(comparison_pairs):
                X, Y, Z = self.sample_2d_slice(self.target_grid, dim1, dim2, steps=30)
                Z_smooth = gaussian_filter(Z, sigma=1.0)
                
                cur_min: float = float(np.min(Z_smooth))
                cur_max: float = float(np.max(Z_smooth))
                energy_ranges.append((title, cur_min, cur_max))
                
                fig = go.Figure(data=[go.Surface(
                    x=X, y=Y, z=Z_smooth,
                    colorscale=colorscales[idx],
                    colorbar={"thickness": 15, "len": 0.5}
                )])
                
                fig.update_layout(
                    title={"text": f"<b>{title}</b>", "font": {"size": 11}, "y": 0.95},
                    margin={"l": 0, "r": 0, "b": 0, "t": 40},
                    scene={"aspectmode": "cube", "xaxis_title": "X", "yaxis_title": "Y", "zaxis_title": "E"},
                    height=450,
                    autosize=True
                )
                plot_divs.append(fig.to_html(full_html=False, include_plotlyjs=False, div_id=f"plot_{idx}"))

            bar_fig = go.Figure(data=[go.Bar(
                x=[str(name) for name, _, _ in energy_ranges],
                y=[float(mx) - float(mn) for _, mn, mx in energy_ranges],
                text=[f"{float(mn):.1f}-{float(mx):.1f}" for _, mn, mx in energy_ranges],
                marker_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
            )])
            bar_fig.update_layout(
                title="Energy Range Delta (Max - Min)",
                margin={"l": 40, "r": 20, "b": 100, "t": 40},
                height=450,
                autosize=True
            )
            plot_divs.append(bar_fig.to_html(full_html=False, include_plotlyjs=False, div_id="plot_bar"))

            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="utf-8" name="viewport" content="width=device-width, initial-scale=1">
                <title>Energy Atlas Dashboard</title>
                <script src="https://cdn.plot.ly/plotly-3.1.1.min.js"></script>
                <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;700&display=swap" rel="stylesheet">
                <style>
                    .modebar-group > a {{ display: none !important; }}
                    body {{ font-family: "IBM Plex Mono", monospace; margin: 0.625em; padding: 1.225em; }}
                    .header {{ text-align: center; margin-bottom: 30px; }}
                    .dashboard-grid {{
                        display: grid;
                        grid-template-columns: 1fr;
                        gap: 20px;
                        max-width: 1600px;
                        margin: 0 auto;
                    }}
                    @media (min-width: 768px) {{
                        .dashboard-grid {{ grid-template-columns: repeat(2, 1fr); }}
                    }}
                    @media (min-width: 1200px) {{
                        .dashboard-grid {{ grid-template-columns: repeat(3, 1fr); }}
                    }}
                    .plot-card {{
                        background: white;
                        border-radius: 12px;
                        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
                        padding: 10px;
                        min-height: 460px;
                        display: flex;
                        flex-direction: column;
                    }}
                    footer {{ margin-top: 1.225em; }}
                </style>
            </head>
            <body>
                <div class="header">
                    <h2>🌋 Interactive Energy Landscape Dashboard</h2>
                    <p>Cell interaction analysis for Nonogram EBM</p>
                </div>
                <div class="dashboard-grid">
                    {''.join([f'<div class="plot-card">{div}</div>' for div in plot_divs])}
                </div>
                <script>
                    window.addEventListener('resize', function() {{
                        const plots = document.querySelectorAll('.js-plotly-plot');
                        plots.forEach(p => Plotly.Plots.resize(p));
                    }});
                </script>
            </body>
            <footer>Made with <a href="https://plotly.com/">Plotly</a></footer>
            </html>
            """
            
            filepath = self.output_dir / "interactive_comparison_dashboard.html"
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(html_content)
            return filepath
        except Exception as e:
            console.print(f"[yellow]Interactive dashboard failed: {e}[/yellow]")
            return None
    
    def create_energy_gradient_field_overview(self):
        cell_pairs = [(0, 1), (0, 5), (1, 6), (12, 13)]
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes_flat: List[Any] = list(axes.flatten())
        cmap_viridis = colormaps["viridis"]
        
        for idx, (dim1, dim2) in enumerate(cell_pairs):
            ax = axes_flat[idx]
            X, Y, Z = self.sample_2d_slice(self.target_grid, dim1, dim2, steps=20)
            Z_smooth = gaussian_filter(Z, sigma=0.5)
            grad_x, grad_y = np.gradient(Z_smooth)
            
            ax.contourf(X, Y, Z_smooth, 15, cmap=cmap_viridis, alpha=0.7)
            ax.quiver(X[::2, ::2], Y[::2, ::2],
                     -grad_x[::2, ::2], -grad_y[::2, ::2], # pyright: ignore [reportIndexIssue]
                     scale=20, color='red', alpha=0.6)
            
            row1, col1 = divmod(dim1, 5)
            row2, col2 = divmod(dim2, 5)
            ax.set_title(f'Gradient Field: ({row1},{col1})-({row2},{col2})', fontsize=11)
            ax.set_xlabel(f'Cell {dim1}')
            ax.set_ylabel(f'Cell {dim2}')
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('Gradient Field Comparison Across Different Cell Pairs',
                    fontsize=14, fontweight='bold')
        plt.tight_layout(rect=(0, 0, 1, 0.96))
        filepath = self.output_dir / "gradient_field_overview.png"
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        return filepath
    
    def generate_comprehensive_atlas(self):
        console.print("[bold cyan]Generating Multi-Cell Energy Atlas...[/bold cyan]")
        atlas_3x3 = self.create_energy_atlas_3x3()
        main_satellites = self.create_main_with_satellites()
        neighborhood = self.create_cell_neighborhood_map()
        gradients = self.create_energy_gradient_field_overview()
        dashboard = self.create_interactive_comparison_dashboard()
        
        self.create_atlas_index([atlas_3x3, main_satellites, neighborhood, gradients, dashboard])
        console.print(f"\n[green]✓ Multi-cell atlas saved to: {self.output_dir}[/green]")
        
        return {
            'atlas_3x3': atlas_3x3,
            'main_satellites': main_satellites,
            'neighborhood': neighborhood,
            'gradients': gradients,
            'dashboard': dashboard
        }
    
    def create_atlas_index(self, filepaths):
        index_path = self.output_dir / "index.html"
        with open(index_path, 'w') as f:
            f.write('''
<!DOCTYPE html>
<html>
<head>
    <title>Energy Landscape Atlas</title>
    <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;700&display=swap" rel="stylesheet">
    <style>
        body { font-family: "IBM Plex Mono", monospace; margin: 0.825em; }
        .header { text-align: center; margin-bottom: 2em; padding: 1em;border-radius: 0.625em; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
        .card { background: white; border-radius: 10px; padding: 20px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); transition: transform 0.3s; }
        .card:hover { transform: translateY(-5px); }
        .card img { width: 100%; height: 200px; object-fit: cover; border-radius: 5px; }
        .card h3 { margin-top: 0; color: #333; }
        .card p { color: #666; font-size: 14px; }
        a { text-decoration: none}
        .no-shadow {box-shadow: none;}
    </style>
</head>
<body>
    <div class="header">
        <h3>🌋 Energy Landscape Atlas</h3>
        <p>Multi-cell visualizations for the trained EBM model</p>
    </div>
    <div class="grid">
''')
            cards = [
                ("Interactive Dashboard", "Comparable 3D surfaces with energy range analysis", "interactive_comparison_dashboard.html", True),
                ("3×3 Energy Atlas", "Comprehensive view of 9 different cell interactions", "energy_atlas_3x3.png", False),
                ("Main with Satellites", "Primary landscape with comparative smaller views", "main_with_satellites.png", False),
                ("Cell Neighborhood", "Horizontal adjacencies in center 3×3 region", "cell_neighborhood_map.png", False),
                ("Gradient Fields", "Descent directions across different cell pairs", "gradient_field_overview.png", False),
            ]
            for title, description, filename, is_interactive in cards:
                filepath = self.output_dir / filename
                if filepath.exists():
                    card_class = "card interactive" if is_interactive else "card"
                    if is_interactive:
                        f.write(f'''
                        <div class="{card_class}">
                            <h3>🔗 {title}</h3>
                            <p>{description}</p>
                            <a href="{filename}" target="_blank">
                                <div style="background: #f0f0f0; padding: 20px; text-align: center; border-radius: 5px;">
                                    Click to open interactive visualization
                                </div>
                            </a>
                        </div>
''')
                    else:
                        f.write(f'''
                        <div class="{card_class}">
                            <h3>📊 {title}</h3>
                            <p>{description}</p>
                            <a href="{filename}" target="_blank">
                                <img src="{filename}" alt="{title}">
                            </a>
                        </div>
''')
            f.write('''
        <div class="card">
            <h3>📝 Interpretation</h3>
            <div>
                <div class="card no-shadow">
                    <div><strong>Valleys</strong></div>
                    <div>Low-energy regions = solution attractors</div>
                </div>
                <div class="card no-shadow">
                    <div><strong>Cliffs</strong></div>
                    <div>Steep gradients = strong constraints</div>
                 </div>
                <div class="card no-shadow">
                    <div><strong>Similar shapes</strong></div>
                    <div>Translation invariance in model</div>
                </div>
                <div class="card no-shadow">
                    <div><strong>Gradient arrows</strong></div>
                    <div>Descent direction towards solutions</div>
                </div>
            </div>
        </div>
    </div>
</body>
</html>
''')
        console.print(f"[dim]  Atlas index created: {index_path}[/dim]")

def create_multi_cell_atlas():
    try:
        visualizer = MultiCellEnergyVisualizer()
        visualizer.generate_comprehensive_atlas()
        return True
    except Exception as e:
        console.print(f"[red]Error creating multi-cell atlas: {e}[/red]")
        return False