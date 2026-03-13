"""Capability surface analysis for evaluation results."""

import numpy as np


class CapabilitySurface:
    """Analyze evaluation results across configuration dimensions.

    Bins results into grids to visualize how model performance varies
    across different configuration parameters.
    """

    def __init__(self, results: list, dimensions: list = None):
        """
        Args:
            results: List of dicts with 'config' and 'scores' keys.
                config is a dict of parameter values.
                scores is a dict of metric values.
            dimensions: Which config dimensions to analyze. If None, uses all.
        """
        self.results = results
        if dimensions is not None:
            self.dimensions = dimensions
        elif results:
            self.dimensions = list(results[0].get("config", {}).keys())
        else:
            self.dimensions = []

    def compute_grid(
        self, x_dim: str, y_dim: str, metric: str, n_bins: int = 5
    ) -> dict:
        """Bin results into a grid, compute mean metric per cell.

        Args:
            x_dim: Config dimension for x-axis.
            y_dim: Config dimension for y-axis.
            metric: Score metric to aggregate.
            n_bins: Number of bins per dimension.

        Returns:
            Dict with 'x_edges', 'y_edges', 'grid' (2D array of mean values),
            and 'counts' (2D array of sample counts per cell).
        """
        x_vals = []
        y_vals = []
        m_vals = []

        for r in self.results:
            config = r.get("config", {})
            scores = r.get("scores", {})
            if x_dim in config and y_dim in config and metric in scores:
                x_vals.append(float(config[x_dim]))
                y_vals.append(float(config[y_dim]))
                m_vals.append(float(scores[metric]))

        if not x_vals:
            return {
                "x_edges": [],
                "y_edges": [],
                "grid": [],
                "counts": [],
            }

        x_vals = np.array(x_vals)
        y_vals = np.array(y_vals)
        m_vals = np.array(m_vals)

        x_edges = np.linspace(x_vals.min(), x_vals.max(), n_bins + 1)
        y_edges = np.linspace(y_vals.min(), y_vals.max(), n_bins + 1)

        grid = np.full((n_bins, n_bins), np.nan)
        counts = np.zeros((n_bins, n_bins), dtype=int)

        x_idx = np.clip(np.digitize(x_vals, x_edges) - 1, 0, n_bins - 1)
        y_idx = np.clip(np.digitize(y_vals, y_edges) - 1, 0, n_bins - 1)

        for i in range(len(m_vals)):
            xi, yi = x_idx[i], y_idx[i]
            if np.isnan(grid[xi, yi]):
                grid[xi, yi] = 0.0
            grid[xi, yi] += m_vals[i]
            counts[xi, yi] += 1

        # Compute means
        nonzero = counts > 0
        grid[nonzero] = grid[nonzero] / counts[nonzero]

        return {
            "x_edges": x_edges.tolist(),
            "y_edges": y_edges.tolist(),
            "grid": grid.tolist(),
            "counts": counts.tolist(),
        }

    def find_failure_boundary(self, metric: str, threshold: float) -> dict:
        """Find parameter values where metric drops below threshold.

        Args:
            metric: Score metric to check.
            threshold: Value below which is considered failure.

        Returns:
            Dict with 'passing' and 'failing' lists of config dicts.
        """
        passing = []
        failing = []

        for r in self.results:
            scores = r.get("scores", {})
            config = r.get("config", {})
            if metric in scores:
                if float(scores[metric]) >= threshold:
                    passing.append(config)
                else:
                    failing.append(config)

        return {"passing": passing, "failing": failing}
