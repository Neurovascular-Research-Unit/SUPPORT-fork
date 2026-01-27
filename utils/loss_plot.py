import numpy as np
def plotASCII(values, width=60, height=15):
    """Plot values as ASCII scatter plot with auto-scaling y-axis."""
    if len(values) == 0 or np.any(np.isnan(values)):
        return
    if not hasattr(values, "__iter__") or isinstance(values, (float, int, np.floating, np.integer)):
        vals = np.array([values], dtype=float)
    else:
        vals = np.array(values, dtype=float)
    y_min, y_max = min(vals), max(vals)
    y_range = y_max - y_min if y_max > y_min else 0.001
    
    # Add small padding to y range
    y_min -= y_range * 0.05
    y_max += y_range * 0.05
    y_range = y_max - y_min
    
    # Create empty grid
    grid = [[' ' for _ in range(width)] for _ in range(height)]
    
    # Plot points
    for i, val in enumerate(vals):
        x = int(i * (width - 1) / max(len(vals) - 1, 1))
        y = int((val - y_min) / y_range * (height - 1))
        y = height - 1 - y  # Flip y-axis
        y = max(0, min(height - 1, y))
        x = max(0, min(width - 1, x))
        grid[y][x] = '●'
    
    # Print plot
    print(f"\n{'─' * (width + 2)}")
    for row in grid:
        print(f"│{''.join(row)}│")
    print(f"{'─' * (width + 2)}")
    print(f" E0{' ' * (width - 6)}E{len(vals)-1}")
    print(f" Loss: {vals[-1]:.4f} (min: {min(vals):.4f}, max: {max(vals):.4f})")