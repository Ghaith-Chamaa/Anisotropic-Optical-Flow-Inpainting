import os
import math
import argparse
import time
import numpy as np
from PIL import Image
from flow_utils import read_flow, write_flow
from flow_viz import save_vis_flow_tofile


def get_neighbours(nn_type, r):
    """Gets the neighborhood definition."""
    if nn_type == 1:
        base_p = [
            (1, 0), (0, 1), (-1, 0), (0, -1),
            (1, 1), (-1, -1), (-1, 1), (1, -1),
            (2, 1), (1, 2), (2, -1), (1, -2),
            (-2, -1), (-1, -2), (-2, 1), (-1, 2),
            (3, 1), (1, 3), (3, -1), (1, -3),
            (-3, -1), (-1, -3), (-3, 1), (-1, 3),
            (3, 2), (2, 3), (3, -2), (2, -3),
            (-3, -2), (-2, -3), (-3, 2), (-2, 3),
        ]
        if r == 1: return np.array(base_p[:8])
        if r == 2: return np.array(base_p[:16])
        if r == 3: return np.array(base_p[:32])
        raise ValueError(f"Invalid radius r={r} for nn_type=1")
    elif nn_type == 2:
        # using a square window side = 2 * r + 1, as defined in the article
        p = []
        for y in range(-r, r + 1):
            for x in range(-r, r + 1):
                if x == 0 and y == 0:
                    continue
                p.append((x, y))
        return np.array(p)
    else:
        raise ValueError(f"Invalid neighbourhood type nn_type={nn_type}")

def compute_weights(guide, lambda_val, w_type, nn_type, r):
    """Computes the weight map for AMLE."""
    h, w, pd = guide.shape
    neighbors = get_neighbours(nn_type, r)
    nn = len(neighbors)
    
    weights = np.zeros((h, w, nn), dtype=np.float32)

    for p_idx, (dx, dy) in enumerate(neighbors):
        # Moving the entire guide image using neighborhood shifts to rolled it for vectorization 
        # This allows for the caluclation of the weights at the associated shift on the entire image in less number instruction
        guide_shifted = np.roll(guide, (-dy, -dx), axis=(0, 1))
        
        color_dist_sq = np.sum((guide - guide_shifted)**2, axis=2) / pd
        
        spatial_dist_sq = dx**2 + dy**2

        if w_type == 1:
            dist = np.sqrt((1 - lambda_val) * color_dist_sq + lambda_val * spatial_dist_sq)
        elif w_type == 2:
            dist = (1 - lambda_val) * np.sqrt(color_dist_sq) + lambda_val * np.sqrt(spatial_dist_sq)
        else:
            dist = (1 - lambda_val) * color_dist_sq + lambda_val * spatial_dist_sq

        weights[:, :, p_idx] = 1.0 / (dist + 1e-9)

    return weights

def amle_iteration(u, weights, mask, nn_type, r):
    """Performs one iteration of the AMLE algorithm."""
    h, w = u.shape
    u_new = u.copy()
    neighbors = get_neighbours(nn_type, r)
    
    mask_coords = np.argwhere(mask)

    for y, x in mask_coords:
        x0 = u[y, x]
        
        neighbor_vals = []
        neighbor_weights = []
        for p_idx, (dx, dy) in enumerate(neighbors):
            nx, ny = x + dx, y + dy
            if 0 <= ny < h and 0 <= nx < w:
                neighbor_vals.append(u[ny, nx])
                neighbor_weights.append(weights[y, x, p_idx])
        
        neighbor_vals = np.array(neighbor_vals)
        neighbor_weights = np.array(neighbor_weights)

        eikonal = (neighbor_vals - x0) * neighbor_weights
        
        ind_pos = np.argmax(eikonal)
        ind_neg = np.argmin(eikonal)

        a = neighbor_weights[ind_neg]
        b = neighbor_weights[ind_pos]
        
        new_val = (a * neighbor_vals[ind_neg] + b * neighbor_vals[ind_pos]) / (a + b)
        u_new[y, x] = new_val
        
    return u_new

def amle_extension(u, weights, mask, niter, err_thresh, nn_type, r):
    """Runs the iterative AMLE extension."""
    u_curr = u.copy()
    n_mask = np.sum(mask)

    for k in range(niter):
        u_prev = u_curr.copy()
        u_curr = amle_iteration(u_curr, weights, mask, nn_type, r)
        
        diff = np.sum(np.abs(u_curr[mask] - u_prev[mask]))
        err = diff / n_mask if n_mask > 0 else 0
        if err < err_thresh:
            print(f"Converged at iteration {k+1}")
            break
            
    return u_curr

def amle_recursive(u_comp, guide, niter, nscales, lambda_val, err_thresh, w_type, nn_type, r):
    """Recursive multi-scale AMLE implementation."""
    if nscales > 1:
        h, w = u_comp.shape
        hs, ws = math.ceil(h / 2), math.ceil(w / 2)

        u_comp_img = Image.fromarray(np.nan_to_num(u_comp))
        guide_img = Image.fromarray((guide * 255).astype(np.uint8))

        u_small = np.array(u_comp_img.resize((ws, hs), Image.BILINEAR), dtype=np.float32)
        guide_small = np.array(guide_img.resize((ws, hs), Image.BILINEAR), dtype=np.float32) / 255.0
        u_small[np.array(Image.fromarray(u_comp).resize((ws, hs), Image.NEAREST)) == 0] = np.nan

        inpainted_small = amle_recursive(u_small, guide_small, niter, nscales - 1, lambda_val, err_thresh, w_type, nn_type, r)
        
        inpainted_img = Image.fromarray(inpainted_small)
        init = np.array(inpainted_img.resize((w, h), Image.BILINEAR), dtype=np.float32)
    else:
        init = np.zeros_like(u_comp)

    mask = np.isnan(u_comp)
    u_comp[mask] = init[mask]

    weights = compute_weights(guide, lambda_val, w_type, nn_type, r)
    
    return amle_extension(u_comp, weights, mask, niter, err_thresh, nn_type, r)

def amle_inpainting(flow, guide, mask, S, lambda_val, epsilon, g, nt, r):
    """Main function for AMLE inpainting."""
    flow[mask > 0] = np.nan

    inpainted_flow = np.zeros_like(flow)

    for i in range(flow.shape[2]):
        u_comp = flow[:, :, i]
        inpainted_flow[:, :, i] = amle_recursive(u_comp, guide, 5000, S, lambda_val, epsilon, g, nt, r)

    return inpainted_flow
    
def main():
    parser = argparse.ArgumentParser(description='AMLE Optical Flow Inpainting with default parameters from the paper.')
    # Required positional arguments
    parser.add_argument('--input_flow', help='Input flow file (.flo)')
    parser.add_argument('--mask', help='Inpainting mask file (PNG)')
    parser.add_argument('--guide', help='Guiding image file (PNG)')

    # Optional arguments with defaults from the paper
    parser.add_argument('--S', type=int, default=4, help='Number of scales (default: 4)')
    parser.add_argument('--lambda_val', type=float, default=0.001, help='Anisotropic weight (default: 0.001)')
    parser.add_argument('--epsilon', type=float, default=0.0001, help='Stopping criterion threshold (default: 0.0001)')
    parser.add_argument('--g', type=int, default=3, help='Weight selection (1, 2, or 3) (default: 3)')
    parser.add_argument('--nt', type=int, default=1, help='Type of local neighbourhood (1 or 2) (default: 1)')
    parser.add_argument('--r', type=int, default=2, help='Neighbourhood ratio (default: 2)')
    
    parser.add_argument('--output_dir', default='output_flow', help='Directory to save .flo files')
    parser.add_argument('--vis_dir', default='output_vis', help='Directory to save flow visualizations')
    args = parser.parse_args()

    start_time = time.time()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.vis_dir, exist_ok=True)

    flow = read_flow(args.input_flow)
    mask = np.array(Image.open(args.mask).convert('L'))
    guide = np.array(Image.open(args.guide).convert('RGB')) / 255.0

    inpainted_flow = amle_inpainting(flow, guide, mask, args.S, args.lambda_val, args.epsilon, args.g, args.nt, args.r)

    end_time = time.time()
    print(f"Inpainting computation finished in {end_time - start_time:.2f} seconds.")
    
    base_name = os.path.splitext(os.path.basename(args.input_flow))[0]

    output_flow_path = os.path.join(args.output_dir, f'{base_name}_amle.flo')
    write_flow(output_flow_path, inpainted_flow)
    print(f"Inpainted flow saved to {output_flow_path}")

    output_vis_path = os.path.join(args.vis_dir, f'{base_name}_amle.png')
    save_vis_flow_tofile(inpainted_flow, output_vis_path)
    print(f"Flow visualization saved to {output_vis_path}")

if __name__ == '__main__':
    main()
