import os
import argparse
import numpy as np
from PIL import Image
from scipy import sparse
from scipy.sparse.linalg import spsolve
from flow_utils import read_flow, write_flow
from flow_viz import save_vis_flow_tofile


def incidence_matrix_4n(h, w):
    """Creates the incidence matrix for a 4-connected grid graph."""
    # path of length h
    x = sparse.spdiags([-np.ones(h), np.ones(h)], [0, 1], h - 1, h)
    # path of length w
    y = sparse.spdiags([-np.ones(w), np.ones(w)], [0, 1], w - 1, w)
    
    # kronecker union for 4-connectivity
    B = sparse.vstack([
        sparse.kron(sparse.eye(w), x),
        sparse.kron(y, sparse.eye(h))
    ], format='csr')
    return B

def laplace_beltrami_interpolation(v, g, kappa, lambda_val, d_val):
    """Performs Laplace-Beltrami interpolation."""
    v1 = v[:, :, 0].flatten()
    v2 = v[:, :, 1].flatten()

    gr = g[:, :, 0].flatten()
    gg = g[:, :, 1].flatten()
    gb = g[:, :, 2].flatten()

    kappa = kappa / np.maximum(1, np.max(kappa))
    h, w = kappa.shape

    B = incidence_matrix_4n(h, w)
    m = B.shape[0]

    X, Y = np.meshgrid(np.arange(w), np.arange(h))
    X_flat = X.flatten()
    Y_flat = Y.flatten()

    # calculate color differences and spatial differences
    color_diff_sq = ((B @ gr)**2 + (B @ gg)**2 + (B @ gb)**2) / 3
    spatial_diff_sq = (B @ X_flat)**2 + (B @ Y_flat)**2

    if d_val == 1:
        D = np.sqrt((1 - lambda_val) * color_diff_sq + lambda_val * spatial_diff_sq)
    elif d_val == 2:
        D = (1 - lambda_val) * np.sqrt(color_diff_sq) + lambda_val * np.sqrt(spatial_diff_sq)
    elif d_val == 3:
        D = (1 - lambda_val) * color_diff_sq + lambda_val * spatial_diff_sq
    else:
        raise ValueError("Invalid distance identifier 'd_val'")

    W_diag = 1.0 / (D + 1e-9)
    W = sparse.spdiags(W_diag, 0, m, m)

    Lw = -B.T @ W @ B

    kappa_flat = kappa.flatten()
    kappas = sparse.spdiags(kappa_flat == 1, 0, w * h, w * h)
    
    A = sparse.eye(w * h) - kappas + kappas @ Lw
    
    b1 = (sparse.eye(w * h) - kappas) @ v1
    b2 = (sparse.eye(w * h) - kappas) @ v2

    x1 = spsolve(A, b1)
    x2 = spsolve(A, b2)

    u1 = x1.reshape(h, w)
    u2 = x2.reshape(h, w)
    
    return np.stack([u1, u2], axis=-1)


def main():
    parser = argparse.ArgumentParser(description='Laplace-Beltrami Optical Flow Inpainting')
    parser.add_argument('--input_flow', help='Input flow file (.flo)')
    parser.add_argument('--mask', help='Inpainting mask file (PNG)')
    parser.add_argument('--guide', help='Guiding image file (PNG)')
    parser.add_argument('--lambda_val', type=float, default=0.001, help='Anisotropic weight')
    parser.add_argument('--g_val', type=int, default=3, help='Weight selection (1, 2, or 3) (default: 3)')
    parser.add_argument('--output_dir', default='output_flow', help='Directory to save .flo files')
    parser.add_argument('--vis_dir', default='output_vis', help='Directory to save flow visualizations')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.vis_dir, exist_ok=True)

    flow = read_flow(args.input_flow)
    mask = np.array(Image.open(args.mask).convert('L'))
    guide = np.array(Image.open(args.guide).convert('RGB')) / 255.0

    inpainted_flow = laplace_beltrami_interpolation(flow, guide, mask, args.lambda_val, args.g_val)

    base_name = os.path.splitext(os.path.basename(args.input_flow))[0]
    
    output_flow_path = os.path.join(args.output_dir, f'{base_name}_lb.flo')
    write_flow(output_flow_path, inpainted_flow)
    print(f"Inpainted flow saved to {output_flow_path}")

    output_vis_path = os.path.join(args.vis_dir, f'{base_name}_lb.png')
    save_vis_flow_tofile(inpainted_flow, output_vis_path)
    print(f"Flow visualization saved to {output_vis_path}")

if __name__ == '__main__':
    main()
