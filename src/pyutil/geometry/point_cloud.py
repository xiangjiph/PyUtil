import numpy as np
import matplotlib.pyplot as plt

class PointCloud3DSurfaceFit():
    def __init__(self, points_xyz: np.ndarray):
        """
        Fit a 2nd-order surface to a 3D point cloud, assuming the global vertical axis is z.

        Args:
            points_xyz: (N, 3) array of [x,y,z] points. 
        """
        self.points_xyz = np.asarray(points_xyz, dtype=float)
        if self.points_xyz.ndim != 2 or self.points_xyz.shape[1] != 3:
            raise ValueError("points_xyz must be shape (N, 3)")
        
        self.centroid = self.points_xyz.mean(axis=0)
        
        X = self.points_xyz - self.centroid
        self.U, self.S, self.V = np.linalg.svd(X, full_matrices=False)
        # PCA axes: columns of V = [u,v,w] in world coords
        self.R = self.V.T  # columns: x-axis, y-axis, z-axis

        # Transform to local coords: [u,v,w] = (P - centroid) @ R
        L = X @ self.R
        u, v, w = L[:, 0], L[:, 1], L[:, 2]

        # Fit quadratic: w = a + b u + c v + d u^2 + e u v + f v^2
        A = PointCloud3DSurfaceFit._uv_to_polynomial_array(u, v)
        self.coeffs, *_ = np.linalg.lstsq(A, w, rcond=None) # the returned residual is the sum of squared residuals

        self.w_fit = self.f(u, v, self.coeffs)
        self.residuals_w = w - self.w_fit

    def xyz_to_uvw(self, xyz): 
        xyz = np.asarray(xyz, dtype=float)
        if xyz.ndim == 1: 
            xyz = xyz.reshape((1, -1))
        return (xyz - self.centroid) @ self.R
    
    def uvw_to_xyz(self, uvw):
        uvw = np.asarray(uvw, dtype=float)
        if uvw.ndim == 1: 
            uvw = uvw.reshape((1, -1))
        return uvw @ self.R.T + self.centroid

    @staticmethod
    def _uv_to_polynomial_array(u, v):
        assert np.asarray(u).shape == np.asarray(v).shape, "u and v must have the same shape"
        u = u.flatten()
        v = v.flatten()
        # w = a + b u + c v + d u^2 + e u v + f v^2
        return np.column_stack([np.ones_like(u), u, v, u**2, u*v, v**2])

    @staticmethod
    def f(u, v, coeffs):
        u_shape = np.asarray(u).shape
        assert np.asarray(v).shape == u_shape, "u and v must have the same shape"
        u = u.flatten()
        v = v.flatten()
        poly_array = PointCloud3DSurfaceFit._uv_to_polynomial_array(u, v)
        w = poly_array @ coeffs
        w = w.reshape(u_shape)
        return w
    
    @staticmethod
    def df_du(u, v, coeffs):
        u = np.asarray(u).flatten()
        v = np.asarray(v).flatten()
        # dw/du = b + 2 d u + e v
        b = coeffs[1]
        d = coeffs[3]
        e = coeffs[4]
        dw_du = b + 2 * d * u + e * v
        dw_du = dw_du.reshape(np.asarray(u).shape)
        return dw_du
    
    @staticmethod
    def df_dv(u, v, coeffs):
        u = np.asarray(u).flatten()
        v = np.asarray(v).flatten()
        # dw/dv = c + e u + 2 f v
        c = coeffs[2]
        e = coeffs[4]
        f = coeffs[5]
        dw_dv = c + e * u + 2 * f * v
        dw_dv = dw_dv.reshape(np.asarray(u).shape)
        return dw_dv

    @staticmethod
    def uv_to_polynomial_array(u, v):
        """Convert (u,v) to polynomial feature array [1, u, v, u^2, uv, v^2]
        Input:
            u: (N,) or (N,1) array of u-coordinates
            v: (N,) or (N,1) array of v-coordinates
        Returns:
            (N, 6) array of polynomial features
        """
        u = np.asarray(u)
        v = np.asarray(v)
        return np.column_stack([np.ones_like(u), u, v, u**2, u*v, v**2])
    
    def project_points_to_surface(self, xyz_points, 
                                  max_iter=100, tol=1e-6, 
                                  return_dist_Q=False):
        """
        Project points in world coordinates onto the fitted surface.

        Args:
            xyz_points: (M, 3) array of points in world coordinates.
            max_iter: Maximum number of iterations for projection.
            tol: Tolerance for convergence.

        Returns:
            uvw_proj: (M, 3) array of projected points in world coordinates.
        """
        uvw = self.xyz_to_uvw(xyz_points)
        u1, v1, w1 = PointCloud3DSurfaceFit._project_uvw_points_to_surface(
            uvw[:, 0], uvw[:, 1], uvw[:, 2], self.coeffs,
            max_iter=max_iter, tol=tol)
        uvw_proj = np.column_stack([u1, v1, w1])
        xyz_proj = self.uvw_to_xyz(uvw_proj)
        if return_dist_Q:
            dists = np.linalg.norm(xyz_points - xyz_proj, axis=1)
            return xyz_proj, dists
        else:
            return xyz_proj

    @staticmethod
    def _project_uvw_points_to_surface(u0, v0, w0, coeffs, 
                                  max_iter=100, tol=1e-6):
        """
        Project points (u0, v0, w0) onto the fitted surface defined by coeffs
        using iterative Newton-Raphson method.

        Args:
            u0, v0, w0: Initial coordinates of points to project.
            coeffs: Coefficients of the fitted surface.
            max_iter: Maximum number of iterations.
            tol: Tolerance for convergence.

        Returns:
            u_proj, v_proj, w_proj: Projected coordinates on the surface.
        """
        u0 = np.asarray(u0, dtype=float).flatten()
        v0 = np.asarray(v0, dtype=float).flatten()
        w0 = np.asarray(w0, dtype=float).flatten()

        u = u0.copy()
        v = v0.copy()
        
        fuu = 2 * coeffs[3]
        fuv = coeffs[4]
        fvv = 2 * coeffs[5]
        eps = 1e-12
        for _ in range(max_iter):
            w_fit = PointCloud3DSurfaceFit.f(u, v, coeffs)
            fu = PointCloud3DSurfaceFit.df_du(u, v, coeffs)
            fv = PointCloud3DSurfaceFit.df_dv(u, v, coeffs)
            # Residuals
            r = w_fit - w0
            
            F1 = (u - u0) + r * fu
            F2 = (v - v0) + r * fv

            # Jacobian
            A = 1.0 + fu ** 2 + 2 * r * fuu
            B = fu * fv + r * fuv
            D = 1.0 + fv ** 2 + 2 * r * fvv
            det = A * D - B * B
            det = np.where(np.abs(det) < eps, np.sign(det) * eps, det)

            # Update step
            du = (D * F1 - B * F2) / det
            dv = (A * F2 - B * F1) / det

            u -= du
            v -= dv

            if np.max(np.abs(du) + np.abs(dv)) < tol:
                break

        w_proj = PointCloud3DSurfaceFit.f(u, v, coeffs)
        return u, v, w_proj


    def vis_points_w_fitted_surface(self, grid_res=80, pad=0.05, 
                                    vis_frame='world'):
        
        P = np.asarray(self.points_xyz, dtype=float)
        centroid = self.centroid
        R = self.R

        # Local coords
        L = self.xyz_to_uvw(P)
        u, v, w = L[:, 0], L[:, 1], L[:, 2]

        # Grid in local (u,v)
        umin, umax = u.min(), u.max()
        vmin, vmax = v.min(), v.max()
        du = (umax - umin) * pad
        dv = (vmax - vmin) * pad

        ug = np.linspace(umin - du, umax + du, grid_res)
        vg = np.linspace(vmin - dv, vmax + dv, grid_res)
        U, V = np.meshgrid(ug, vg)
        W = PointCloud3DSurfaceFit.f(U, V, self.coeffs)

        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(111, projection="3d")
        
        if vis_frame == 'local':
            ax.scatter(L[:, 0], L[:, 1], L[:, 2], s=1, alpha=0.9)
            ax.plot_surface(U, V, W, alpha=0.25, linewidth=0)
        else:
            world_grid = self.uvw_to_xyz(np.stack([U, V, W], axis=-1))
            ax.scatter(P[:, 0], P[:, 1], P[:, 2], s=1, alpha=0.9)
            ax.plot_surface(world_grid[..., 0], world_grid[..., 1], world_grid[..., 2], alpha=0.25, linewidth=0)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(f"Quadratic surface fit in PCA-aligned {vis_frame} frame")

        try:
            if vis_frame == 'local':
                ax.set_box_aspect([np.ptp(L[:, 0]), np.ptp(L[:, 1]), np.ptp(L[:, 2])])
            else:
                ax.set_box_aspect([np.ptp(P[:, 0]), np.ptp(P[:, 1]), np.ptp(P[:, 2])])
        except Exception:
            pass

        plt.tight_layout()
        return fig, ax