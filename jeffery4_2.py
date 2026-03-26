from scipy.integrate import quad
from scipy.optimize import brentq, root_scalar
import numpy as np
import jax
import jax.numpy as jnp
from jax import grad, jacfwd, jacrev, hessian, custom_jvp
from functools import partial

# Enable 64-bit precision in JAX
jax.config.update("jax_enable_x64", True)


class Ellipsoid:
    def __init__(self, a, epsilon, mu=1):
        self.a = np.array(a)
        self.n = len(a)
        self.mu = mu
        self.epsilon = np.array(epsilon)
        self.set_strain(epsilon)
        self.a_jax = jnp.array(a)
        self.set_coefs()

    def set_omega(self, omega):
        self.omega = np.array(omega)

    def jeffery_omega(self):
        n = self.n
        omega = np.zeros(n)
        for i in range(n):
            i1 = (i+1)%n; i2 = (i+2)%n
            a1sq = self.a[i1]**2; a2sq = self.a[i2]**2
            vorticity = self.epsilon_anti[i2, i1]
            strain_contrib = (a1sq - a2sq) / (a1sq + a2sq) * self.epsilon_symm[i2, i1]
            omega[i] = vorticity + strain_contrib
        return omega

    def set_strain(self, epsilon):
        self.epsilon = np.array(epsilon)
        self.epsilon_anti = 0.5*(self.epsilon - self.epsilon.T)
        self.epsilon_symm = 0.5*(self.epsilon + self.epsilon.T)

    def ellipse(self, x, l):
        return jnp.sum(x**2 / (self.a**2 + l)) - 1

    def delta(self, l):
        return jnp.prod(self.a_jax**2 + l)**(1/2)

    def find_l0(self, x):
        x_np = np.asarray(x)
        a_np = self.a
        def ellipse_np(l):
            return np.sum(x_np**2 / (a_np**2 + l)) - 1
        l_min = -np.min(a_np**2) + 1e-10
        l_max = 1e6
        try:
            result = root_scalar(ellipse_np, bracket=[l_min, l_max], method='brentq')
            return result.root
        except ValueError:
            from scipy.optimize import fsolve
            result = fsolve(ellipse_np, 0.0)[0]
            return result

    def find_l0_jax(self, x):
        @custom_jvp
        def _find_l0_jax_inner(x_arg):
            x_np = np.asarray(x_arg)
            a_np = self.a
            def ellipse_np(l):
                return np.sum(x_np**2 / (a_np**2 + l)) - 1
            l_min = -np.min(a_np**2) + 1e-10
            l_max = 1e6
            try:
                result = root_scalar(ellipse_np, bracket=[l_min, l_max], method='brentq')
                return jnp.array(result.root)
            except ValueError:
                from scipy.optimize import fsolve
                result = fsolve(ellipse_np, 0.0)[0]
                return jnp.array(result)

        @_find_l0_jax_inner.defjvp
        def _find_l0_jax_jvp(primals, tangents):
            x_arg, = primals
            x_dot, = tangents
            l0 = _find_l0_jax_inner(x_arg)
            a = self.a_jax
            dF_dx = 2 * x_arg / (a**2 + l0)
            dF_dl0 = -jnp.sum(x_arg**2 / (a**2 + l0)**2)
            dl0_dx = -dF_dx / dF_dl0
            l0_dot = jnp.dot(dl0_dx, x_dot)
            return l0, l0_dot

        return _find_l0_jax_inner(x)

    def I(self, x, zero=False):
        x_np = np.asarray(x)
        a_np = self.a
        lower_limit = 0.0 if zero else self.find_l0(x)
        result = np.zeros(self.n)
        for i in range(self.n):
            def integrand(l, i=i):
                delta_val = np.prod(a_np**2 + l)**(1/2)
                return 1 / delta_val / (a_np[i]**2 + l)
            result[i], _ = quad(integrand, lower_limit, np.inf)
        return result

    def I_(self, x, zero=False):
        x_np = np.asarray(x)
        a_np = self.a
        lower_limit = 0.0 if zero else self.find_l0(x)
        result = np.zeros(self.n)
        for i in range(self.n):
            def integrand(l, i=i):
                delta_val = np.prod(a_np**2 + l)**(1/2)
                return (a_np[i]**2 + l) / delta_val**3
            result[i], _ = quad(integrand, lower_limit, np.inf)
        return result

    def I__(self, x, zero=False):
        x_np = np.asarray(x)
        a_np = self.a
        lower_limit = 0.0 if zero else self.find_l0(x)
        result = np.zeros(self.n)
        for i in range(self.n):
            def integrand(l, i=i):
                delta_val = np.prod(a_np**2 + l)**(1/2)
                return l * (a_np[i]**2 + l) / delta_val**3
            result[i], _ = quad(integrand, lower_limit, np.inf)
        return result

    @property
    def I0(self):
        return self.I(None, zero=True)

    @property
    def I0_(self):
        return self.I_(None, zero=True)

    @property
    def I0__(self):
        return self.I__(None, zero=True)

    def gen_ABC(self):
        n = self.n
        ABC = [None]*n
        for i in range(n):
            i1 = (i+1)%n; i2 = (i+2)%n
            ABC[i] = 1/6*(2*self.I0__[i] * self.epsilon_symm[i,i]
                          -self.I0__[i1] * self.epsilon_symm[i1,i1]
                          -self.I0__[i2] * self.epsilon_symm[i2,i2])/(self.I0__[i1]*self.I0__[i2]
                                                                     +self.I0__[i2]*self.I0__[i]
                                                                     +self.I0__[i]*self.I0__[i1])
        return ABC

    def gen_FGH(self, prime=False):
        al0, be0, ga0 = self.I0[0], self.I0[1], self.I0[2]
        alp0, bep0, gap0 = self.I0_[0], self.I0_[1], self.I0_[2]
        a0, b0, c0 = self.a[0], self.a[1], self.a[2]
        f = self.epsilon_symm[1,2]
        g = self.epsilon_symm[0,2]
        h = self.epsilon_symm[0,1]
        xi   = self.epsilon_anti[2,1]
        eta  = self.epsilon_anti[0,2]
        zeta = self.epsilon_anti[1,0]
        om1, om2, om3 = self.omega[0], self.omega[1], self.omega[2]

        if not prime:
            F = (be0*f - c0**2*al0*(xi  - om1)) / (2*alp0*(b0**2*be0 + c0**2*ga0))
            G = (ga0*g - a0**2*be0*(eta - om2)) / (2*bep0*(a0**2*al0 + c0**2*ga0))
            H = (al0*h - b0**2*ga0*(zeta- om3)) / (2*gap0*(a0**2*al0 + b0**2*be0))
            return [F, G, H]
        else:
            F_ = (ga0*f + b0**2*al0*(xi  - om1)) / (2*alp0*(b0**2*be0 + c0**2*ga0))
            G_ = (al0*g + c0**2*be0*(eta - om2)) / (2*bep0*(a0**2*al0 + c0**2*ga0))
            H_ = (be0*h + a0**2*ga0*(zeta- om3)) / (2*gap0*(a0**2*al0 + b0**2*be0))
            return [F_, G_, H_]

    def gen_RST(self):
        n = self.n
        RST = [None]*n
        for i in range(n):
            i1 = (i+1)%n; i2 = (i+2)%n
            RST[i] = -self.epsilon_symm[i2,i1]/self.I0_[i]
        return RST

    def gen_UVW(self):
        n = self.n
        UVW = [None]*n
        for i in range(n):
            i1 = (i+1)%n; i2 = (i+2)%n
            UVW[i] = 2*self.a[i1]**2*self.ABC[i1]-2*self.a[i2]**2*self.ABC[i2]
        return UVW

    def P(self, x):
        lam = self.find_l0_jax(x)
        return jnp.sqrt(jnp.sum(x**2 / (self.a_jax**2 + lam)**2)**(-1))

    def set_coefs(self):
        self.omega = self.jeffery_omega()
        self.ABC = self.gen_ABC()
        self.FGH = self.gen_FGH()
        self.FGH_ = self.gen_FGH(prime=True)
        self.RST = self.gen_RST()
        self.UVW = self.gen_UVW()

    def dOm_dx(self, x):
        n = self.n
        D = jnp.zeros(n)
        I_vals = self.I_all_jax(x)
        for i in range(n):
            D = D.at[i].set(2*I_vals[i]*x[i])
        return D

    def d2Om_dx2(self, x):
        lam = self.find_l0_jax(x)
        n = self.n
        I_vals = self.I_all_jax(x)
        P_val = self.P(x)
        delta_val = self.delta(lam)
        D_elements = []
        for i in range(n):
            row = []
            for j in range(n):
                if i == j:
                    val = 2*I_vals[i] - (4*x[i]**2 * P_val**2 /
                                        ((self.a_jax[i]**2 + lam)**2 * delta_val))
                    row.append(val)
                else:
                    if (i, j) in [(1, 2), (2, 1)]:
                        val = -4*x[1]*x[2]*P_val**2 / (
                            (self.a_jax[1]**2 + lam)*(self.a_jax[2]**2 + lam)*delta_val)
                    elif (i, j) in [(2, 0), (0, 2)]:
                        val = -4*x[2]*x[0]*P_val**2 / (
                            (self.a_jax[2]**2 + lam)*(self.a_jax[0]**2 + lam)*delta_val)
                    elif (i, j) in [(0, 1), (1, 0)]:
                        val = -4*x[0]*x[1]*P_val**2 / (
                            (self.a_jax[0]**2 + lam)*(self.a_jax[1]**2 + lam)*delta_val)
                    else:
                        val = 0.0
                    row.append(val)
            D_elements.append(row)
        return jnp.array(D_elements)

    def dchi_dx(self, x):
        lam = self.find_l0_jax(x)
        n = self.n
        D = jnp.zeros([n,n])
        I_vals = self.I_all_jax(x)
        I__vals = self.I__all_jax(x)
        for i in range(n):
            i1 = (i+1)%n; i2 = (i+2)%n
            D = D.at[i,i].set(-2*jnp.prod(x)*self.P(x)**2/self.delta(lam)**3)
            D = D.at[i,i1].set(I__vals[i]*x[i2] - 2*x[i1]**2*x[i2]*self.P(x)**2/(self.a_jax[i1]**2+lam)**2/(self.a_jax[i2]**2+lam)/self.delta(lam))
            D = D.at[i,i2].set(I__vals[i]*x[i1] - 2*x[i1]*x[i2]**2*self.P(x)**2/(self.a_jax[i1]**2+lam)/(self.a_jax[i2]**2+lam)**2/self.delta(lam))
        return D

    def u(self, x):
        lam = self.find_l0_jax(x)
        n = self.n
        u_components = []
        chi = self.chi(x)
        dchi_dx = self.dchi_dx(x)
        dOm_dx = self.dOm_dx(x)
        d2Om_dx2 = self.d2Om_dx2(x)
        R,S,T = self.RST
        for i in range(n):
            i1 = (i+1)%n; i2 = (i+2)%n
            u_i = ( (R*dchi_dx[0,i]+S*dchi_dx[1,i]+T*dchi_dx[2,i])
                    + self.UVW[i2]*dchi_dx[i2,i1] - self.UVW[i1]*dchi_dx[i1,i2]
                    + self.ABC[i]*(x[i]*d2Om_dx2[i,i]-dOm_dx[i]) + self.FGH[i2]*(x[i]*d2Om_dx2[i,i1]-dOm_dx[i1]) + self.FGH_[i1]*(x[i]*d2Om_dx2[i,i2]-dOm_dx[i2])
                    + x[i1] * (self.FGH_[i2]*d2Om_dx2[i,i] + self.ABC[i1]*d2Om_dx2[i,i1] + self.FGH[i]*d2Om_dx2[i,i2])
                    + x[i2] * (self.FGH[i1]*d2Om_dx2[i,i] + self.FGH_[i]*d2Om_dx2[i,i1] +  self.ABC[i2]*d2Om_dx2[i,i2]) )
            u_components.append(u_i)

        u_pert = jnp.array(u_components)
        u_0 = jnp.dot(self.epsilon_symm + self.epsilon_anti, x)

        return u_pert + u_0

    def p(self, x):
        n = self.n
        d2Om_dx2 = self.d2Om_dx2(x)
        P = jnp.zeros([n,n])
        for i in range(n):
            P = P.at[i, i].set(self.ABC[i] * d2Om_dx2[i, i])
            P = P.at[(i+1)%n, (i+2)%n].set(self.FGH[i] * d2Om_dx2[(i+1)%n, (i+2)%n])
            P = P.at[(i+2)%n, (i+1)%n].set(self.FGH_[i] * d2Om_dx2[(i+2)%n, (i+1)%n])
        return 2 * self.mu * jnp.sum(P)

    def du_dx(self, x):
        return jacfwd(self.u)(x)

    def I_jax(self, x, i):
        @custom_jvp
        def _I_jax_inner(x_arg):
            l0_val = self.find_l0(x_arg)
            return jnp.array(self.I_(x_arg)[i])

        @_I_jax_inner.defjvp
        def _I_jax_jvp(primals, tangents):
            x_arg, = primals
            x_dot, = tangents
            I_val = _I_jax_inner(x_arg)
            l0_val = self.find_l0_jax(x_arg)
            a = self.a_jax
            dF_dx = 2 * x_arg / (a**2 + l0_val)
            dF_dl0 = -jnp.sum(x_arg**2 / (a**2 + l0_val)**2)
            dl0_dx = -dF_dx / dF_dl0
            delta_at_l0 = jnp.prod(a**2 + l0_val)**(1/2)
            boundary_term = -(a[i]**2 + l0_val) / delta_at_l0**3
            dI_dx = boundary_term * dl0_dx
            I_dot = jnp.dot(dI_dx, x_dot)
            return I_val, I_dot

        return _I_jax_inner(x)

    def I_all_jax(self, x):
        @custom_jvp
        def _I_all_jax_inner(x_arg):
            return jnp.array(self.I(x_arg))

        @_I_all_jax_inner.defjvp
        def _I_all_jax_jvp(primals, tangents):
            x_arg, = primals
            x_dot, = tangents
            I_vals = _I_all_jax_inner(x_arg)
            l0_val = self.find_l0_jax(x_arg)
            a = self.a_jax
            dF_dx = 2 * x_arg / (a**2 + l0_val)
            dF_dl0 = -jnp.sum(x_arg**2 / (a**2 + l0_val)**2)
            dl0_dx = -dF_dx / dF_dl0
            delta_at_l0 = jnp.prod(a**2 + l0_val)**(1/2)
            boundary_terms = -1 / (delta_at_l0 * (a**2 + l0_val))
            dI_dx = jnp.outer(boundary_terms, dl0_dx)
            I_dot = jnp.dot(dI_dx, x_dot)
            return I_vals, I_dot

        return _I_all_jax_inner(x)

    def I__all_jax(self, x):
        @custom_jvp
        def _I__all_jax_inner(x_arg):
            return jnp.array(self.I_(x_arg))

        @_I__all_jax_inner.defjvp
        def _I__all_jax_jvp(primals, tangents):
            x_arg, = primals
            x_dot, = tangents
            I__vals = _I__all_jax_inner(x_arg)
            l0_val = self.find_l0_jax(x_arg)
            a = self.a_jax
            dF_dx = 2 * x_arg / (a**2 + l0_val)
            dF_dl0 = -jnp.sum(x_arg**2 / (a**2 + l0_val)**2)
            dl0_dx = -dF_dx / dF_dl0
            delta_at_l0 = jnp.prod(a**2 + l0_val)**(1/2)
            boundary_terms = -(a**2 + l0_val) / delta_at_l0**3
            dI__dx = jnp.outer(boundary_terms, dl0_dx)
            I__dot = jnp.dot(dI__dx, x_dot)
            return I__vals, I__dot

        return _I__all_jax_inner(x)

    def chi(self, x):
        n = len(x)
        chi_vals = []
        for i in range(n):
            I_val = self.I_jax(x, i)
            chi_vals.append(I_val * x[(i+1) % n] * x[(i+2) % n])
        return jnp.array(chi_vals)

    def pack(self, epsilon):
        return epsilon.flatten()

    def unpack(self, mode):
        epsilon = np.reshape(mode[:self.n**2], [self.n, self.n])
        return epsilon

    def use_surface_mode(self):
        """
        Switch to fast surface-point evaluation mode.

        On the ellipsoid surface l0(x) = 0 by definition, so all
        x-dependent integrals I(x), I'(x), I''(x) collapse to the
        precomputed constants I0, I0_, I0__.  Patching the four JAX
        integral helpers to return those constants means jacfwd(self.u)
        differentiates through pure JAX arithmetic only — no scipy quad,
        no brentq — cutting per-point cost from ~seconds to ~milliseconds.

        Call this ONCE after constructing the Ellipsoid and before the
        precompute loop.  Undo with restore_full_mode().
        """
        # Cache originals so we can restore later
        self._orig_find_l0_jax = self.find_l0_jax
        self._orig_I_all_jax   = self.I_all_jax
        self._orig_I__all_jax  = self.I__all_jax
        self._orig_I_jax       = self.I_jax

        # Precompute the three constant integral arrays (depend only on a)
        I0_const   = jnp.array(self.I0)    # shape (n,)  — I
        I0_const_  = jnp.array(self.I0_)   # shape (n,)  — I'
        I0_const__ = jnp.array(self.I0__)  # shape (n,)  — I''

        # l0 = 0 on the surface
        self.find_l0_jax = lambda x: jnp.array(0.0)

        # I(x)  → I0   (used in dOm_dx, d2Om_dx2)
        self.I_all_jax  = lambda x: I0_const

        # I'(x) → I0_  (used in dchi_dx)
        self.I__all_jax = lambda x: I0_const_

        # I'_i(x) → I0_[i]  (used in chi, scalar per component)
        self.I_jax      = lambda x, i: I0_const_[i]

    def sigma(self, x):
        """
        Stress tensor at point x using fully analytical expressions.

        Avoids jacfwd entirely — pure arithmetic in terms of the
        precomputed coefficients (ABC, FGH, FGH_, RST, UVW) and the
        geometric functions evaluated at lam = find_l0(x).

        On the ellipsoid surface lam=0, so use_surface_mode() makes
        this essentially free (no quad, no brentq).

        Parameters
        ----------
        x : array-like, shape (3,)
            Position vector.

        Returns
        -------
        sig : np.ndarray, shape (3, 3)
            Full stress tensor sigma_ij = -p*delta_ij + 2*mu*e_ij
        """
        x = np.asarray(x, dtype=float)
        xv, yv, zv = x[0], x[1], x[2]
        a0, b0, c0 = self.a[0], self.a[1], self.a[2]
        mu = self.mu

        A, B, C   = self.ABC
        F, G, H   = self.FGH      # FGH[0]=F, [1]=G, [2]=H
        F_, G_, H_ = self.FGH_    # FGH_ primed variants
        R, S, T   = self.RST      # RST[0]=R, [1]=S, [2]=T
        U, V, W   = self.UVW

        # print(f"ABC: A={A:.6f}, B={B:.6f}, C={C:.6f}")
        # print(f"FGH: F={F:.6f}, G={G:.6f}, H={H:.6f}")
        # print(f"FGH_: F_={F_:.6f}, G_={G_:.6f}, H_={H_:.6f}")
        # print(f"RST: R={R:.6f}, S={S:.6f}, T={T:.6f}")
        # print(f"UVW: U={U:.6f}, V={V:.6f}, W={W:.6f}")

        # --- strain components from epsilon_symm --------------------------
        # matching sigma code notation:
        # a_bold = eps_symm[0,0], b_bold = eps_symm[1,1], c_bold = eps_symm[2,2]
        # f_bold = eps_symm[1,2], g_bold = eps_symm[0,2], h_bold = eps_symm[0,1]
        # xii    = eps_anti[0,1], eta = eps_anti[0,2],    xi = eps_anti[1,2]  (careful signs)
        a_bold = self.epsilon_symm[0, 0]
        b_bold = self.epsilon_symm[1, 1]
        c_bold = self.epsilon_symm[2, 2]
        f_bold = self.epsilon_symm[1, 2]
        g_bold = self.epsilon_symm[0, 2]
        h_bold = self.epsilon_symm[0, 1]
        xii    =  self.epsilon_anti[2, 1]   # (1/2)(m2n3-m3n2)
        eta    =  self.epsilon_anti[0, 2]   # (1/2)(m3n1-m1n3)
        xi     =  self.epsilon_anti[1, 0]   # (1/2)(m1n2-m2n1)

        # --- lambda and geometric quantities ------------------------------
        lam = float(self.find_l0_jax(x))

        # I integrals at lam (use patched versions so surface mode is respected)
        # I_all_jax  -> self.I()  -> alpha, beta, gamma    (I  integrals)
        # I__all_jax -> self.I_() -> alpha', beta', gamma'   (I' integrals, single prime)
        I_vals  = np.array(self.I_all_jax(x))   # [alpha,  beta,  gamma ]
        Ip_vals = np.array(self.I__all_jax(x))  # [alpha', beta', gamma']

        al  = float(I_vals[0]);  be  = float(I_vals[1]);  ga  = float(I_vals[2])
        alp = float(Ip_vals[0]); bep = float(Ip_vals[1]); gap = float(Ip_vals[2])

        Delta_val = float(np.sqrt((a0**2+lam)*(b0**2+lam)*(c0**2+lam)))

        # P² = 1/F_coeff
        F_coeff = (xv**2/(a0**2+lam)**2 + yv**2/(b0**2+lam)**2 + zv**2/(c0**2+lam)**2)
        P2  = 1.0 / F_coeff
        P_  = np.sqrt(P2)   # P (used in dP/d* expressions)

        # dl/d*
        dl_dx = (2*xv*P2) / (a0**2+lam)
        dl_dy = (2*yv*P2) / (b0**2+lam)
        dl_dz = (2*zv*P2) / (c0**2+lam)

        # dP/d*
        dP_dx = (-0.5 * F_coeff**(-1.5) *
                 (2*xv/(a0**2+lam)**2
                  - 2*xv**2*dl_dx/(a0**2+lam)**3
                  - 2*yv**2*dl_dx/(b0**2+lam)**3
                  - 2*zv**2*dl_dx/(c0**2+lam)**3))
        dP_dy = (-0.5 * F_coeff**(-1.5) *
                 (2*yv/(b0**2+lam)**2
                  - 2*xv**2*dl_dy/(a0**2+lam)**3
                  - 2*yv**2*dl_dy/(b0**2+lam)**3
                  - 2*zv**2*dl_dy/(c0**2+lam)**3))
        dP_dz = (-0.5 * F_coeff**(-1.5) *
                 (2*zv/(c0**2+lam)**2
                  - 2*xv**2*dl_dz/(a0**2+lam)**3
                  - 2*yv**2*dl_dz/(b0**2+lam)**3
                  - 2*zv**2*dl_dz/(c0**2+lam)**3))

        # d(Delta^-1)/d*
        dDm_dx = (-0.5*((a0**2+lam)*(b0**2+lam)*(c0**2+lam))**(-1.5)
                  * dl_dx*((b0**2+lam)*(c0**2+lam)+(a0**2+lam)*(c0**2+lam)+(a0**2+lam)*(b0**2+lam)))
        dDm_dy = (-0.5*((a0**2+lam)*(b0**2+lam)*(c0**2+lam))**(-1.5)
                  * dl_dy*((b0**2+lam)*(c0**2+lam)+(a0**2+lam)*(c0**2+lam)+(a0**2+lam)*(b0**2+lam)))
        dDm_dz = (-0.5*((a0**2+lam)*(b0**2+lam)*(c0**2+lam))**(-1.5)
                  * dl_dz*((b0**2+lam)*(c0**2+lam)+(a0**2+lam)*(c0**2+lam)+(a0**2+lam)*(b0**2+lam)))

        # d(alpha)/d*, d(beta)/d*, d(gamma)/d*  (Leibniz boundary terms)
        dal_dx = -dl_dx / ((a0**2+lam)*Delta_val)
        dbe_dx = -dl_dx / ((b0**2+lam)*Delta_val)
        dga_dx = -dl_dx / ((c0**2+lam)*Delta_val)
        dal_dy = -dl_dy / ((a0**2+lam)*Delta_val)
        dbe_dy = -dl_dy / ((b0**2+lam)*Delta_val)
        dga_dy = -dl_dy / ((c0**2+lam)*Delta_val)
        dal_dz = -dl_dz / ((a0**2+lam)*Delta_val)
        dbe_dz = -dl_dz / ((b0**2+lam)*Delta_val)
        dga_dz = -dl_dz / ((c0**2+lam)*Delta_val)

        # d(alpha')/d*, d(beta')/d*, d(gamma')/d*
        dalp_dx = -dl_dx / ((b0**2+lam)*(c0**2+lam)*Delta_val)
        dbep_dx = -dl_dx / ((a0**2+lam)*(c0**2+lam)*Delta_val)
        dgap_dx = -dl_dx / ((a0**2+lam)*(b0**2+lam)*Delta_val)
        dalp_dy = -dl_dy / ((b0**2+lam)*(c0**2+lam)*Delta_val)
        dbep_dy = -dl_dy / ((a0**2+lam)*(c0**2+lam)*Delta_val)
        dgap_dy = -dl_dy / ((a0**2+lam)*(b0**2+lam)*Delta_val)
        dalp_dz = -dl_dz / ((b0**2+lam)*(c0**2+lam)*Delta_val)
        dbep_dz = -dl_dz / ((a0**2+lam)*(c0**2+lam)*Delta_val)
        dgap_dz = -dl_dz / ((a0**2+lam)*(b0**2+lam)*Delta_val)

        # shared bracket Q used in all velocity gradient components
        def Q(lm):
            return (((R + 2*(b0**2+lm)*F + 2*(c0**2 +lm)*F_)*yv*zv)/((b0**2+lm)*(c0**2+lm))
                    + ((S + 2*(c0**2+lm)*G + 2*(a0**2 +lm)*G_)*zv*xv)/((c0**2+lm)*(a0**2+lm))
                    + ((T + 2*(a0**2+lm)*H + 2*(b0**2 +lm)*H_)*xv*yv)/((a0**2+lm)*(b0**2+lm))
                    + (W - 2*(a0**2+lm)*A + 2*(b0**2 +lm)*B)*yv**2/((b0**2+lm)**2)
                    - (V - 2*(c0**2+lm)*C + 2*(a0**2 +lm)*A)*zv**2/((c0**2+lm)**2))

        def Qv(lm):   # Q variant for v-component
            return (((R + 2*(b0**2+lm)*F + 2*(c0**2 +lm)*F_)*yv*zv)/((b0**2+lm)*(c0**2+lm))
                    + ((S + 2*(c0**2+lm)*G + 2*(a0**2 +lm)*G_)*zv*xv)/((c0**2+lm)*(a0**2+lm))
                    + ((T + 2*(a0**2+lm)*H + 2*(b0**2 +lm)*H_)*xv*yv)/((a0**2+lm)*(b0**2+lm))
                    + (U - 2*(b0**2+lm)*B + 2*(c0**2 +lm)*C)*zv**2/((c0**2+lm)**2)
                    - (W - 2*(a0**2+lm)*A + 2*(b0**2 +lm)*B)*xv**2/((a0**2+lm)**2))


        def Qw(lm):   # Q variant for w-component
            return (((R + 2*(b0**2+lm)*F + 2*(c0**2 +lm)*F_)*yv*zv)/((b0**2+lm)*(c0**2+lm))
                    + ((S + 2*(c0**2+lm)*G + 2*(a0**2 +lm)*G_)*zv*xv)/((c0**2+lm)*(a0**2+lm))
                    + ((T + 2*(a0**2+lm)*H + 2*(b0**2 +lm)*H_)*xv*yv)/((a0**2+lm)*(b0**2+lm))
                    + (V - 2*(c0**2+lm)*C + 2*(a0**2 +lm)*A)*xv**2/((a0**2+lm)**2)
                    - (U - 2*(b0**2+lm)*B + 2*(c0**2 +lm)*C)*yv**2/((b0**2+lm)**2))


        # shared dQ/d* terms (d/dx of Q bracket, for u-component)
        def dQ_dx(lm):
            return ((2*F + 2*F_)*dl_dx*yv*zv/((b0**2+lm)*(c0**2+lm))
                    - ((R + 2*(b0**2+lm)*F + 2*(c0**2 +lm)*F_)*yv*zv*dl_dx)/((b0**2+lm)**2*(c0**2+lm))
                    - ((R + 2*(b0**2+lm)*F + 2*(c0**2 +lm)*F_)*yv*zv*dl_dx)/((b0**2+lm)*(c0**2+lm)**2) 
                    + ((S + 2*(c0**2+lm)*G + 2*(a0**2 +lm)*G_)*zv)/((c0**2+lm)*(a0**2+lm))
                    + ((2*G + 2*G_)*dl_dx*zv*xv)/((c0**2+lm)*(a0**2+lm))
                    - ((S + 2*(c0**2+lm)*G + 2*(a0**2 +lm)*G_)*zv*xv*dl_dx)/((c0**2+lm)**2*(a0**2+lm))
                    - ((S + 2*(c0**2+lm)*G + 2*(a0**2 +lm)*G_)*zv*xv*dl_dx)/((c0**2+lm)*(a0**2+lm)**2)
                    + ((T + 2*(a0**2+lm)*H + 2*(b0**2 +lm)*H_)*yv)/((a0**2+lm)*(b0**2+lm))
                    + ((2*H + 2*H_)*dl_dx*xv*yv)/((a0**2+lm)*(b0**2+lm))
                    - ((T + 2*(a0**2+lm)*H + 2*(b0**2 +lm)*H_)*xv*yv*dl_dx)/((a0**2+lm)**2*(b0**2+lm))
                    - ((T + 2*(a0**2+lm)*H + 2*(b0**2 +lm)*H_)*xv*yv*dl_dx)/((a0**2+lm)*(b0**2+lm)**2)
                    + (2*B-2*A)*yv**2*dl_dx/((b0**2+lm)**2)
                    - 2*dl_dx*(W - 2*(a0**2+lm)*A + 2*(b0**2 +lm)*B)*yv**2/((b0**2+lm)**3)
                    + (2*C-2*A)*zv**2*dl_dx/((c0**2+lm)**2)
                    + 2*dl_dx*(V - 2*(c0**2+lm)*C + 2*(a0**2 +lm)*A)*zv**2/((c0**2+lm)**3))

        def dQ_dy(lm):
            return (((R + 2*(b0**2+lm)*F + 2*(c0**2 +lm)*F_)*zv)/((b0**2+lm)*(c0**2+lm))
                    + (2*F + 2*F_)*dl_dy*yv*zv/((b0**2+lm)*(c0**2+lm))
                    - ((R + 2*(b0**2+lm)*F + 2*(c0**2 +lm)*F_)*yv*zv*dl_dy)/((b0**2+lm)**2*(c0**2+lm))
                    - ((R + 2*(b0**2+lm)*F + 2*(c0**2 +lm)*F_)*yv*zv*dl_dy)/((b0**2+lm)*(c0**2+lm)**2) 
                    + ((2*G + 2*G_)*dl_dy*zv*xv)/((c0**2+lm)*(a0**2+lm))
                    - ((S + 2*(c0**2+lm)*G + 2*(a0**2 +lm)*G_)*zv*xv*dl_dy)/((c0**2+lm)**2*(a0**2+lm))
                    - ((S + 2*(c0**2+lm)*G + 2*(a0**2 +lm)*G_)*zv*xv*dl_dy)/((c0**2+lm)*(a0**2+lm)**2)
                    + ((T + 2*(a0**2+lm)*H + 2*(b0**2 +lm)*H_)*xv)/((a0**2+lm)*(b0**2+lm))
                    + ((2*H + 2*H_)*dl_dy*xv*yv)/((a0**2+lm)*(b0**2+lm))
                    - ((T + 2*(a0**2+lm)*H + 2*(b0**2 +lm)*H_)*xv*yv*dl_dy)/((a0**2+lm)**2*(b0**2+lm))
                    - ((T + 2*(a0**2+lm)*H + 2*(b0**2 +lm)*H_)*xv*yv*dl_dy)/((a0**2+lm)*(b0**2+lm)**2)
                    + (W - 2*(a0**2+lm)*A + 2*(b0**2 +lm)*B)*2*yv/((b0**2+lm)**2)
                    + (2*B-2*A)*yv**2*dl_dy/((b0**2+lm)**2)
                    - 2*dl_dy*(W - 2*(a0**2+lm)*A + 2*(b0**2 +lm)*B)*yv**2/((b0**2+lm)**3)
                    + (2*C-2*A)*zv**2*dl_dy/((c0**2+lm)**2)
                    + 2*dl_dy*(V - 2*(c0**2+lm)*C + 2*(a0**2 +lm)*A)*zv**2/((c0**2+lm)**3))
                    
        def dQ_dz(lm):
            return (((R + 2*(b0**2+lm)*F + 2*(c0**2 +lm)*F_)*yv)/((b0**2+lm)*(c0**2+lm))
                    + (2*F + 2*F_)*dl_dz*yv*zv/((b0**2+lm)*(c0**2+lm))
                    - ((R + 2*(b0**2+lm)*F + 2*(c0**2 +lm)*F_)*yv*zv*dl_dz)/((b0**2+lm)**2*(c0**2+lm))
                    - ((R + 2*(b0**2+lm)*F + 2*(c0**2 +lm)*F_)*yv*zv*dl_dz)/((b0**2+lm)*(c0**2+lm)**2) 
                    + ((S + 2*(c0**2+lm)*G + 2*(a0**2 +lm)*G_)*xv)/((c0**2+lm)*(a0**2+lm))
                    + ((2*G + 2*G_)*dl_dz*zv*xv)/((c0**2+lm)*(a0**2+lm))
                    - ((S + 2*(c0**2+lm)*G + 2*(a0**2 +lm)*G_)*zv*xv*dl_dz)/((c0**2+lm)**2*(a0**2+lm))
                    - ((S + 2*(c0**2+lm)*G + 2*(a0**2 +lm)*G_)*zv*xv*dl_dz)/((c0**2+lm)*(a0**2+lm)**2)
                    + ((2*H + 2*H_)*dl_dz*xv*yv)/((a0**2+lm)*(b0**2+lm))
                    - ((T + 2*(a0**2+lm)*H + 2*(b0**2 +lm)*H_)*xv*yv*dl_dz)/((a0**2+lm)**2*(b0**2+lm))
                    - ((T + 2*(a0**2+lm)*H + 2*(b0**2 +lm)*H_)*xv*yv*dl_dz)/((a0**2+lm)*(b0**2+lm)**2)
                    + (2*B-2*A)*yv**2*dl_dz/((b0**2+lm)**2)
                    - 2*dl_dz*(W - 2*(a0**2+lm)*A + 2*(b0**2 +lm)*B)*yv**2/((b0**2+lm)**3)
                    - 2*(V - 2*(c0**2+lm)*C + 2*(a0**2 +lm)*A)*zv/((c0**2+lm)**2)
                    + (2*C-2*A)*zv**2*dl_dz/((c0**2+lm)**2)
                    + 2*dl_dz*(V - 2*(c0**2+lm)*C + 2*(a0**2 +lm)*A)*zv**2/((c0**2+lm)**3))

        # dQv/d* variants for v-component
        def dQv_dx(lm):
            return (((2*F + 2*F_)*dl_dx*yv*zv)/((b0**2+lm)*(c0**2+lm))
                    - ((R + 2*(b0**2+lm)*F + 2*(c0**2 +lm)*F_)*yv*zv*dl_dx)/((b0**2+lm)**2*(c0**2+lm))
                    - ((R + 2*(b0**2+lm)*F + 2*(c0**2 +lm)*F_)*yv*zv*dl_dx)/((b0**2+lm)*(c0**2+lm)**2)
                    + ((2*G + 2*G_)*dl_dx*zv*xv)/((c0**2+lm)*(a0**2+lm))
                    + ((S + 2*(c0**2+lm)*G + 2*(a0**2 +lm)*G_)*zv)/((c0**2+lm)*(a0**2+lm))
                    - ((S + 2*(c0**2+lm)*G + 2*(a0**2 +lm)*G_)*zv*xv*dl_dx)/((c0**2+lm)**2*(a0**2+lm))
                    - ((S + 2*(c0**2+lm)*G + 2*(a0**2 +lm)*G_)*zv*xv*dl_dx)/((c0**2+lm)*(a0**2+lm)**2)
                    + ((2*H + 2*H_)*dl_dx*xv*yv)/((a0**2+lm)*(b0**2+lm))
                    + ((T + 2*(a0**2+lm)*H + 2*(b0**2 +lm)*H_)*yv)/((a0**2+lm)*(b0**2+lm))
                    - ((T + 2*(a0**2+lm)*H + 2*(b0**2 +lm)*H_)*xv*yv*dl_dx)/((a0**2+lm)**2*(b0**2+lm))
                    - ((T + 2*(a0**2+lm)*H + 2*(b0**2 +lm)*H_)*xv*yv*dl_dx)/((a0**2+lm)*(b0**2+lm)**2)
                    + (2*C-2*B)*zv**2*dl_dx/((c0**2+lm)**2)
                    - 2*dl_dx*(U - 2*(b0**2+lm)*B + 2*(c0**2 +lm)*C)*zv**2/((c0**2+lm)**3)
                    + (2*A-2*B)*xv**2*dl_dx/((a0**2+lm)**2)
                    + 2*dl_dx*(W - 2*(a0**2+lm)*A + 2*(b0**2 +lm)*B)*xv**2/((a0**2+lm)**3)
                    - 2*(W - 2*(a0**2+lm)*A + 2*(b0**2 +lm)*B)*xv/((a0**2+lm)**2))

        def dQv_dy(lm):
            return (((R + 2*(b0**2+lm)*F + 2*(c0**2 +lm)*F_)*zv)/((b0**2+lm)*(c0**2+lm))
                    + ((2*F + 2*F_)*dl_dy*yv*zv)/((b0**2+lm)*(c0**2+lm))
                    - ((R + 2*(b0**2+lm)*F + 2*(c0**2 +lm)*F_)*yv*zv*dl_dy)/((b0**2+lm)**2*(c0**2+lm))
                    - ((R + 2*(b0**2+lm)*F + 2*(c0**2 +lm)*F_)*yv*zv*dl_dy)/((b0**2+lm)*(c0**2+lm)**2)
                    + ((2*G + 2*G_)*dl_dy*zv*xv)/((c0**2+lm)*(a0**2+lm))
                    - ((S + 2*(c0**2+lm)*G + 2*(a0**2 +lm)*G_)*zv*xv*dl_dy)/((c0**2+lm)**2*(a0**2+lm))
                    - ((S + 2*(c0**2+lm)*G + 2*(a0**2 +lm)*G_)*zv*xv*dl_dy)/((c0**2+lm)*(a0**2+lm)**2)
                    + ((2*H + 2*H_)*dl_dy*yv*xv)/((a0**2+lm)*(b0**2+lm))
                    + ((T + 2*(a0**2+lm)*H + 2*(b0**2 +lm)*H_)*xv)/((a0**2+lm)*(b0**2+lm))
                    - ((T + 2*(a0**2+lm)*H + 2*(b0**2 +lm)*H_)*xv*yv*dl_dy)/((a0**2+lm)**2*(b0**2+lm))
                    - ((T + 2*(a0**2+lm)*H + 2*(b0**2 +lm)*H_)*xv*yv*dl_dy)/((a0**2+lm)*(b0**2+lm)**2)
                    + (2*C-2*B)*zv**2*dl_dy/((c0**2+lm)**2)
                    - 2*dl_dy*(U - 2*(b0**2+lm)*B + 2*(c0**2 +lm)*C)*zv**2/((c0**2+lm)**3)
                    + (2*A-2*B)*xv**2*dl_dy/((a0**2+lm)**2)
                    + 2*dl_dy*(W - 2*(a0**2+lm)*A + 2*(b0**2 +lm)*B)*xv**2/((a0**2+lm)**3))

        def dQv_dz(lm):
            return (((R + 2*(b0**2+lm)*F + 2*(c0**2 +lm)*F_)*yv)/((b0**2+lm)*(c0**2+lm))
                    + ((2*F + 2*F_)*dl_dz*yv*zv)/((b0**2+lm)*(c0**2+lm))
                    - ((R + 2*(b0**2+lm)*F + 2*(c0**2 +lm)*F_)*yv*zv*dl_dz)/((b0**2+lm)**2*(c0**2+lm))
                    - ((R + 2*(b0**2+lm)*F + 2*(c0**2 +lm)*F_)*yv*zv*dl_dz)/((b0**2+lm)*(c0**2+lm)**2)
                    + ((S + 2*(c0**2+lm)*G + 2*(a0**2 +lm)*G_)*xv)/((c0**2+lm)*(a0**2+lm))
                    + ((2*G + 2*G_)*dl_dz*zv*xv)/((c0**2+lm)*(a0**2+lm))
                    - ((S + 2*(c0**2+lm)*G + 2*(a0**2 +lm)*G_)*zv*xv*dl_dz)/((c0**2+lm)**2*(a0**2+lm))
                    - ((S + 2*(c0**2+lm)*G + 2*(a0**2 +lm)*G_)*zv*xv*dl_dz)/((c0**2+lm)*(a0**2+lm)**2)
                    + ((2*H + 2*H_)*dl_dz*yv*xv)/((a0**2+lm)*(b0**2+lm))
                    - ((T + 2*(a0**2+lm)*H + 2*(b0**2 +lm)*H_)*xv*yv*dl_dz)/((a0**2+lm)**2*(b0**2+lm))
                    - ((T + 2*(a0**2+lm)*H + 2*(b0**2 +lm)*H_)*xv*yv*dl_dz)/((a0**2+lm)*(b0**2+lm)**2)
                    + 2*(U - 2*(b0**2+lm)*B + 2*(c0**2 +lm)*C)*zv/((c0**2+lm)**2)
                    + (2*C-2*B)*zv**2*dl_dz/((c0**2+lm)**2)
                    - 2*dl_dz*(U - 2*(b0**2+lm)*B + 2*(c0**2 +lm)*C)*zv**2/((c0**2+lm)**3)
                    + (2*A-2*B)*xv**2*dl_dz/((a0**2+lm)**2)
                    + 2*dl_dz*(W - 2*(a0**2+lm)*A + 2*(b0**2 +lm)*B)*xv**2/((a0**2+lm)**3))

        # dQw/d* variants for w-component
        def dQw_dx(lm):
            return (((2*F + 2*F_)*dl_dx*yv*zv)/((b0**2+lm)*(c0**2+lm))
                    - ((R + 2*(b0**2+lm)*F + 2*(c0**2 +lm)*F_)*yv*zv*dl_dx)/((b0**2+lm)**2*(c0**2+lm))
                    - ((R + 2*(b0**2+lm)*F + 2*(c0**2 +lm)*F_)*yv*zv*dl_dx)/((b0**2+lm)*(c0**2+lm)**2)
                    + ((S + 2*(c0**2+lm)*G + 2*(a0**2 +lm)*G_)*zv)/((c0**2+lm)*(a0**2+lm))
                    + ((2*G + 2*G_)*dl_dx*zv*xv)/((c0**2+lm)*(a0**2+lm))
                    - ((S + 2*(c0**2+lm)*G + 2*(a0**2 +lm)*G_)*zv*xv*dl_dx)/((c0**2+lm)**2*(a0**2+lm))
                    - ((S + 2*(c0**2+lm)*G + 2*(a0**2 +lm)*G_)*zv*xv*dl_dx)/((c0**2+lm)*(a0**2+lm)**2)
                    + ((T + 2*(a0**2+lm)*H + 2*(b0**2 +lm)*H_)*yv)/((a0**2+lm)*(b0**2+lm))
                    + ((2*H + 2*H_)*dl_dx*xv*yv)/((a0**2+lm)*(b0**2+lm))
                    - ((T + 2*(a0**2+lm)*H + 2*(b0**2 +lm)*H_)*xv*yv*dl_dx)/((a0**2+lm)**2*(b0**2+lm))
                    - ((T + 2*(a0**2+lm)*H + 2*(b0**2 +lm)*H_)*xv*yv*dl_dx)/((a0**2+lm)*(b0**2+lm)**2)
                    + (2*A-2*C)*xv**2*dl_dx/((a0**2+lm)**2)
                    - 2*dl_dx*(V - 2*(c0**2+lm)*C + 2*(a0**2 +lm)*A)*xv**2/((a0**2+lm)**3)
                    + 2*(V - 2*(c0**2+lm)*C + 2*(a0**2 +lm)*A)*xv/((a0**2+lm)**2)
                    + (2*B-2*C)*yv**2*dl_dx/((b0**2+lm)**2)
                    + 2*dl_dx*(U - 2*(b0**2+lm)*B + 2*(c0**2 +lm)*C)*yv**2/((b0**2+lm)**3))

        def dQw_dy(lm):
            return (((2*F + 2*F_)*dl_dy*yv*zv)/((b0**2+lm)*(c0**2+lm))
                    + ((R + 2*(b0**2+lm)*F + 2*(c0**2 +lm)*F_)*zv)/((b0**2+lm)*(c0**2+lm))
                    - ((R + 2*(b0**2+lm)*F + 2*(c0**2 +lm)*F_)*yv*zv*dl_dy)/((b0**2+lm)**2*(c0**2+lm))
                    - ((R + 2*(b0**2+lm)*F + 2*(c0**2 +lm)*F_)*yv*zv*dl_dy)/((b0**2+lm)*(c0**2+lm)**2)
                    + ((2*G + 2*G_)*dl_dy*zv*xv)/((c0**2+lm)*(a0**2+lm))
                    - ((S + 2*(c0**2+lm)*G + 2*(a0**2 +lm)*G_)*zv*xv*dl_dy)/((c0**2+lm)**2*(a0**2+lm))
                    - ((S + 2*(c0**2+lm)*G + 2*(a0**2 +lm)*G_)*zv*xv*dl_dy)/((c0**2+lm)*(a0**2+lm)**2)
                    + ((T + 2*(a0**2+lm)*H + 2*(b0**2 +lm)*H_)*xv)/((a0**2+lm)*(b0**2+lm))
                    + ((2*H + 2*H_)*dl_dy*xv*yv)/((a0**2+lm)*(b0**2+lm))
                    - ((T + 2*(a0**2+lm)*H + 2*(b0**2 +lm)*H_)*xv*yv*dl_dy)/((a0**2+lm)**2*(b0**2+lm))
                    - ((T + 2*(a0**2+lm)*H + 2*(b0**2 +lm)*H_)*xv*yv*dl_dy)/((a0**2+lm)*(b0**2+lm)**2)
                    + (2*A-2*C)*xv**2*dl_dy/((a0**2+lm)**2)
                    - 2*dl_dy*(V - 2*(c0**2+lm)*C + 2*(a0**2 +lm)*A)*xv**2/((a0**2+lm)**3)
                    + (2*B-2*C)*yv**2*dl_dy/((b0**2+lm)**2)
                    - 2*(U - 2*(b0**2+lm)*B + 2*(c0**2 +lm)*C)*yv/((b0**2+lm)**2)
                    + 2*dl_dy*(U - 2*(b0**2+lm)*B + 2*(c0**2 +lm)*C)*yv**2/((b0**2+lm)**3))

        def dQw_dz(lm):
            return (((2*F + 2*F_)*dl_dz*yv*zv)/((b0**2+lm)*(c0**2+lm))
                    + ((R + 2*(b0**2+lm)*F + 2*(c0**2 +lm)*F_)*yv)/((b0**2+lm)*(c0**2+lm))
                    - ((R + 2*(b0**2+lm)*F + 2*(c0**2 +lm)*F_)*yv*zv*dl_dz)/((b0**2+lm)**2*(c0**2+lm))
                    - ((R + 2*(b0**2+lm)*F + 2*(c0**2 +lm)*F_)*yv*zv*dl_dz)/((b0**2+lm)*(c0**2+lm)**2)
                    + ((S + 2*(c0**2+lm)*G + 2*(a0**2 +lm)*G_)*xv)/((c0**2+lm)*(a0**2+lm))
                    + ((2*G + 2*G_)*dl_dz*zv*xv)/((c0**2+lm)*(a0**2+lm))
                    - ((S + 2*(c0**2+lm)*G + 2*(a0**2 +lm)*G_)*zv*xv*dl_dz)/((c0**2+lm)**2*(a0**2+lm))
                    - ((S + 2*(c0**2+lm)*G + 2*(a0**2 +lm)*G_)*zv*xv*dl_dz)/((c0**2+lm)*(a0**2+lm)**2)
                    + ((2*H + 2*H_)*dl_dz*xv*yv)/((a0**2+lm)*(b0**2+lm))
                    - ((T + 2*(a0**2+lm)*H + 2*(b0**2 +lm)*H_)*xv*yv*dl_dz)/((a0**2+lm)**2*(b0**2+lm))
                    - ((T + 2*(a0**2+lm)*H + 2*(b0**2 +lm)*H_)*xv*yv*dl_dz)/((a0**2+lm)*(b0**2+lm)**2)
                    + (2*A-2*C)*xv**2*dl_dz/((a0**2+lm)**2)
                    - 2*dl_dz*(V - 2*(c0**2+lm)*C + 2*(a0**2 +lm)*A)*xv**2/((a0**2+lm)**3)
                    + (2*B-2*C)*yv**2*dl_dz/((b0**2+lm)**2)
                    + 2*dl_dz*(U - 2*(b0**2+lm)*B + 2*(c0**2 +lm)*C)*yv**2/((b0**2+lm)**3))

        # --- prefix factors for each row ---------------------------------
        px = (2*xv*P2) / ((a0**2+lam)*Delta_val)
        py = (2*yv*P2) / ((b0**2+lam)*Delta_val)
        pz = (2*zv*P2) / ((c0**2+lam)*Delta_val)

        dpx_dx = ((-2*P2)/((a0**2+lam)*Delta_val)
                  - (4*xv*P_*dP_dx)/((a0**2+lam)*Delta_val)
                  + (2*xv*P2*dl_dx)/((a0**2+lam)**2*Delta_val)
                  - (2*xv*P2*dDm_dx)/(a0**2+lam))
        dpx_dy = ((-4*xv*P_*dP_dy)/((a0**2+lam)*Delta_val)
                  + (2*xv*P2*dl_dy)/((a0**2+lam)**2*Delta_val)
                  - (2*xv*P2*dDm_dy)/(a0**2+lam))
        dpx_dz = ((-4*xv*P_*dP_dz)/((a0**2+lam)*Delta_val)
                  + (2*xv*P2*dl_dz)/((a0**2+lam)**2*Delta_val)
                  - (2*xv*P2*dDm_dz)/(a0**2+lam))

        dpy_dx = ((-4*yv*P_*dP_dx)/((b0**2+lam)*Delta_val)
                  + (2*yv*P2*dl_dx)/((b0**2+lam)**2*Delta_val)
                  - (2*yv*P2*dDm_dx)/(b0**2+lam))
        dpy_dy = ((-2*P2)/((b0**2+lam)*Delta_val)
                  - (4*yv*P_*dP_dy)/((b0**2+lam)*Delta_val)
                  + (2*yv*P2*dl_dy)/((b0**2+lam)**2*Delta_val)
                  - (2*yv*P2*dDm_dy)/(b0**2+lam))
        dpy_dz = ((-4*yv*P_*dP_dz)/((b0**2+lam)*Delta_val)
                  + (2*yv*P2*dl_dz)/((b0**2+lam)**2*Delta_val)
                  - (2*yv*P2*dDm_dz)/(b0**2+lam))

        dpz_dx = ((-4*zv*P_*dP_dx)/((c0**2+lam)*Delta_val)
                  + (2*zv*P2*dl_dx)/((c0**2+lam)**2*Delta_val)
                  - (2*zv*P2*dDm_dx)/(c0**2+lam))
        dpz_dy = ((-4*zv*P_*dP_dy)/((c0**2+lam)*Delta_val)
                  + (2*zv*P2*dl_dy)/((c0**2+lam)**2*Delta_val)
                  - (2*zv*P2*dDm_dy)/(c0**2+lam))
        dpz_dz = ((-2*P2)/((c0**2+lam)*Delta_val)
                  - (4*zv*P_*dP_dz)/((c0**2+lam)*Delta_val)
                  + (2*zv*P2*dl_dz)/((c0**2+lam)**2*Delta_val)
                  - (2*zv*P2*dDm_dz)/(c0**2+lam))

        # --- velocity gradients (du/dx etc.) directly from sigma code ----
        u_x = (a_bold + gap*W - bep*V - 2*(al+be+ga)*A
               + xv*W*dgap_dx - xv*V*dbep_dx
               - 2*A*xv*(dal_dx+dbe_dx+dga_dx)
               + yv*T*dgap_dx - 2*yv*H*dbe_dx + 2*yv*H_*dal_dx
               + zv*S*dbep_dx - 2*zv*G_*dga_dx + 2*zv*G*dal_dx
               + dpx_dx*Q(lam) - px*dQ_dx(lam))

        u_y = (xv*W*dgap_dy - xv*V*dbep_dy
               - 2*A*xv*(dal_dy+dbe_dy+dga_dy)
               + h_bold - xi + gap*T - 2*H*be + 2*al*H_
               + yv*T*dgap_dy - 2*yv*H*dbe_dy+2*yv*H_*dal_dy
               + zv*S*dbep_dy - 2*zv*G_*dga_dy + 2*zv*G*dal_dy
               + dpx_dy*Q(lam) - px*dQ_dy(lam))

        u_z = (xv*W*dgap_dz - xv*V*dbep_dz
               - 2*A*xv*(dal_dz+dbe_dz+dga_dz)
               + yv*T*dgap_dz - 2*yv*H*dbe_dz + 2*yv*H_*dal_dz
               + g_bold + eta + bep*S - 2*ga*G_ + 2*al*G
               + zv*S*dbep_dz - 2*zv*G_*dga_dz+ 2*zv*G*dal_dz
               + dpx_dz*Q(lam) - px*dQ_dz(lam))

        v_x = (h_bold + xi + gap*T + 2*H*be - 2*H_*al
               + xv*T*dgap_dx + 2*xv*H*dbe_dx-2*xv*H_*dal_dx
               + yv*U*dalp_dx - yv*W*dgap_dx
               - 2*yv*B*(dal_dx+dbe_dx+dga_dx)
               + zv*R*dalp_dx - 2*zv*F*dga_dx+2*zv*F_*dbe_dx
               + dpy_dx*Qv(lam) - py*dQv_dx(lam))

        v_y = (xv*T*dgap_dy + 2*xv*H*dbe_dy-2*xv*H_*dal_dy
               + b_bold + alp*U - gap*W - 2*(al+be+ga)*B
               + yv*U*dalp_dy - yv*W*dgap_dy
               - 2*yv*B*(dal_dy+dbe_dy+dga_dy)
               + zv*R*dalp_dy - 2*zv*F*dga_dy+2*zv*F_*dbe_dy
               + dpy_dy*Qv(lam) - py*dQv_dy(lam))

        v_z = (xv*T*dgap_dz + 2*xv*H*dbe_dz-2*xv*H_*dal_dz
               + yv*U*dalp_dz - yv*W*dgap_dz
               - 2*yv*B*(dal_dz+dbe_dz+dga_dz)
               + f_bold - xii + alp*R - 2*ga*F + 2*be*F_
               + zv*R*dalp_dz - 2*zv*F*dga_dz+2*zv*F_*dbe_dz
               + dpy_dz*Qv(lam) - py*dQv_dz(lam))

        w_x = (g_bold - eta + bep*S - 2*al*G+2*ga*G_
               + xv*S*dbep_dx - 2*xv*G*dal_dx+2*xv*G_*dga_dx
               + yv*R*dalp_dx + 2*yv*F*dga_dx-2*yv*F_*dbe_dx
               + zv*V*dbep_dx - zv*U*dalp_dx
               - 2*zv*C*(dal_dx+dbe_dx+dga_dx)
               + dpz_dx*Qw(lam) - pz*dQw_dx(lam))

        w_y = (xv*S*dbep_dy - 2*xv*G*dal_dy+2*xv*G_*dga_dy
               + f_bold + xii + alp*R + 2*F*ga-2*be*F_
               + yv*R*dalp_dy + 2*yv*F*dga_dy-2*yv*F_*dbe_dy
               + zv*V*dbep_dy - zv*U*dalp_dy
               - 2*zv*C*(dal_dy+dbe_dy+dga_dy)
               + dpz_dy*Qw(lam) - pz*dQw_dy(lam))

        w_z = (xv*S*dbep_dz - 2*xv*G*dal_dz+2*xv*G_*dga_dz
               + yv*R*dalp_dz + 2*yv*F*dga_dz-2*yv*F_*dbe_dz
               + c_bold + bep*V - alp*U - 2*C*(al+be+ga)
               + zv*V*dbep_dz - zv*U*dalp_dz
               - 2*zv*C*(dal_dz+dbe_dz+dga_dz)
               + dpz_dz*Qw(lam) - pz*dQw_dz(lam))

        # --- pressure -----------------------------------------------------
        # Use jeffery.p(x) directly — consistent with jacfwd and d2Om_dx2 formulation.
        # The sigma-code pressure formula uses different integral expressions
        # that are only equivalent for a sphere, not a general ellipsoid.
        press = float(self.p(jnp.array(x)))

        # --- assemble sigma = -p*I + mu*(du_dx + du_dx^T) ----------------
        # Note: mu already absorbed into press above (press = 2mu*...),
        # so sigma_ij = -press*delta_ij + mu*(du_i/dx_j + du_j/dx_i)
        # but your sigma code uses sigma = -p + 2*u_x (mu=1 implicit),
        # so here we keep mu explicit:
        sig = np.zeros((3, 3))


        #dudx_ref = jax.jacfwd(self.u)(jnp.array(x)) # add this line and the labels line below to check analytical sol
        #----------------------------------------------
        
        gradu = np.array([[u_x, u_y, u_z],
                          [v_x, v_y, v_z],
                          [w_x, w_y, w_z]])
        sig = -press * np.eye(3) + mu * (gradu + gradu.T)
        
        # labels = [['u_x','u_y','u_z'],['v_x','v_y','v_z'],['w_x','w_y','w_z']]
        # for i in range(3):
        #     for j in range(3):
        #         print(f"{labels[i][j]}: analytical={float(gradu[i,j]):.6f}, jacfwd={float(dudx_ref[i,j]):.6f}, diff={float(gradu[i,j]-dudx_ref[i,j]):.6f}")
        return sig
    
    def gradu(self, x):
        x = np.asarray(x, dtype=float)
        xv, yv, zv = x[0], x[1], x[2]
        a0, b0, c0 = self.a[0], self.a[1], self.a[2]
        mu = self.mu

        A, B, C   = self.ABC
        F, G, H   = self.FGH      # FGH[0]=F, [1]=G, [2]=H
        F_, G_, H_ = self.FGH_    # FGH_ primed variants
        R, S, T   = self.RST      # RST[0]=R, [1]=S, [2]=T
        U, V, W   = self.UVW

        # print(f"ABC: A={A:.6f}, B={B:.6f}, C={C:.6f}")
        # print(f"FGH: F={F:.6f}, G={G:.6f}, H={H:.6f}")
        # print(f"FGH_: F_={F_:.6f}, G_={G_:.6f}, H_={H_:.6f}")
        # print(f"RST: R={R:.6f}, S={S:.6f}, T={T:.6f}")
        # print(f"UVW: U={U:.6f}, V={V:.6f}, W={W:.6f}")

        # --- strain components from epsilon_symm --------------------------
        # matching sigma code notation:
        # a_bold = eps_symm[0,0], b_bold = eps_symm[1,1], c_bold = eps_symm[2,2]
        # f_bold = eps_symm[1,2], g_bold = eps_symm[0,2], h_bold = eps_symm[0,1]
        # xii    = eps_anti[0,1], eta = eps_anti[0,2],    xi = eps_anti[1,2]  (careful signs)
        a_bold = self.epsilon_symm[0, 0]
        b_bold = self.epsilon_symm[1, 1]
        c_bold = self.epsilon_symm[2, 2]
        f_bold = self.epsilon_symm[1, 2]
        g_bold = self.epsilon_symm[0, 2]
        h_bold = self.epsilon_symm[0, 1]
        xii    =  self.epsilon_anti[2, 1]   # (1/2)(m2n3-m3n2)
        eta    =  self.epsilon_anti[0, 2]   # (1/2)(m3n1-m1n3)
        xi     =  self.epsilon_anti[1, 0]   # (1/2)(m1n2-m2n1)

        # --- lambda and geometric quantities ------------------------------
        lam = float(self.find_l0_jax(x))

        # I integrals at lam (use patched versions so surface mode is respected)
        # I_all_jax  -> self.I()  -> alpha, beta, gamma    (I  integrals)
        # I__all_jax -> self.I_() -> alpha', beta', gamma'   (I' integrals, single prime)
        I_vals  = np.array(self.I_all_jax(x))   # [alpha,  beta,  gamma ]
        Ip_vals = np.array(self.I__all_jax(x))  # [alpha', beta', gamma']

        al  = float(I_vals[0]);  be  = float(I_vals[1]);  ga  = float(I_vals[2])
        alp = float(Ip_vals[0]); bep = float(Ip_vals[1]); gap = float(Ip_vals[2])

        Delta_val = float(np.sqrt((a0**2+lam)*(b0**2+lam)*(c0**2+lam)))

        # P² = 1/F_coeff
        F_coeff = (xv**2/(a0**2+lam)**2 + yv**2/(b0**2+lam)**2 + zv**2/(c0**2+lam)**2)
        P2  = 1.0 / F_coeff
        P_  = np.sqrt(P2)   # P (used in dP/d* expressions)

        # dl/d*
        dl_dx = (2*xv*P2) / (a0**2+lam)
        dl_dy = (2*yv*P2) / (b0**2+lam)
        dl_dz = (2*zv*P2) / (c0**2+lam)

        # dP/d*
        dP_dx = (-0.5 * F_coeff**(-1.5) *
                 (2*xv/(a0**2+lam)**2
                  - 2*xv**2*dl_dx/(a0**2+lam)**3
                  - 2*yv**2*dl_dx/(b0**2+lam)**3
                  - 2*zv**2*dl_dx/(c0**2+lam)**3))
        dP_dy = (-0.5 * F_coeff**(-1.5) *
                 (2*yv/(b0**2+lam)**2
                  - 2*xv**2*dl_dy/(a0**2+lam)**3
                  - 2*yv**2*dl_dy/(b0**2+lam)**3
                  - 2*zv**2*dl_dy/(c0**2+lam)**3))
        dP_dz = (-0.5 * F_coeff**(-1.5) *
                 (2*zv/(c0**2+lam)**2
                  - 2*xv**2*dl_dz/(a0**2+lam)**3
                  - 2*yv**2*dl_dz/(b0**2+lam)**3
                  - 2*zv**2*dl_dz/(c0**2+lam)**3))

        # d(Delta^-1)/d*
        dDm_dx = (-0.5*((a0**2+lam)*(b0**2+lam)*(c0**2+lam))**(-1.5)
                  * dl_dx*((b0**2+lam)*(c0**2+lam)+(a0**2+lam)*(c0**2+lam)+(a0**2+lam)*(b0**2+lam)))
        dDm_dy = (-0.5*((a0**2+lam)*(b0**2+lam)*(c0**2+lam))**(-1.5)
                  * dl_dy*((b0**2+lam)*(c0**2+lam)+(a0**2+lam)*(c0**2+lam)+(a0**2+lam)*(b0**2+lam)))
        dDm_dz = (-0.5*((a0**2+lam)*(b0**2+lam)*(c0**2+lam))**(-1.5)
                  * dl_dz*((b0**2+lam)*(c0**2+lam)+(a0**2+lam)*(c0**2+lam)+(a0**2+lam)*(b0**2+lam)))

        # d(alpha)/d*, d(beta)/d*, d(gamma)/d*  (Leibniz boundary terms)
        dal_dx = -dl_dx / ((a0**2+lam)*Delta_val)
        dbe_dx = -dl_dx / ((b0**2+lam)*Delta_val)
        dga_dx = -dl_dx / ((c0**2+lam)*Delta_val)
        dal_dy = -dl_dy / ((a0**2+lam)*Delta_val)
        dbe_dy = -dl_dy / ((b0**2+lam)*Delta_val)
        dga_dy = -dl_dy / ((c0**2+lam)*Delta_val)
        dal_dz = -dl_dz / ((a0**2+lam)*Delta_val)
        dbe_dz = -dl_dz / ((b0**2+lam)*Delta_val)
        dga_dz = -dl_dz / ((c0**2+lam)*Delta_val)

        # d(alpha')/d*, d(beta')/d*, d(gamma')/d*
        dalp_dx = -dl_dx / ((b0**2+lam)*(c0**2+lam)*Delta_val)
        dbep_dx = -dl_dx / ((a0**2+lam)*(c0**2+lam)*Delta_val)
        dgap_dx = -dl_dx / ((a0**2+lam)*(b0**2+lam)*Delta_val)
        dalp_dy = -dl_dy / ((b0**2+lam)*(c0**2+lam)*Delta_val)
        dbep_dy = -dl_dy / ((a0**2+lam)*(c0**2+lam)*Delta_val)
        dgap_dy = -dl_dy / ((a0**2+lam)*(b0**2+lam)*Delta_val)
        dalp_dz = -dl_dz / ((b0**2+lam)*(c0**2+lam)*Delta_val)
        dbep_dz = -dl_dz / ((a0**2+lam)*(c0**2+lam)*Delta_val)
        dgap_dz = -dl_dz / ((a0**2+lam)*(b0**2+lam)*Delta_val)

        # shared bracket Q used in all velocity gradient components
        def Q(lm):
            return (((R + 2*(b0**2+lm)*F + 2*(c0**2 +lm)*F_)*yv*zv)/((b0**2+lm)*(c0**2+lm))
                    + ((S + 2*(c0**2+lm)*G + 2*(a0**2 +lm)*G_)*zv*xv)/((c0**2+lm)*(a0**2+lm))
                    + ((T + 2*(a0**2+lm)*H + 2*(b0**2 +lm)*H_)*xv*yv)/((a0**2+lm)*(b0**2+lm))
                    + (W - 2*(a0**2+lm)*A + 2*(b0**2 +lm)*B)*yv**2/((b0**2+lm)**2)
                    - (V - 2*(c0**2+lm)*C + 2*(a0**2 +lm)*A)*zv**2/((c0**2+lm)**2))

        def Qv(lm):   # Q variant for v-component
            return (((R + 2*(b0**2+lm)*F + 2*(c0**2 +lm)*F_)*yv*zv)/((b0**2+lm)*(c0**2+lm))
                    + ((S + 2*(c0**2+lm)*G + 2*(a0**2 +lm)*G_)*zv*xv)/((c0**2+lm)*(a0**2+lm))
                    + ((T + 2*(a0**2+lm)*H + 2*(b0**2 +lm)*H_)*xv*yv)/((a0**2+lm)*(b0**2+lm))
                    + (U - 2*(b0**2+lm)*B + 2*(c0**2 +lm)*C)*zv**2/((c0**2+lm)**2)
                    - (W - 2*(a0**2+lm)*A + 2*(b0**2 +lm)*B)*xv**2/((a0**2+lm)**2))


        def Qw(lm):   # Q variant for w-component
            return (((R + 2*(b0**2+lm)*F + 2*(c0**2 +lm)*F_)*yv*zv)/((b0**2+lm)*(c0**2+lm))
                    + ((S + 2*(c0**2+lm)*G + 2*(a0**2 +lm)*G_)*zv*xv)/((c0**2+lm)*(a0**2+lm))
                    + ((T + 2*(a0**2+lm)*H + 2*(b0**2 +lm)*H_)*xv*yv)/((a0**2+lm)*(b0**2+lm))
                    + (V - 2*(c0**2+lm)*C + 2*(a0**2 +lm)*A)*xv**2/((a0**2+lm)**2)
                    - (U - 2*(b0**2+lm)*B + 2*(c0**2 +lm)*C)*yv**2/((b0**2+lm)**2))


        # shared dQ/d* terms (d/dx of Q bracket, for u-component)
        def dQ_dx(lm):
            return ((2*F + 2*F_)*dl_dx*yv*zv/((b0**2+lm)*(c0**2+lm))
                    - ((R + 2*(b0**2+lm)*F + 2*(c0**2 +lm)*F_)*yv*zv*dl_dx)/((b0**2+lm)**2*(c0**2+lm))
                    - ((R + 2*(b0**2+lm)*F + 2*(c0**2 +lm)*F_)*yv*zv*dl_dx)/((b0**2+lm)*(c0**2+lm)**2) 
                    + ((S + 2*(c0**2+lm)*G + 2*(a0**2 +lm)*G_)*zv)/((c0**2+lm)*(a0**2+lm))
                    + ((2*G + 2*G_)*dl_dx*zv*xv)/((c0**2+lm)*(a0**2+lm))
                    - ((S + 2*(c0**2+lm)*G + 2*(a0**2 +lm)*G_)*zv*xv*dl_dx)/((c0**2+lm)**2*(a0**2+lm))
                    - ((S + 2*(c0**2+lm)*G + 2*(a0**2 +lm)*G_)*zv*xv*dl_dx)/((c0**2+lm)*(a0**2+lm)**2)
                    + ((T + 2*(a0**2+lm)*H + 2*(b0**2 +lm)*H_)*yv)/((a0**2+lm)*(b0**2+lm))
                    + ((2*H + 2*H_)*dl_dx*xv*yv)/((a0**2+lm)*(b0**2+lm))
                    - ((T + 2*(a0**2+lm)*H + 2*(b0**2 +lm)*H_)*xv*yv*dl_dx)/((a0**2+lm)**2*(b0**2+lm))
                    - ((T + 2*(a0**2+lm)*H + 2*(b0**2 +lm)*H_)*xv*yv*dl_dx)/((a0**2+lm)*(b0**2+lm)**2)
                    + (2*B-2*A)*yv**2*dl_dx/((b0**2+lm)**2)
                    - 2*dl_dx*(W - 2*(a0**2+lm)*A + 2*(b0**2 +lm)*B)*yv**2/((b0**2+lm)**3)
                    + (2*C-2*A)*zv**2*dl_dx/((c0**2+lm)**2)
                    + 2*dl_dx*(V - 2*(c0**2+lm)*C + 2*(a0**2 +lm)*A)*zv**2/((c0**2+lm)**3))

        def dQ_dy(lm):
            return (((R + 2*(b0**2+lm)*F + 2*(c0**2 +lm)*F_)*zv)/((b0**2+lm)*(c0**2+lm))
                    + (2*F + 2*F_)*dl_dy*yv*zv/((b0**2+lm)*(c0**2+lm))
                    - ((R + 2*(b0**2+lm)*F + 2*(c0**2 +lm)*F_)*yv*zv*dl_dy)/((b0**2+lm)**2*(c0**2+lm))
                    - ((R + 2*(b0**2+lm)*F + 2*(c0**2 +lm)*F_)*yv*zv*dl_dy)/((b0**2+lm)*(c0**2+lm)**2) 
                    + ((2*G + 2*G_)*dl_dy*zv*xv)/((c0**2+lm)*(a0**2+lm))
                    - ((S + 2*(c0**2+lm)*G + 2*(a0**2 +lm)*G_)*zv*xv*dl_dy)/((c0**2+lm)**2*(a0**2+lm))
                    - ((S + 2*(c0**2+lm)*G + 2*(a0**2 +lm)*G_)*zv*xv*dl_dy)/((c0**2+lm)*(a0**2+lm)**2)
                    + ((T + 2*(a0**2+lm)*H + 2*(b0**2 +lm)*H_)*xv)/((a0**2+lm)*(b0**2+lm))
                    + ((2*H + 2*H_)*dl_dy*xv*yv)/((a0**2+lm)*(b0**2+lm))
                    - ((T + 2*(a0**2+lm)*H + 2*(b0**2 +lm)*H_)*xv*yv*dl_dy)/((a0**2+lm)**2*(b0**2+lm))
                    - ((T + 2*(a0**2+lm)*H + 2*(b0**2 +lm)*H_)*xv*yv*dl_dy)/((a0**2+lm)*(b0**2+lm)**2)
                    + (W - 2*(a0**2+lm)*A + 2*(b0**2 +lm)*B)*2*yv/((b0**2+lm)**2)
                    + (2*B-2*A)*yv**2*dl_dy/((b0**2+lm)**2)
                    - 2*dl_dy*(W - 2*(a0**2+lm)*A + 2*(b0**2 +lm)*B)*yv**2/((b0**2+lm)**3)
                    + (2*C-2*A)*zv**2*dl_dy/((c0**2+lm)**2)
                    + 2*dl_dy*(V - 2*(c0**2+lm)*C + 2*(a0**2 +lm)*A)*zv**2/((c0**2+lm)**3))
                    
        def dQ_dz(lm):
            return (((R + 2*(b0**2+lm)*F + 2*(c0**2 +lm)*F_)*yv)/((b0**2+lm)*(c0**2+lm))
                    + (2*F + 2*F_)*dl_dz*yv*zv/((b0**2+lm)*(c0**2+lm))
                    - ((R + 2*(b0**2+lm)*F + 2*(c0**2 +lm)*F_)*yv*zv*dl_dz)/((b0**2+lm)**2*(c0**2+lm))
                    - ((R + 2*(b0**2+lm)*F + 2*(c0**2 +lm)*F_)*yv*zv*dl_dz)/((b0**2+lm)*(c0**2+lm)**2) 
                    + ((S + 2*(c0**2+lm)*G + 2*(a0**2 +lm)*G_)*xv)/((c0**2+lm)*(a0**2+lm))
                    + ((2*G + 2*G_)*dl_dz*zv*xv)/((c0**2+lm)*(a0**2+lm))
                    - ((S + 2*(c0**2+lm)*G + 2*(a0**2 +lm)*G_)*zv*xv*dl_dz)/((c0**2+lm)**2*(a0**2+lm))
                    - ((S + 2*(c0**2+lm)*G + 2*(a0**2 +lm)*G_)*zv*xv*dl_dz)/((c0**2+lm)*(a0**2+lm)**2)
                    + ((2*H + 2*H_)*dl_dz*xv*yv)/((a0**2+lm)*(b0**2+lm))
                    - ((T + 2*(a0**2+lm)*H + 2*(b0**2 +lm)*H_)*xv*yv*dl_dz)/((a0**2+lm)**2*(b0**2+lm))
                    - ((T + 2*(a0**2+lm)*H + 2*(b0**2 +lm)*H_)*xv*yv*dl_dz)/((a0**2+lm)*(b0**2+lm)**2)
                    + (2*B-2*A)*yv**2*dl_dz/((b0**2+lm)**2)
                    - 2*dl_dz*(W - 2*(a0**2+lm)*A + 2*(b0**2 +lm)*B)*yv**2/((b0**2+lm)**3)
                    - 2*(V - 2*(c0**2+lm)*C + 2*(a0**2 +lm)*A)*zv/((c0**2+lm)**2)
                    + (2*C-2*A)*zv**2*dl_dz/((c0**2+lm)**2)
                    + 2*dl_dz*(V - 2*(c0**2+lm)*C + 2*(a0**2 +lm)*A)*zv**2/((c0**2+lm)**3))

        # dQv/d* variants for v-component
        def dQv_dx(lm):
            return (((2*F + 2*F_)*dl_dx*yv*zv)/((b0**2+lm)*(c0**2+lm))
                    - ((R + 2*(b0**2+lm)*F + 2*(c0**2 +lm)*F_)*yv*zv*dl_dx)/((b0**2+lm)**2*(c0**2+lm))
                    - ((R + 2*(b0**2+lm)*F + 2*(c0**2 +lm)*F_)*yv*zv*dl_dx)/((b0**2+lm)*(c0**2+lm)**2)
                    + ((2*G + 2*G_)*dl_dx*zv*xv)/((c0**2+lm)*(a0**2+lm))
                    + ((S + 2*(c0**2+lm)*G + 2*(a0**2 +lm)*G_)*zv)/((c0**2+lm)*(a0**2+lm))
                    - ((S + 2*(c0**2+lm)*G + 2*(a0**2 +lm)*G_)*zv*xv*dl_dx)/((c0**2+lm)**2*(a0**2+lm))
                    - ((S + 2*(c0**2+lm)*G + 2*(a0**2 +lm)*G_)*zv*xv*dl_dx)/((c0**2+lm)*(a0**2+lm)**2)
                    + ((2*H + 2*H_)*dl_dx*xv*yv)/((a0**2+lm)*(b0**2+lm))
                    + ((T + 2*(a0**2+lm)*H + 2*(b0**2 +lm)*H_)*yv)/((a0**2+lm)*(b0**2+lm))
                    - ((T + 2*(a0**2+lm)*H + 2*(b0**2 +lm)*H_)*xv*yv*dl_dx)/((a0**2+lm)**2*(b0**2+lm))
                    - ((T + 2*(a0**2+lm)*H + 2*(b0**2 +lm)*H_)*xv*yv*dl_dx)/((a0**2+lm)*(b0**2+lm)**2)
                    + (2*C-2*B)*zv**2*dl_dx/((c0**2+lm)**2)
                    - 2*dl_dx*(U - 2*(b0**2+lm)*B + 2*(c0**2 +lm)*C)*zv**2/((c0**2+lm)**3)
                    + (2*A-2*B)*xv**2*dl_dx/((a0**2+lm)**2)
                    + 2*dl_dx*(W - 2*(a0**2+lm)*A + 2*(b0**2 +lm)*B)*xv**2/((a0**2+lm)**3)
                    - 2*(W - 2*(a0**2+lm)*A + 2*(b0**2 +lm)*B)*xv/((a0**2+lm)**2))

        def dQv_dy(lm):
            return (((R + 2*(b0**2+lm)*F + 2*(c0**2 +lm)*F_)*zv)/((b0**2+lm)*(c0**2+lm))
                    + ((2*F + 2*F_)*dl_dy*yv*zv)/((b0**2+lm)*(c0**2+lm))
                    - ((R + 2*(b0**2+lm)*F + 2*(c0**2 +lm)*F_)*yv*zv*dl_dy)/((b0**2+lm)**2*(c0**2+lm))
                    - ((R + 2*(b0**2+lm)*F + 2*(c0**2 +lm)*F_)*yv*zv*dl_dy)/((b0**2+lm)*(c0**2+lm)**2)
                    + ((2*G + 2*G_)*dl_dy*zv*xv)/((c0**2+lm)*(a0**2+lm))
                    - ((S + 2*(c0**2+lm)*G + 2*(a0**2 +lm)*G_)*zv*xv*dl_dy)/((c0**2+lm)**2*(a0**2+lm))
                    - ((S + 2*(c0**2+lm)*G + 2*(a0**2 +lm)*G_)*zv*xv*dl_dy)/((c0**2+lm)*(a0**2+lm)**2)
                    + ((2*H + 2*H_)*dl_dy*yv*xv)/((a0**2+lm)*(b0**2+lm))
                    + ((T + 2*(a0**2+lm)*H + 2*(b0**2 +lm)*H_)*xv)/((a0**2+lm)*(b0**2+lm))
                    - ((T + 2*(a0**2+lm)*H + 2*(b0**2 +lm)*H_)*xv*yv*dl_dy)/((a0**2+lm)**2*(b0**2+lm))
                    - ((T + 2*(a0**2+lm)*H + 2*(b0**2 +lm)*H_)*xv*yv*dl_dy)/((a0**2+lm)*(b0**2+lm)**2)
                    + (2*C-2*B)*zv**2*dl_dy/((c0**2+lm)**2)
                    - 2*dl_dy*(U - 2*(b0**2+lm)*B + 2*(c0**2 +lm)*C)*zv**2/((c0**2+lm)**3)
                    + (2*A-2*B)*xv**2*dl_dy/((a0**2+lm)**2)
                    + 2*dl_dy*(W - 2*(a0**2+lm)*A + 2*(b0**2 +lm)*B)*xv**2/((a0**2+lm)**3))

        def dQv_dz(lm):
            return (((R + 2*(b0**2+lm)*F + 2*(c0**2 +lm)*F_)*yv)/((b0**2+lm)*(c0**2+lm))
                    + ((2*F + 2*F_)*dl_dz*yv*zv)/((b0**2+lm)*(c0**2+lm))
                    - ((R + 2*(b0**2+lm)*F + 2*(c0**2 +lm)*F_)*yv*zv*dl_dz)/((b0**2+lm)**2*(c0**2+lm))
                    - ((R + 2*(b0**2+lm)*F + 2*(c0**2 +lm)*F_)*yv*zv*dl_dz)/((b0**2+lm)*(c0**2+lm)**2)
                    + ((S + 2*(c0**2+lm)*G + 2*(a0**2 +lm)*G_)*xv)/((c0**2+lm)*(a0**2+lm))
                    + ((2*G + 2*G_)*dl_dz*zv*xv)/((c0**2+lm)*(a0**2+lm))
                    - ((S + 2*(c0**2+lm)*G + 2*(a0**2 +lm)*G_)*zv*xv*dl_dz)/((c0**2+lm)**2*(a0**2+lm))
                    - ((S + 2*(c0**2+lm)*G + 2*(a0**2 +lm)*G_)*zv*xv*dl_dz)/((c0**2+lm)*(a0**2+lm)**2)
                    + ((2*H + 2*H_)*dl_dz*yv*xv)/((a0**2+lm)*(b0**2+lm))
                    - ((T + 2*(a0**2+lm)*H + 2*(b0**2 +lm)*H_)*xv*yv*dl_dz)/((a0**2+lm)**2*(b0**2+lm))
                    - ((T + 2*(a0**2+lm)*H + 2*(b0**2 +lm)*H_)*xv*yv*dl_dz)/((a0**2+lm)*(b0**2+lm)**2)
                    + 2*(U - 2*(b0**2+lm)*B + 2*(c0**2 +lm)*C)*zv/((c0**2+lm)**2)
                    + (2*C-2*B)*zv**2*dl_dz/((c0**2+lm)**2)
                    - 2*dl_dz*(U - 2*(b0**2+lm)*B + 2*(c0**2 +lm)*C)*zv**2/((c0**2+lm)**3)
                    + (2*A-2*B)*xv**2*dl_dz/((a0**2+lm)**2)
                    + 2*dl_dz*(W - 2*(a0**2+lm)*A + 2*(b0**2 +lm)*B)*xv**2/((a0**2+lm)**3))

        # dQw/d* variants for w-component
        def dQw_dx(lm):
            return (((2*F + 2*F_)*dl_dx*yv*zv)/((b0**2+lm)*(c0**2+lm))
                    - ((R + 2*(b0**2+lm)*F + 2*(c0**2 +lm)*F_)*yv*zv*dl_dx)/((b0**2+lm)**2*(c0**2+lm))
                    - ((R + 2*(b0**2+lm)*F + 2*(c0**2 +lm)*F_)*yv*zv*dl_dx)/((b0**2+lm)*(c0**2+lm)**2)
                    + ((S + 2*(c0**2+lm)*G + 2*(a0**2 +lm)*G_)*zv)/((c0**2+lm)*(a0**2+lm))
                    + ((2*G + 2*G_)*dl_dx*zv*xv)/((c0**2+lm)*(a0**2+lm))
                    - ((S + 2*(c0**2+lm)*G + 2*(a0**2 +lm)*G_)*zv*xv*dl_dx)/((c0**2+lm)**2*(a0**2+lm))
                    - ((S + 2*(c0**2+lm)*G + 2*(a0**2 +lm)*G_)*zv*xv*dl_dx)/((c0**2+lm)*(a0**2+lm)**2)
                    + ((T + 2*(a0**2+lm)*H + 2*(b0**2 +lm)*H_)*yv)/((a0**2+lm)*(b0**2+lm))
                    + ((2*H + 2*H_)*dl_dx*xv*yv)/((a0**2+lm)*(b0**2+lm))
                    - ((T + 2*(a0**2+lm)*H + 2*(b0**2 +lm)*H_)*xv*yv*dl_dx)/((a0**2+lm)**2*(b0**2+lm))
                    - ((T + 2*(a0**2+lm)*H + 2*(b0**2 +lm)*H_)*xv*yv*dl_dx)/((a0**2+lm)*(b0**2+lm)**2)
                    + (2*A-2*C)*xv**2*dl_dx/((a0**2+lm)**2)
                    - 2*dl_dx*(V - 2*(c0**2+lm)*C + 2*(a0**2 +lm)*A)*xv**2/((a0**2+lm)**3)
                    + 2*(V - 2*(c0**2+lm)*C + 2*(a0**2 +lm)*A)*xv/((a0**2+lm)**2)
                    + (2*B-2*C)*yv**2*dl_dx/((b0**2+lm)**2)
                    + 2*dl_dx*(U - 2*(b0**2+lm)*B + 2*(c0**2 +lm)*C)*yv**2/((b0**2+lm)**3))

        def dQw_dy(lm):
            return (((2*F + 2*F_)*dl_dy*yv*zv)/((b0**2+lm)*(c0**2+lm))
                    + ((R + 2*(b0**2+lm)*F + 2*(c0**2 +lm)*F_)*zv)/((b0**2+lm)*(c0**2+lm))
                    - ((R + 2*(b0**2+lm)*F + 2*(c0**2 +lm)*F_)*yv*zv*dl_dy)/((b0**2+lm)**2*(c0**2+lm))
                    - ((R + 2*(b0**2+lm)*F + 2*(c0**2 +lm)*F_)*yv*zv*dl_dy)/((b0**2+lm)*(c0**2+lm)**2)
                    + ((2*G + 2*G_)*dl_dy*zv*xv)/((c0**2+lm)*(a0**2+lm))
                    - ((S + 2*(c0**2+lm)*G + 2*(a0**2 +lm)*G_)*zv*xv*dl_dy)/((c0**2+lm)**2*(a0**2+lm))
                    - ((S + 2*(c0**2+lm)*G + 2*(a0**2 +lm)*G_)*zv*xv*dl_dy)/((c0**2+lm)*(a0**2+lm)**2)
                    + ((T + 2*(a0**2+lm)*H + 2*(b0**2 +lm)*H_)*xv)/((a0**2+lm)*(b0**2+lm))
                    + ((2*H + 2*H_)*dl_dy*xv*yv)/((a0**2+lm)*(b0**2+lm))
                    - ((T + 2*(a0**2+lm)*H + 2*(b0**2 +lm)*H_)*xv*yv*dl_dy)/((a0**2+lm)**2*(b0**2+lm))
                    - ((T + 2*(a0**2+lm)*H + 2*(b0**2 +lm)*H_)*xv*yv*dl_dy)/((a0**2+lm)*(b0**2+lm)**2)
                    + (2*A-2*C)*xv**2*dl_dy/((a0**2+lm)**2)
                    - 2*dl_dy*(V - 2*(c0**2+lm)*C + 2*(a0**2 +lm)*A)*xv**2/((a0**2+lm)**3)
                    + (2*B-2*C)*yv**2*dl_dy/((b0**2+lm)**2)
                    - 2*(U - 2*(b0**2+lm)*B + 2*(c0**2 +lm)*C)*yv/((b0**2+lm)**2)
                    + 2*dl_dy*(U - 2*(b0**2+lm)*B + 2*(c0**2 +lm)*C)*yv**2/((b0**2+lm)**3))

        def dQw_dz(lm):
            return (((2*F + 2*F_)*dl_dz*yv*zv)/((b0**2+lm)*(c0**2+lm))
                    + ((R + 2*(b0**2+lm)*F + 2*(c0**2 +lm)*F_)*yv)/((b0**2+lm)*(c0**2+lm))
                    - ((R + 2*(b0**2+lm)*F + 2*(c0**2 +lm)*F_)*yv*zv*dl_dz)/((b0**2+lm)**2*(c0**2+lm))
                    - ((R + 2*(b0**2+lm)*F + 2*(c0**2 +lm)*F_)*yv*zv*dl_dz)/((b0**2+lm)*(c0**2+lm)**2)
                    + ((S + 2*(c0**2+lm)*G + 2*(a0**2 +lm)*G_)*xv)/((c0**2+lm)*(a0**2+lm))
                    + ((2*G + 2*G_)*dl_dz*zv*xv)/((c0**2+lm)*(a0**2+lm))
                    - ((S + 2*(c0**2+lm)*G + 2*(a0**2 +lm)*G_)*zv*xv*dl_dz)/((c0**2+lm)**2*(a0**2+lm))
                    - ((S + 2*(c0**2+lm)*G + 2*(a0**2 +lm)*G_)*zv*xv*dl_dz)/((c0**2+lm)*(a0**2+lm)**2)
                    + ((2*H + 2*H_)*dl_dz*xv*yv)/((a0**2+lm)*(b0**2+lm))
                    - ((T + 2*(a0**2+lm)*H + 2*(b0**2 +lm)*H_)*xv*yv*dl_dz)/((a0**2+lm)**2*(b0**2+lm))
                    - ((T + 2*(a0**2+lm)*H + 2*(b0**2 +lm)*H_)*xv*yv*dl_dz)/((a0**2+lm)*(b0**2+lm)**2)
                    + (2*A-2*C)*xv**2*dl_dz/((a0**2+lm)**2)
                    - 2*dl_dz*(V - 2*(c0**2+lm)*C + 2*(a0**2 +lm)*A)*xv**2/((a0**2+lm)**3)
                    + (2*B-2*C)*yv**2*dl_dz/((b0**2+lm)**2)
                    + 2*dl_dz*(U - 2*(b0**2+lm)*B + 2*(c0**2 +lm)*C)*yv**2/((b0**2+lm)**3))

        # --- prefix factors for each row ---------------------------------
        px = (2*xv*P2) / ((a0**2+lam)*Delta_val)
        py = (2*yv*P2) / ((b0**2+lam)*Delta_val)
        pz = (2*zv*P2) / ((c0**2+lam)*Delta_val)

        dpx_dx = ((-2*P2)/((a0**2+lam)*Delta_val)
                  - (4*xv*P_*dP_dx)/((a0**2+lam)*Delta_val)
                  + (2*xv*P2*dl_dx)/((a0**2+lam)**2*Delta_val)
                  - (2*xv*P2*dDm_dx)/(a0**2+lam))
        dpx_dy = ((-4*xv*P_*dP_dy)/((a0**2+lam)*Delta_val)
                  + (2*xv*P2*dl_dy)/((a0**2+lam)**2*Delta_val)
                  - (2*xv*P2*dDm_dy)/(a0**2+lam))
        dpx_dz = ((-4*xv*P_*dP_dz)/((a0**2+lam)*Delta_val)
                  + (2*xv*P2*dl_dz)/((a0**2+lam)**2*Delta_val)
                  - (2*xv*P2*dDm_dz)/(a0**2+lam))

        dpy_dx = ((-4*yv*P_*dP_dx)/((b0**2+lam)*Delta_val)
                  + (2*yv*P2*dl_dx)/((b0**2+lam)**2*Delta_val)
                  - (2*yv*P2*dDm_dx)/(b0**2+lam))
        dpy_dy = ((-2*P2)/((b0**2+lam)*Delta_val)
                  - (4*yv*P_*dP_dy)/((b0**2+lam)*Delta_val)
                  + (2*yv*P2*dl_dy)/((b0**2+lam)**2*Delta_val)
                  - (2*yv*P2*dDm_dy)/(b0**2+lam))
        dpy_dz = ((-4*yv*P_*dP_dz)/((b0**2+lam)*Delta_val)
                  + (2*yv*P2*dl_dz)/((b0**2+lam)**2*Delta_val)
                  - (2*yv*P2*dDm_dz)/(b0**2+lam))

        dpz_dx = ((-4*zv*P_*dP_dx)/((c0**2+lam)*Delta_val)
                  + (2*zv*P2*dl_dx)/((c0**2+lam)**2*Delta_val)
                  - (2*zv*P2*dDm_dx)/(c0**2+lam))
        dpz_dy = ((-4*zv*P_*dP_dy)/((c0**2+lam)*Delta_val)
                  + (2*zv*P2*dl_dy)/((c0**2+lam)**2*Delta_val)
                  - (2*zv*P2*dDm_dy)/(c0**2+lam))
        dpz_dz = ((-2*P2)/((c0**2+lam)*Delta_val)
                  - (4*zv*P_*dP_dz)/((c0**2+lam)*Delta_val)
                  + (2*zv*P2*dl_dz)/((c0**2+lam)**2*Delta_val)
                  - (2*zv*P2*dDm_dz)/(c0**2+lam))

        # --- velocity gradients (du/dx etc.) directly from sigma code ----
        u_x = (a_bold + gap*W - bep*V - 2*(al+be+ga)*A
               + xv*W*dgap_dx - xv*V*dbep_dx
               - 2*A*xv*(dal_dx+dbe_dx+dga_dx)
               + yv*T*dgap_dx - 2*yv*H*dbe_dx + 2*yv*H_*dal_dx
               + zv*S*dbep_dx - 2*zv*G_*dga_dx + 2*zv*G*dal_dx
               + dpx_dx*Q(lam) - px*dQ_dx(lam))

        u_y = (xv*W*dgap_dy - xv*V*dbep_dy
               - 2*A*xv*(dal_dy+dbe_dy+dga_dy)
               + h_bold - xi + gap*T - 2*H*be + 2*al*H_
               + yv*T*dgap_dy - 2*yv*H*dbe_dy+2*yv*H_*dal_dy
               + zv*S*dbep_dy - 2*zv*G_*dga_dy + 2*zv*G*dal_dy
               + dpx_dy*Q(lam) - px*dQ_dy(lam))

        u_z = (xv*W*dgap_dz - xv*V*dbep_dz
               - 2*A*xv*(dal_dz+dbe_dz+dga_dz)
               + yv*T*dgap_dz - 2*yv*H*dbe_dz + 2*yv*H_*dal_dz
               + g_bold + eta + bep*S - 2*ga*G_ + 2*al*G
               + zv*S*dbep_dz - 2*zv*G_*dga_dz+ 2*zv*G*dal_dz
               + dpx_dz*Q(lam) - px*dQ_dz(lam))

        v_x = (h_bold + xi + gap*T + 2*H*be - 2*H_*al
               + xv*T*dgap_dx + 2*xv*H*dbe_dx-2*xv*H_*dal_dx
               + yv*U*dalp_dx - yv*W*dgap_dx
               - 2*yv*B*(dal_dx+dbe_dx+dga_dx)
               + zv*R*dalp_dx - 2*zv*F*dga_dx+2*zv*F_*dbe_dx
               + dpy_dx*Qv(lam) - py*dQv_dx(lam))

        v_y = (xv*T*dgap_dy + 2*xv*H*dbe_dy-2*xv*H_*dal_dy
               + b_bold + alp*U - gap*W - 2*(al+be+ga)*B
               + yv*U*dalp_dy - yv*W*dgap_dy
               - 2*yv*B*(dal_dy+dbe_dy+dga_dy)
               + zv*R*dalp_dy - 2*zv*F*dga_dy+2*zv*F_*dbe_dy
               + dpy_dy*Qv(lam) - py*dQv_dy(lam))

        v_z = (xv*T*dgap_dz + 2*xv*H*dbe_dz-2*xv*H_*dal_dz
               + yv*U*dalp_dz - yv*W*dgap_dz
               - 2*yv*B*(dal_dz+dbe_dz+dga_dz)
               + f_bold - xii + alp*R - 2*ga*F + 2*be*F_
               + zv*R*dalp_dz - 2*zv*F*dga_dz+2*zv*F_*dbe_dz
               + dpy_dz*Qv(lam) - py*dQv_dz(lam))

        w_x = (g_bold - eta + bep*S - 2*al*G+2*ga*G_
               + xv*S*dbep_dx - 2*xv*G*dal_dx+2*xv*G_*dga_dx
               + yv*R*dalp_dx + 2*yv*F*dga_dx-2*yv*F_*dbe_dx
               + zv*V*dbep_dx - zv*U*dalp_dx
               - 2*zv*C*(dal_dx+dbe_dx+dga_dx)
               + dpz_dx*Qw(lam) - pz*dQw_dx(lam))

        w_y = (xv*S*dbep_dy - 2*xv*G*dal_dy+2*xv*G_*dga_dy
               + f_bold + xii + alp*R + 2*F*ga-2*be*F_
               + yv*R*dalp_dy + 2*yv*F*dga_dy-2*yv*F_*dbe_dy
               + zv*V*dbep_dy - zv*U*dalp_dy
               - 2*zv*C*(dal_dy+dbe_dy+dga_dy)
               + dpz_dy*Qw(lam) - pz*dQw_dy(lam))

        w_z = (xv*S*dbep_dz - 2*xv*G*dal_dz+2*xv*G_*dga_dz
               + yv*R*dalp_dz + 2*yv*F*dga_dz-2*yv*F_*dbe_dz
               + c_bold + bep*V - alp*U - 2*C*(al+be+ga)
               + zv*V*dbep_dz - zv*U*dalp_dz
               - 2*zv*C*(dal_dz+dbe_dz+dga_dz)
               + dpz_dz*Qw(lam) - pz*dQw_dz(lam))
        
        return np.array([[u_x, u_y, u_z],
                     [v_x, v_y, v_z],
                     [w_x, w_y, w_z]])

    def gradu_batch(self, points):
        """
        Compute velocity gradient at multiple points simultaneously.
        
        Parameters
        ----------
        points : np.ndarray, shape (N, 3)
            Surface points.
        
        Returns
        -------
        gradu : np.ndarray, shape (N, 3, 3)
            Velocity gradient at each point.
        """
        points = np.asarray(points, dtype=float)
        N = len(points)
        xv = points[:, 0]
        yv = points[:, 1]
        zv = points[:, 2]
        a0, b0, c0 = self.a[0], self.a[1], self.a[2]
        mu = self.mu

        A, B, C   = self.ABC
        F, G, H   = self.FGH
        F_, G_, H_ = self.FGH_
        R, S, T   = self.RST
        U, V, W   = self.UVW

        a_bold = self.epsilon_symm[0, 0]
        b_bold = self.epsilon_symm[1, 1]
        c_bold = self.epsilon_symm[2, 2]
        f_bold = self.epsilon_symm[1, 2]
        g_bold = self.epsilon_symm[0, 2]
        h_bold = self.epsilon_symm[0, 1]
        xii    =  self.epsilon_anti[2, 1]
        eta    =  self.epsilon_anti[0, 2]
        xi     =  self.epsilon_anti[1, 0]

        # lam = 0 on surface
        lam = np.zeros(N)

        Delta_val = np.sqrt((a0**2+lam)*(b0**2+lam)*(c0**2+lam))  # (N,)

        F_coeff = (xv**2/(a0**2+lam)**2 + yv**2/(b0**2+lam)**2 + zv**2/(c0**2+lam)**2)
        P2  = 1.0 / F_coeff
        P_  = np.sqrt(P2)

        dl_dx = (2*xv*P2) / (a0**2+lam)
        dl_dy = (2*yv*P2) / (b0**2+lam)
        dl_dz = (2*zv*P2) / (c0**2+lam)

        dP_dx = (-0.5 * F_coeff**(-1.5) *
                (2*xv/(a0**2+lam)**2
                - 2*xv**2*dl_dx/(a0**2+lam)**3
                - 2*yv**2*dl_dx/(b0**2+lam)**3
                - 2*zv**2*dl_dx/(c0**2+lam)**3))
        dP_dy = (-0.5 * F_coeff**(-1.5) *
                (2*yv/(b0**2+lam)**2
                - 2*xv**2*dl_dy/(a0**2+lam)**3
                - 2*yv**2*dl_dy/(b0**2+lam)**3
                - 2*zv**2*dl_dy/(c0**2+lam)**3))
        dP_dz = (-0.5 * F_coeff**(-1.5) *
                (2*zv/(c0**2+lam)**2
                - 2*xv**2*dl_dz/(a0**2+lam)**3
                - 2*yv**2*dl_dz/(b0**2+lam)**3
                - 2*zv**2*dl_dz/(c0**2+lam)**3))

        dDm_dx = (-0.5*((a0**2+lam)*(b0**2+lam)*(c0**2+lam))**(-1.5)
                * dl_dx*((b0**2+lam)*(c0**2+lam)+(a0**2+lam)*(c0**2+lam)+(a0**2+lam)*(b0**2+lam)))
        dDm_dy = (-0.5*((a0**2+lam)*(b0**2+lam)*(c0**2+lam))**(-1.5)
                * dl_dy*((b0**2+lam)*(c0**2+lam)+(a0**2+lam)*(c0**2+lam)+(a0**2+lam)*(b0**2+lam)))
        dDm_dz = (-0.5*((a0**2+lam)*(b0**2+lam)*(c0**2+lam))**(-1.5)
                * dl_dz*((b0**2+lam)*(c0**2+lam)+(a0**2+lam)*(c0**2+lam)+(a0**2+lam)*(b0**2+lam)))

        dal_dx = -dl_dx / ((a0**2+lam)*Delta_val)
        dbe_dx = -dl_dx / ((b0**2+lam)*Delta_val)
        dga_dx = -dl_dx / ((c0**2+lam)*Delta_val)
        dal_dy = -dl_dy / ((a0**2+lam)*Delta_val)
        dbe_dy = -dl_dy / ((b0**2+lam)*Delta_val)
        dga_dy = -dl_dy / ((c0**2+lam)*Delta_val)
        dal_dz = -dl_dz / ((a0**2+lam)*Delta_val)
        dbe_dz = -dl_dz / ((b0**2+lam)*Delta_val)
        dga_dz = -dl_dz / ((c0**2+lam)*Delta_val)

        dalp_dx = -dl_dx / ((b0**2+lam)*(c0**2+lam)*Delta_val)
        dbep_dx = -dl_dx / ((a0**2+lam)*(c0**2+lam)*Delta_val)
        dgap_dx = -dl_dx / ((a0**2+lam)*(b0**2+lam)*Delta_val)
        dalp_dy = -dl_dy / ((b0**2+lam)*(c0**2+lam)*Delta_val)
        dbep_dy = -dl_dy / ((a0**2+lam)*(c0**2+lam)*Delta_val)
        dgap_dy = -dl_dy / ((a0**2+lam)*(b0**2+lam)*Delta_val)
        dalp_dz = -dl_dz / ((b0**2+lam)*(c0**2+lam)*Delta_val)
        dbep_dz = -dl_dz / ((a0**2+lam)*(c0**2+lam)*Delta_val)
        dgap_dz = -dl_dz / ((a0**2+lam)*(b0**2+lam)*Delta_val)

        # integrals at lam (on surface, these are constants broadcast to (N,))
        I_vals  = np.array(self.I_all_jax(jnp.zeros(3)))
        Ip_vals = np.array(self.I__all_jax(jnp.zeros(3)))
        al  = float(I_vals[0]);  be  = float(I_vals[1]);  ga  = float(I_vals[2])
        alp = float(Ip_vals[0]); bep = float(Ip_vals[1]); gap = float(Ip_vals[2])

        # prefix factors, shape (N,)
        px = (2*xv*P2) / ((a0**2+lam)*Delta_val)
        py = (2*yv*P2) / ((b0**2+lam)*Delta_val)
        pz = (2*zv*P2) / ((c0**2+lam)*Delta_val)

        dpx_dx = ((-2*P2)/((a0**2+lam)*Delta_val)
                - (4*xv*P_*dP_dx)/((a0**2+lam)*Delta_val)
                + (2*xv*P2*dl_dx)/((a0**2+lam)**2*Delta_val)
                - (2*xv*P2*dDm_dx)/(a0**2+lam))
        dpx_dy = ((-4*xv*P_*dP_dy)/((a0**2+lam)*Delta_val)
                + (2*xv*P2*dl_dy)/((a0**2+lam)**2*Delta_val)
                - (2*xv*P2*dDm_dy)/(a0**2+lam))
        dpx_dz = ((-4*xv*P_*dP_dz)/((a0**2+lam)*Delta_val)
                + (2*xv*P2*dl_dz)/((a0**2+lam)**2*Delta_val)
                - (2*xv*P2*dDm_dz)/(a0**2+lam))

        dpy_dx = ((-4*yv*P_*dP_dx)/((b0**2+lam)*Delta_val)
                + (2*yv*P2*dl_dx)/((b0**2+lam)**2*Delta_val)
                - (2*yv*P2*dDm_dx)/(b0**2+lam))
        dpy_dy = ((-2*P2)/((b0**2+lam)*Delta_val)
                - (4*yv*P_*dP_dy)/((b0**2+lam)*Delta_val)
                + (2*yv*P2*dl_dy)/((b0**2+lam)**2*Delta_val)
                - (2*yv*P2*dDm_dy)/(b0**2+lam))
        dpy_dz = ((-4*yv*P_*dP_dz)/((b0**2+lam)*Delta_val)
                + (2*yv*P2*dl_dz)/((b0**2+lam)**2*Delta_val)
                - (2*yv*P2*dDm_dz)/(b0**2+lam))

        dpz_dx = ((-4*zv*P_*dP_dx)/((c0**2+lam)*Delta_val)
                + (2*zv*P2*dl_dx)/((c0**2+lam)**2*Delta_val)
                - (2*zv*P2*dDm_dx)/(c0**2+lam))
        dpz_dy = ((-4*zv*P_*dP_dy)/((c0**2+lam)*Delta_val)
                + (2*zv*P2*dl_dy)/((c0**2+lam)**2*Delta_val)
                - (2*zv*P2*dDm_dy)/(c0**2+lam))
        dpz_dz = ((-2*P2)/((c0**2+lam)*Delta_val)
                - (4*zv*P_*dP_dz)/((c0**2+lam)*Delta_val)
                + (2*zv*P2*dl_dz)/((c0**2+lam)**2*Delta_val)
                - (2*zv*P2*dDm_dz)/(c0**2+lam))

        # Q brackets, shape (N,)
        Q = (((R + 2*(b0**2+lam)*F + 2*(c0**2+lam)*F_)*yv*zv)/((b0**2+lam)*(c0**2+lam))
            + ((S + 2*(c0**2+lam)*G + 2*(a0**2+lam)*G_)*zv*xv)/((c0**2+lam)*(a0**2+lam))
            + ((T + 2*(a0**2+lam)*H + 2*(b0**2+lam)*H_)*xv*yv)/((a0**2+lam)*(b0**2+lam))
            + (W - 2*(a0**2+lam)*A + 2*(b0**2+lam)*B)*yv**2/((b0**2+lam)**2)
            - (V - 2*(c0**2+lam)*C + 2*(a0**2+lam)*A)*zv**2/((c0**2+lam)**2))

        Qv = (((R + 2*(b0**2+lam)*F + 2*(c0**2+lam)*F_)*yv*zv)/((b0**2+lam)*(c0**2+lam))
            + ((S + 2*(c0**2+lam)*G + 2*(a0**2+lam)*G_)*zv*xv)/((c0**2+lam)*(a0**2+lam))
            + ((T + 2*(a0**2+lam)*H + 2*(b0**2+lam)*H_)*xv*yv)/((a0**2+lam)*(b0**2+lam))
            + (U - 2*(b0**2+lam)*B + 2*(c0**2+lam)*C)*zv**2/((c0**2+lam)**2)
            - (W - 2*(a0**2+lam)*A + 2*(b0**2+lam)*B)*xv**2/((a0**2+lam)**2))

        Qw = (((R + 2*(b0**2+lam)*F + 2*(c0**2+lam)*F_)*yv*zv)/((b0**2+lam)*(c0**2+lam))
            + ((S + 2*(c0**2+lam)*G + 2*(a0**2+lam)*G_)*zv*xv)/((c0**2+lam)*(a0**2+lam))
            + ((T + 2*(a0**2+lam)*H + 2*(b0**2+lam)*H_)*xv*yv)/((a0**2+lam)*(b0**2+lam))
            + (V - 2*(c0**2+lam)*C + 2*(a0**2+lam)*A)*xv**2/((a0**2+lam)**2)
            - (U - 2*(b0**2+lam)*B + 2*(c0**2+lam)*C)*yv**2/((b0**2+lam)**2))

        dQ_dx = ((2*F + 2*F_)*dl_dx*yv*zv/((b0**2+lam)*(c0**2+lam))
                - ((R + 2*(b0**2+lam)*F + 2*(c0**2+lam)*F_)*yv*zv*dl_dx)/((b0**2+lam)**2*(c0**2+lam))
                - ((R + 2*(b0**2+lam)*F + 2*(c0**2+lam)*F_)*yv*zv*dl_dx)/((b0**2+lam)*(c0**2+lam)**2)
                + ((S + 2*(c0**2+lam)*G + 2*(a0**2+lam)*G_)*zv)/((c0**2+lam)*(a0**2+lam))
                + ((2*G + 2*G_)*dl_dx*zv*xv)/((c0**2+lam)*(a0**2+lam))
                - ((S + 2*(c0**2+lam)*G + 2*(a0**2+lam)*G_)*zv*xv*dl_dx)/((c0**2+lam)**2*(a0**2+lam))
                - ((S + 2*(c0**2+lam)*G + 2*(a0**2+lam)*G_)*zv*xv*dl_dx)/((c0**2+lam)*(a0**2+lam)**2)
                + ((T + 2*(a0**2+lam)*H + 2*(b0**2+lam)*H_)*yv)/((a0**2+lam)*(b0**2+lam))
                + ((2*H + 2*H_)*dl_dx*xv*yv)/((a0**2+lam)*(b0**2+lam))
                - ((T + 2*(a0**2+lam)*H + 2*(b0**2+lam)*H_)*xv*yv*dl_dx)/((a0**2+lam)**2*(b0**2+lam))
                - ((T + 2*(a0**2+lam)*H + 2*(b0**2+lam)*H_)*xv*yv*dl_dx)/((a0**2+lam)*(b0**2+lam)**2)
                + (2*B-2*A)*yv**2*dl_dx/((b0**2+lam)**2)
                - 2*dl_dx*(W - 2*(a0**2+lam)*A + 2*(b0**2+lam)*B)*yv**2/((b0**2+lam)**3)
                + (2*C-2*A)*zv**2*dl_dx/((c0**2+lam)**2)
                + 2*dl_dx*(V - 2*(c0**2+lam)*C + 2*(a0**2+lam)*A)*zv**2/((c0**2+lam)**3))

        dQ_dy = (((R + 2*(b0**2+lam)*F + 2*(c0**2+lam)*F_)*zv)/((b0**2+lam)*(c0**2+lam))
                + (2*F + 2*F_)*dl_dy*yv*zv/((b0**2+lam)*(c0**2+lam))
                - ((R + 2*(b0**2+lam)*F + 2*(c0**2+lam)*F_)*yv*zv*dl_dy)/((b0**2+lam)**2*(c0**2+lam))
                - ((R + 2*(b0**2+lam)*F + 2*(c0**2+lam)*F_)*yv*zv*dl_dy)/((b0**2+lam)*(c0**2+lam)**2)
                + ((2*G + 2*G_)*dl_dy*zv*xv)/((c0**2+lam)*(a0**2+lam))
                - ((S + 2*(c0**2+lam)*G + 2*(a0**2+lam)*G_)*zv*xv*dl_dy)/((c0**2+lam)**2*(a0**2+lam))
                - ((S + 2*(c0**2+lam)*G + 2*(a0**2+lam)*G_)*zv*xv*dl_dy)/((c0**2+lam)*(a0**2+lam)**2)
                + ((T + 2*(a0**2+lam)*H + 2*(b0**2+lam)*H_)*xv)/((a0**2+lam)*(b0**2+lam))
                + ((2*H + 2*H_)*dl_dy*xv*yv)/((a0**2+lam)*(b0**2+lam))
                - ((T + 2*(a0**2+lam)*H + 2*(b0**2+lam)*H_)*xv*yv*dl_dy)/((a0**2+lam)**2*(b0**2+lam))
                - ((T + 2*(a0**2+lam)*H + 2*(b0**2+lam)*H_)*xv*yv*dl_dy)/((a0**2+lam)*(b0**2+lam)**2)
                + (W - 2*(a0**2+lam)*A + 2*(b0**2+lam)*B)*2*yv/((b0**2+lam)**2)
                + (2*B-2*A)*yv**2*dl_dy/((b0**2+lam)**2)
                - 2*dl_dy*(W - 2*(a0**2+lam)*A + 2*(b0**2+lam)*B)*yv**2/((b0**2+lam)**3)
                + (2*C-2*A)*zv**2*dl_dy/((c0**2+lam)**2)
                + 2*dl_dy*(V - 2*(c0**2+lam)*C + 2*(a0**2+lam)*A)*zv**2/((c0**2+lam)**3))

        dQ_dz = (((R + 2*(b0**2+lam)*F + 2*(c0**2+lam)*F_)*yv)/((b0**2+lam)*(c0**2+lam))
                + (2*F + 2*F_)*dl_dz*yv*zv/((b0**2+lam)*(c0**2+lam))
                - ((R + 2*(b0**2+lam)*F + 2*(c0**2+lam)*F_)*yv*zv*dl_dz)/((b0**2+lam)**2*(c0**2+lam))
                - ((R + 2*(b0**2+lam)*F + 2*(c0**2+lam)*F_)*yv*zv*dl_dz)/((b0**2+lam)*(c0**2+lam)**2)
                + ((S + 2*(c0**2+lam)*G + 2*(a0**2+lam)*G_)*xv)/((c0**2+lam)*(a0**2+lam))
                + ((2*G + 2*G_)*dl_dz*zv*xv)/((c0**2+lam)*(a0**2+lam))
                - ((S + 2*(c0**2+lam)*G + 2*(a0**2+lam)*G_)*zv*xv*dl_dz)/((c0**2+lam)**2*(a0**2+lam))
                - ((S + 2*(c0**2+lam)*G + 2*(a0**2+lam)*G_)*zv*xv*dl_dz)/((c0**2+lam)*(a0**2+lam)**2)
                + ((2*H + 2*H_)*dl_dz*xv*yv)/((a0**2+lam)*(b0**2+lam))
                - ((T + 2*(a0**2+lam)*H + 2*(b0**2+lam)*H_)*xv*yv*dl_dz)/((a0**2+lam)**2*(b0**2+lam))
                - ((T + 2*(a0**2+lam)*H + 2*(b0**2+lam)*H_)*xv*yv*dl_dz)/((a0**2+lam)*(b0**2+lam)**2)
                + (2*B-2*A)*yv**2*dl_dz/((b0**2+lam)**2)
                - 2*dl_dz*(W - 2*(a0**2+lam)*A + 2*(b0**2+lam)*B)*yv**2/((b0**2+lam)**3)
                - 2*(V - 2*(c0**2+lam)*C + 2*(a0**2+lam)*A)*zv/((c0**2+lam)**2)
                + (2*C-2*A)*zv**2*dl_dz/((c0**2+lam)**2)
                + 2*dl_dz*(V - 2*(c0**2+lam)*C + 2*(a0**2+lam)*A)*zv**2/((c0**2+lam)**3))

        dQv_dx = (((2*F + 2*F_)*dl_dx*yv*zv)/((b0**2+lam)*(c0**2+lam))
                - ((R + 2*(b0**2+lam)*F + 2*(c0**2+lam)*F_)*yv*zv*dl_dx)/((b0**2+lam)**2*(c0**2+lam))
                - ((R + 2*(b0**2+lam)*F + 2*(c0**2+lam)*F_)*yv*zv*dl_dx)/((b0**2+lam)*(c0**2+lam)**2)
                + ((2*G + 2*G_)*dl_dx*zv*xv)/((c0**2+lam)*(a0**2+lam))
                + ((S + 2*(c0**2+lam)*G + 2*(a0**2+lam)*G_)*zv)/((c0**2+lam)*(a0**2+lam))
                - ((S + 2*(c0**2+lam)*G + 2*(a0**2+lam)*G_)*zv*xv*dl_dx)/((c0**2+lam)**2*(a0**2+lam))
                - ((S + 2*(c0**2+lam)*G + 2*(a0**2+lam)*G_)*zv*xv*dl_dx)/((c0**2+lam)*(a0**2+lam)**2)
                + ((2*H + 2*H_)*dl_dx*xv*yv)/((a0**2+lam)*(b0**2+lam))
                + ((T + 2*(a0**2+lam)*H + 2*(b0**2+lam)*H_)*yv)/((a0**2+lam)*(b0**2+lam))
                - ((T + 2*(a0**2+lam)*H + 2*(b0**2+lam)*H_)*xv*yv*dl_dx)/((a0**2+lam)**2*(b0**2+lam))
                - ((T + 2*(a0**2+lam)*H + 2*(b0**2+lam)*H_)*xv*yv*dl_dx)/((a0**2+lam)*(b0**2+lam)**2)
                + (2*C-2*B)*zv**2*dl_dx/((c0**2+lam)**2)
                - 2*dl_dx*(U - 2*(b0**2+lam)*B + 2*(c0**2+lam)*C)*zv**2/((c0**2+lam)**3)
                + (2*A-2*B)*xv**2*dl_dx/((a0**2+lam)**2)
                + 2*dl_dx*(W - 2*(a0**2+lam)*A + 2*(b0**2+lam)*B)*xv**2/((a0**2+lam)**3)
                - 2*(W - 2*(a0**2+lam)*A + 2*(b0**2+lam)*B)*xv/((a0**2+lam)**2))

        dQv_dy = (((R + 2*(b0**2+lam)*F + 2*(c0**2+lam)*F_)*zv)/((b0**2+lam)*(c0**2+lam))
                + ((2*F + 2*F_)*dl_dy*yv*zv)/((b0**2+lam)*(c0**2+lam))
                - ((R + 2*(b0**2+lam)*F + 2*(c0**2+lam)*F_)*yv*zv*dl_dy)/((b0**2+lam)**2*(c0**2+lam))
                - ((R + 2*(b0**2+lam)*F + 2*(c0**2+lam)*F_)*yv*zv*dl_dy)/((b0**2+lam)*(c0**2+lam)**2)
                + ((2*G + 2*G_)*dl_dy*zv*xv)/((c0**2+lam)*(a0**2+lam))
                - ((S + 2*(c0**2+lam)*G + 2*(a0**2+lam)*G_)*zv*xv*dl_dy)/((c0**2+lam)**2*(a0**2+lam))
                - ((S + 2*(c0**2+lam)*G + 2*(a0**2+lam)*G_)*zv*xv*dl_dy)/((c0**2+lam)*(a0**2+lam)**2)
                + ((2*H + 2*H_)*dl_dy*yv*xv)/((a0**2+lam)*(b0**2+lam))
                + ((T + 2*(a0**2+lam)*H + 2*(b0**2+lam)*H_)*xv)/((a0**2+lam)*(b0**2+lam))
                - ((T + 2*(a0**2+lam)*H + 2*(b0**2+lam)*H_)*xv*yv*dl_dy)/((a0**2+lam)**2*(b0**2+lam))
                - ((T + 2*(a0**2+lam)*H + 2*(b0**2+lam)*H_)*xv*yv*dl_dy)/((a0**2+lam)*(b0**2+lam)**2)
                + (2*C-2*B)*zv**2*dl_dy/((c0**2+lam)**2)
                - 2*dl_dy*(U - 2*(b0**2+lam)*B + 2*(c0**2+lam)*C)*zv**2/((c0**2+lam)**3)
                + (2*A-2*B)*xv**2*dl_dy/((a0**2+lam)**2)
                + 2*dl_dy*(W - 2*(a0**2+lam)*A + 2*(b0**2+lam)*B)*xv**2/((a0**2+lam)**3))

        dQv_dz = (((R + 2*(b0**2+lam)*F + 2*(c0**2+lam)*F_)*yv)/((b0**2+lam)*(c0**2+lam))
                + ((2*F + 2*F_)*dl_dz*yv*zv)/((b0**2+lam)*(c0**2+lam))
                - ((R + 2*(b0**2+lam)*F + 2*(c0**2+lam)*F_)*yv*zv*dl_dz)/((b0**2+lam)**2*(c0**2+lam))
                - ((R + 2*(b0**2+lam)*F + 2*(c0**2+lam)*F_)*yv*zv*dl_dz)/((b0**2+lam)*(c0**2+lam)**2)
                + ((S + 2*(c0**2+lam)*G + 2*(a0**2+lam)*G_)*xv)/((c0**2+lam)*(a0**2+lam))
                + ((2*G + 2*G_)*dl_dz*zv*xv)/((c0**2+lam)*(a0**2+lam))
                - ((S + 2*(c0**2+lam)*G + 2*(a0**2+lam)*G_)*zv*xv*dl_dz)/((c0**2+lam)**2*(a0**2+lam))
                - ((S + 2*(c0**2+lam)*G + 2*(a0**2+lam)*G_)*zv*xv*dl_dz)/((c0**2+lam)*(a0**2+lam)**2)
                + ((2*H + 2*H_)*dl_dz*yv*xv)/((a0**2+lam)*(b0**2+lam))
                - ((T + 2*(a0**2+lam)*H + 2*(b0**2+lam)*H_)*xv*yv*dl_dz)/((a0**2+lam)**2*(b0**2+lam))
                - ((T + 2*(a0**2+lam)*H + 2*(b0**2+lam)*H_)*xv*yv*dl_dz)/((a0**2+lam)*(b0**2+lam)**2)
                + 2*(U - 2*(b0**2+lam)*B + 2*(c0**2+lam)*C)*zv/((c0**2+lam)**2)
                + (2*C-2*B)*zv**2*dl_dz/((c0**2+lam)**2)
                - 2*dl_dz*(U - 2*(b0**2+lam)*B + 2*(c0**2+lam)*C)*zv**2/((c0**2+lam)**3)
                + (2*A-2*B)*xv**2*dl_dz/((a0**2+lam)**2)
                + 2*dl_dz*(W - 2*(a0**2+lam)*A + 2*(b0**2+lam)*B)*xv**2/((a0**2+lam)**3))

        dQw_dx = (((2*F + 2*F_)*dl_dx*yv*zv)/((b0**2+lam)*(c0**2+lam))
                - ((R + 2*(b0**2+lam)*F + 2*(c0**2+lam)*F_)*yv*zv*dl_dx)/((b0**2+lam)**2*(c0**2+lam))
                - ((R + 2*(b0**2+lam)*F + 2*(c0**2+lam)*F_)*yv*zv*dl_dx)/((b0**2+lam)*(c0**2+lam)**2)
                + ((S + 2*(c0**2+lam)*G + 2*(a0**2+lam)*G_)*zv)/((c0**2+lam)*(a0**2+lam))
                + ((2*G + 2*G_)*dl_dx*zv*xv)/((c0**2+lam)*(a0**2+lam))
                - ((S + 2*(c0**2+lam)*G + 2*(a0**2+lam)*G_)*zv*xv*dl_dx)/((c0**2+lam)**2*(a0**2+lam))
                - ((S + 2*(c0**2+lam)*G + 2*(a0**2+lam)*G_)*zv*xv*dl_dx)/((c0**2+lam)*(a0**2+lam)**2)
                + ((T + 2*(a0**2+lam)*H + 2*(b0**2+lam)*H_)*yv)/((a0**2+lam)*(b0**2+lam))
                + ((2*H + 2*H_)*dl_dx*xv*yv)/((a0**2+lam)*(b0**2+lam))
                - ((T + 2*(a0**2+lam)*H + 2*(b0**2+lam)*H_)*xv*yv*dl_dx)/((a0**2+lam)**2*(b0**2+lam))
                - ((T + 2*(a0**2+lam)*H + 2*(b0**2+lam)*H_)*xv*yv*dl_dx)/((a0**2+lam)*(b0**2+lam)**2)
                + (2*A-2*C)*xv**2*dl_dx/((a0**2+lam)**2)
                - 2*dl_dx*(V - 2*(c0**2+lam)*C + 2*(a0**2+lam)*A)*xv**2/((a0**2+lam)**3)
                + 2*(V - 2*(c0**2+lam)*C + 2*(a0**2+lam)*A)*xv/((a0**2+lam)**2)
                + (2*B-2*C)*yv**2*dl_dx/((b0**2+lam)**2)
                + 2*dl_dx*(U - 2*(b0**2+lam)*B + 2*(c0**2+lam)*C)*yv**2/((b0**2+lam)**3))

        dQw_dy = (((2*F + 2*F_)*dl_dy*yv*zv)/((b0**2+lam)*(c0**2+lam))
                + ((R + 2*(b0**2+lam)*F + 2*(c0**2+lam)*F_)*zv)/((b0**2+lam)*(c0**2+lam))
                - ((R + 2*(b0**2+lam)*F + 2*(c0**2+lam)*F_)*yv*zv*dl_dy)/((b0**2+lam)**2*(c0**2+lam))
                - ((R + 2*(b0**2+lam)*F + 2*(c0**2+lam)*F_)*yv*zv*dl_dy)/((b0**2+lam)*(c0**2+lam)**2)
                + ((2*G + 2*G_)*dl_dy*zv*xv)/((c0**2+lam)*(a0**2+lam))
                - ((S + 2*(c0**2+lam)*G + 2*(a0**2+lam)*G_)*zv*xv*dl_dy)/((c0**2+lam)**2*(a0**2+lam))
                - ((S + 2*(c0**2+lam)*G + 2*(a0**2+lam)*G_)*zv*xv*dl_dy)/((c0**2+lam)*(a0**2+lam)**2)
                + ((T + 2*(a0**2+lam)*H + 2*(b0**2+lam)*H_)*xv)/((a0**2+lam)*(b0**2+lam))
                + ((2*H + 2*H_)*dl_dy*xv*yv)/((a0**2+lam)*(b0**2+lam))
                - ((T + 2*(a0**2+lam)*H + 2*(b0**2+lam)*H_)*xv*yv*dl_dy)/((a0**2+lam)**2*(b0**2+lam))
                - ((T + 2*(a0**2+lam)*H + 2*(b0**2+lam)*H_)*xv*yv*dl_dy)/((a0**2+lam)*(b0**2+lam)**2)
                + (2*A-2*C)*xv**2*dl_dy/((a0**2+lam)**2)
                - 2*dl_dy*(V - 2*(c0**2+lam)*C + 2*(a0**2+lam)*A)*xv**2/((a0**2+lam)**3)
                + (2*B-2*C)*yv**2*dl_dy/((b0**2+lam)**2)
                - 2*(U - 2*(b0**2+lam)*B + 2*(c0**2+lam)*C)*yv/((b0**2+lam)**2)
                + 2*dl_dy*(U - 2*(b0**2+lam)*B + 2*(c0**2+lam)*C)*yv**2/((b0**2+lam)**3))

        dQw_dz = (((2*F + 2*F_)*dl_dz*yv*zv)/((b0**2+lam)*(c0**2+lam))
                + ((R + 2*(b0**2+lam)*F + 2*(c0**2+lam)*F_)*yv)/((b0**2+lam)*(c0**2+lam))
                - ((R + 2*(b0**2+lam)*F + 2*(c0**2+lam)*F_)*yv*zv*dl_dz)/((b0**2+lam)**2*(c0**2+lam))
                - ((R + 2*(b0**2+lam)*F + 2*(c0**2+lam)*F_)*yv*zv*dl_dz)/((b0**2+lam)*(c0**2+lam)**2)
                + ((S + 2*(c0**2+lam)*G + 2*(a0**2+lam)*G_)*xv)/((c0**2+lam)*(a0**2+lam))
                + ((2*G + 2*G_)*dl_dz*zv*xv)/((c0**2+lam)*(a0**2+lam))
                - ((S + 2*(c0**2+lam)*G + 2*(a0**2+lam)*G_)*zv*xv*dl_dz)/((c0**2+lam)**2*(a0**2+lam))
                - ((S + 2*(c0**2+lam)*G + 2*(a0**2+lam)*G_)*zv*xv*dl_dz)/((c0**2+lam)*(a0**2+lam)**2)
                + ((2*H + 2*H_)*dl_dz*xv*yv)/((a0**2+lam)*(b0**2+lam))
                - ((T + 2*(a0**2+lam)*H + 2*(b0**2+lam)*H_)*xv*yv*dl_dz)/((a0**2+lam)**2*(b0**2+lam))
                - ((T + 2*(a0**2+lam)*H + 2*(b0**2+lam)*H_)*xv*yv*dl_dz)/((a0**2+lam)*(b0**2+lam)**2)
                + (2*A-2*C)*xv**2*dl_dz/((a0**2+lam)**2)
                - 2*dl_dz*(V - 2*(c0**2+lam)*C + 2*(a0**2+lam)*A)*xv**2/((a0**2+lam)**3)
                + (2*B-2*C)*yv**2*dl_dz/((b0**2+lam)**2)
                + 2*dl_dz*(U - 2*(b0**2+lam)*B + 2*(c0**2+lam)*C)*yv**2/((b0**2+lam)**3))

        # velocity gradients, all shape (N,)
        u_x = (a_bold + gap*W - bep*V - 2*(al+be+ga)*A
            + xv*W*dgap_dx - xv*V*dbep_dx
            - 2*A*xv*(dal_dx+dbe_dx+dga_dx)
            + yv*T*dgap_dx - 2*yv*H*dbe_dx + 2*yv*H_*dal_dx
            + zv*S*dbep_dx - 2*zv*G_*dga_dx + 2*zv*G*dal_dx
            + dpx_dx*Q - px*dQ_dx)

        u_y = (xv*W*dgap_dy - xv*V*dbep_dy
            - 2*A*xv*(dal_dy+dbe_dy+dga_dy)
            + h_bold - xi + gap*T - 2*H*be + 2*al*H_
            + yv*T*dgap_dy - 2*yv*H*dbe_dy + 2*yv*H_*dal_dy
            + zv*S*dbep_dy - 2*zv*G_*dga_dy + 2*zv*G*dal_dy
            + dpx_dy*Q - px*dQ_dy)

        u_z = (xv*W*dgap_dz - xv*V*dbep_dz
            - 2*A*xv*(dal_dz+dbe_dz+dga_dz)
            + yv*T*dgap_dz - 2*yv*H*dbe_dz + 2*yv*H_*dal_dz
            + g_bold + eta + bep*S - 2*ga*G_ + 2*al*G
            + zv*S*dbep_dz - 2*zv*G_*dga_dz + 2*zv*G*dal_dz
            + dpx_dz*Q - px*dQ_dz)

        v_x = (h_bold + xi + gap*T + 2*H*be - 2*H_*al
            + xv*T*dgap_dx + 2*xv*H*dbe_dx - 2*xv*H_*dal_dx
            + yv*U*dalp_dx - yv*W*dgap_dx
            - 2*yv*B*(dal_dx+dbe_dx+dga_dx)
            + zv*R*dalp_dx - 2*zv*F*dga_dx + 2*zv*F_*dbe_dx
            + dpy_dx*Qv - py*dQv_dx)

        v_y = (xv*T*dgap_dy + 2*xv*H*dbe_dy - 2*xv*H_*dal_dy
            + b_bold + alp*U - gap*W - 2*(al+be+ga)*B
            + yv*U*dalp_dy - yv*W*dgap_dy
            - 2*yv*B*(dal_dy+dbe_dy+dga_dy)
            + zv*R*dalp_dy - 2*zv*F*dga_dy + 2*zv*F_*dbe_dy
            + dpy_dy*Qv - py*dQv_dy)

        v_z = (xv*T*dgap_dz + 2*xv*H*dbe_dz - 2*xv*H_*dal_dz
            + yv*U*dalp_dz - yv*W*dgap_dz
            - 2*yv*B*(dal_dz+dbe_dz+dga_dz)
            + f_bold - xii + alp*R - 2*ga*F + 2*be*F_
            + zv*R*dalp_dz - 2*zv*F*dga_dz + 2*zv*F_*dbe_dz
            + dpy_dz*Qv - py*dQv_dz)

        w_x = (g_bold - eta + bep*S - 2*al*G + 2*ga*G_
            + xv*S*dbep_dx - 2*xv*G*dal_dx + 2*xv*G_*dga_dx
            + yv*R*dalp_dx + 2*yv*F*dga_dx - 2*yv*F_*dbe_dx
            + zv*V*dbep_dx - zv*U*dalp_dx
            - 2*zv*C*(dal_dx+dbe_dx+dga_dx)
            + dpz_dx*Qw - pz*dQw_dx)

        w_y = (xv*S*dbep_dy - 2*xv*G*dal_dy + 2*xv*G_*dga_dy
            + f_bold + xii + alp*R + 2*F*ga - 2*be*F_
            + yv*R*dalp_dy + 2*yv*F*dga_dy - 2*yv*F_*dbe_dy
            + zv*V*dbep_dy - zv*U*dalp_dy
            - 2*zv*C*(dal_dy+dbe_dy+dga_dy)
            + dpz_dy*Qw - pz*dQw_dy)

        w_z = (xv*S*dbep_dz - 2*xv*G*dal_dz + 2*xv*G_*dga_dz
            + yv*R*dalp_dz + 2*yv*F*dga_dz - 2*yv*F_*dbe_dz
            + c_bold + bep*V - alp*U - 2*C*(al+be+ga)
            + zv*V*dbep_dz - zv*U*dalp_dz
            - 2*zv*C*(dal_dz+dbe_dz+dga_dz)
            + dpz_dz*Qw - pz*dQw_dz)

        # assemble (N, 3, 3)
        gradu = np.stack([
            np.stack([u_x, u_y, u_z], axis=1),
            np.stack([v_x, v_y, v_z], axis=1),
            np.stack([w_x, w_y, w_z], axis=1),
        ], axis=1)

        return gradu

    def restore_full_mode(self):
        """Restore the original (off-surface) integral evaluators."""
        self.find_l0_jax = self._orig_find_l0_jax
        self.I_all_jax   = self._orig_I_all_jax
        self.I__all_jax  = self._orig_I__all_jax
        self.I_jax       = self._orig_I_jax

    # ------------------------------------------------------------------
    # Vectorised stress transfer — fast path for surface-grid computations
    # ------------------------------------------------------------------

    def _coefs_from_epsilon(self, epsilon):
        """
        Compute all stress coefficients (ABC, FGH, FGH_, RST, UVW, omega)
        from a given epsilon WITHOUT mutating any instance state.

        All quantities are linear in epsilon, so this is the safe stateless
        version used when cycling through basis strains.

        Returns
        -------
        dict with keys: A,B,C, F,G,H, F_,G_,H_, R,S,T, U,V,W,
                        a_bold,b_bold,c_bold,f_bold,g_bold,h_bold, xii,eta,xi
        """
        eps  = np.asarray(epsilon, dtype=float)
        es   = 0.5 * (eps + eps.T)   # symmetric part
        ea   = 0.5 * (eps - eps.T)   # antisymmetric part

        a = self.a
        I0   = self.I0    # shape (3,) — depend only on a, precomputed
        I0_  = self.I0_
        I0__ = self.I0__

        al0, be0, ga0   = I0
        alp0, bep0, gap0 = I0_
        a0, b0, c0 = a[0], a[1], a[2]

        # --- ABC (linear in diagonal of epsilon_symm) ---
        denom_ABC = I0__[1]*I0__[2] + I0__[2]*I0__[0] + I0__[0]*I0__[1]
        A = (1/6) * (2*I0__[0]*es[0,0] - I0__[1]*es[1,1] - I0__[2]*es[2,2]) / denom_ABC
        B = (1/6) * (2*I0__[1]*es[1,1] - I0__[2]*es[2,2] - I0__[0]*es[0,0]) / denom_ABC
        C = (1/6) * (2*I0__[2]*es[2,2] - I0__[0]*es[0,0] - I0__[1]*es[1,1]) / denom_ABC

        # --- omega (jeffery) ---
        om1 = ea[2,1] + (a[1]**2 - a[2]**2)/(a[1]**2 + a[2]**2) * es[2,1]
        om2 = ea[0,2] + (a[2]**2 - a[0]**2)/(a[2]**2 + a[0]**2) * es[0,2]
        om3 = ea[1,0] + (a[0]**2 - a[1]**2)/(a[0]**2 + a[1]**2) * es[1,0]

        f    = es[1,2];  g    = es[0,2];  h    = es[0,1]
        xi   = ea[2,1];  eta  = ea[0,2];  zeta = ea[1,0]

        denom_F  = 2*alp0*(b0**2*be0 + c0**2*ga0)
        denom_G  = 2*bep0*(a0**2*al0 + c0**2*ga0)
        denom_H  = 2*gap0*(a0**2*al0 + b0**2*be0)

        F  = (be0*f  - c0**2*al0*(xi   - om1)) / denom_F
        G  = (ga0*g  - a0**2*be0*(eta  - om2)) / denom_G
        H  = (al0*h  - b0**2*ga0*(zeta - om3)) / denom_H
        F_ = (ga0*f  + b0**2*al0*(xi   - om1)) / denom_F
        G_ = (al0*g  + c0**2*be0*(eta  - om2)) / denom_G
        H_ = (be0*h  + a0**2*ga0*(zeta - om3)) / denom_H

        # --- RST ---
        R = -es[2,1] / I0_[0]
        S = -es[0,2] / I0_[1]
        T = -es[1,0] / I0_[2]

        # --- UVW ---
        U = 2*a[1]**2*B - 2*a[2]**2*C
        V = 2*a[2]**2*C - 2*a[0]**2*A
        W = 2*a[0]**2*A - 2*a[1]**2*B

        return dict(
            A=A, B=B, C=C,
            F=F, G=G, H=H,
            F_=F_, G_=G_, H_=H_,
            R=R, S=S, T=T,
            U=U, V=V, W=W,
            a_bold=es[0,0], b_bold=es[1,1], c_bold=es[2,2],
            f_bold=es[1,2], g_bold=es[0,2], h_bold=es[0,1],
            xii=ea[2,1], eta=ea[0,2], xi=ea[1,0],
        )

    def sigma_batch(self, X, epsilon, lam=0.0):
        """
        Stress tensor at N surface points simultaneously.

        On the ellipsoid surface lam = 0 (use_surface_mode assumption), so
        all geometric factors reduce to constants in the axes lengths and the
        coordinates xv, yv, zv.  Every operation is a pure numpy broadcast
        over shape (N,), giving a vectorised evaluation with no Python loop
        over points.

        Parameters
        ----------
        X : array-like, shape (N, 3)
            Surface point coordinates.
        epsilon : array-like, shape (3, 3)
            Far-field velocity-gradient tensor (for one basis strain at a time).
        lam : float
            Lambda value.  Should be 0.0 when called from build_transfer_matrices_batch
            (surface mode).

        Returns
        -------
        sig : np.ndarray, shape (N, 9)
            Flattened stress tensor sigma.ravel() at each point.
        """
        X    = np.asarray(X, dtype=float)          # (N, 3)
        xv   = X[:, 0]; yv = X[:, 1]; zv = X[:, 2]
        a0, b0, c0 = self.a[0], self.a[1], self.a[2]
        mu   = self.mu

        c = self._coefs_from_epsilon(epsilon)
        A=c['A']; B=c['B']; C_=c['C']
        F=c['F']; G=c['G']; H=c['H']
        F_=c['F_']; G_=c['G_']; H_=c['H_']
        R=c['R']; S=c['S']; T=c['T']
        U=c['U']; V=c['V']; W=c['W']
        a_bold=c['a_bold']; b_bold=c['b_bold']; c_bold=c['c_bold']
        f_bold=c['f_bold']; g_bold=c['g_bold']; h_bold=c['h_bold']
        xii=c['xii']; eta=c['eta']; xi=c['xi']

        # I integrals (surface mode: constants)
        I0   = np.asarray(self.I0)
        I0_  = np.asarray(self.I0_)
        al = I0[0];  be = I0[1];  ga = I0[2]
        alp= I0_[0]; bep= I0_[1]; gap= I0_[2]

        # geometric denominators at lambda=lam
        aa = a0**2 + lam; bb = b0**2 + lam; cc = c0**2 + lam
        Delta_val = np.sqrt(aa * bb * cc)

        # P² = (x²/aa² + y²/bb² + z²/cc²)⁻¹  — shape (N,)
        F_coeff = xv**2/aa**2 + yv**2/bb**2 + zv**2/cc**2
        P2  = 1.0 / F_coeff
        P_  = np.sqrt(P2)

        # dl/d* — shape (N,)
        dl_dx = 2*xv*P2 / aa
        dl_dy = 2*yv*P2 / bb
        dl_dz = 2*zv*P2 / cc

        # dP/d*
        dP_dx = (-0.5 * F_coeff**(-1.5) *
                 (2*xv/aa**2 - 2*xv**2*dl_dx/aa**3
                  - 2*yv**2*dl_dx/bb**3 - 2*zv**2*dl_dx/cc**3))
        dP_dy = (-0.5 * F_coeff**(-1.5) *
                 (2*yv/bb**2 - 2*xv**2*dl_dy/aa**3
                  - 2*yv**2*dl_dy/bb**3 - 2*zv**2*dl_dy/cc**3))
        dP_dz = (-0.5 * F_coeff**(-1.5) *
                 (2*zv/cc**2 - 2*xv**2*dl_dz/aa**3
                  - 2*yv**2*dl_dz/bb**3 - 2*zv**2*dl_dz/cc**3))

        # d(Delta⁻¹)/d*
        dDm_fac = -0.5 * (aa*bb*cc)**(-1.5) * (bb*cc + aa*cc + aa*bb)
        dDm_dx = dDm_fac * dl_dx
        dDm_dy = dDm_fac * dl_dy
        dDm_dz = dDm_fac * dl_dz

        # d(alpha,beta,gamma)/d*
        dal_dx = -dl_dx / (aa * Delta_val)
        dbe_dx = -dl_dx / (bb * Delta_val)
        dga_dx = -dl_dx / (cc * Delta_val)
        dal_dy = -dl_dy / (aa * Delta_val)
        dbe_dy = -dl_dy / (bb * Delta_val)
        dga_dy = -dl_dy / (cc * Delta_val)
        dal_dz = -dl_dz / (aa * Delta_val)
        dbe_dz = -dl_dz / (bb * Delta_val)
        dga_dz = -dl_dz / (cc * Delta_val)

        # d(alpha',beta',gamma')/d*
        dalp_dx = -dl_dx / (bb * cc * Delta_val)
        dbep_dx = -dl_dx / (aa * cc * Delta_val)
        dgap_dx = -dl_dx / (aa * bb * Delta_val)
        dalp_dy = -dl_dy / (bb * cc * Delta_val)
        dbep_dy = -dl_dy / (aa * cc * Delta_val)
        dgap_dy = -dl_dy / (aa * bb * Delta_val)
        dalp_dz = -dl_dz / (bb * cc * Delta_val)
        dbep_dz = -dl_dz / (aa * cc * Delta_val)
        dgap_dz = -dl_dz / (aa * bb * Delta_val)

        # --- Q bracket helpers (all shape (N,)) ---
        def _Q_common():
            return (((R + 2*bb*F + 2*cc*F_)*yv*zv)/(bb*cc)
                    + ((S + 2*cc*G + 2*aa*G_)*zv*xv)/(cc*aa)
                    + ((T + 2*aa*H + 2*bb*H_)*xv*yv)/(aa*bb))

        Q_common = _Q_common()

        Q   = (Q_common + (W - 2*aa*A + 2*bb*B)*yv**2/bb**2
                        - (V - 2*cc*C_ + 2*aa*A)*zv**2/cc**2)
        Qv  = (Q_common + (U - 2*bb*B + 2*cc*C_)*zv**2/cc**2
                        - (W - 2*aa*A + 2*bb*B)*xv**2/aa**2)
        Qw  = (Q_common + (V - 2*cc*C_ + 2*aa*A)*xv**2/aa**2
                        - (U - 2*bb*B + 2*cc*C_)*yv**2/bb**2)

        # --- dQ/d* helpers ---
        _FF = R + 2*bb*F + 2*cc*F_;  _GG = S + 2*cc*G + 2*aa*G_;  _HH = T + 2*aa*H + 2*bb*H_
        _WW = W - 2*aa*A + 2*bb*B;   _VV = V - 2*cc*C_ + 2*aa*A;  _UU = U - 2*bb*B + 2*cc*C_

        def dQ_common_dx():
            return ((2*F+2*F_)*dl_dx*yv*zv/(bb*cc)
                    - _FF*yv*zv*dl_dx/bb**2/cc - _FF*yv*zv*dl_dx/bb/cc**2
                    + _GG*zv/(cc*aa) + (2*G+2*G_)*dl_dx*zv*xv/(cc*aa)
                    - _GG*zv*xv*dl_dx/cc**2/aa - _GG*zv*xv*dl_dx/cc/aa**2
                    + _HH*yv/(aa*bb) + (2*H+2*H_)*dl_dx*xv*yv/(aa*bb)
                    - _HH*xv*yv*dl_dx/aa**2/bb - _HH*xv*yv*dl_dx/aa/bb**2)

        def dQ_common_dy():
            return (_FF*zv/(bb*cc) + (2*F+2*F_)*dl_dy*yv*zv/(bb*cc)
                    - _FF*yv*zv*dl_dy/bb**2/cc - _FF*yv*zv*dl_dy/bb/cc**2
                    + (2*G+2*G_)*dl_dy*zv*xv/(cc*aa)
                    - _GG*zv*xv*dl_dy/cc**2/aa - _GG*zv*xv*dl_dy/cc/aa**2
                    + _HH*xv/(aa*bb) + (2*H+2*H_)*dl_dy*xv*yv/(aa*bb)
                    - _HH*xv*yv*dl_dy/aa**2/bb - _HH*xv*yv*dl_dy/aa/bb**2)

        def dQ_common_dz():
            return (_FF*yv/(bb*cc) + (2*F+2*F_)*dl_dz*yv*zv/(bb*cc)
                    - _FF*yv*zv*dl_dz/bb**2/cc - _FF*yv*zv*dl_dz/bb/cc**2
                    + _GG*xv/(cc*aa) + (2*G+2*G_)*dl_dz*zv*xv/(cc*aa)
                    - _GG*zv*xv*dl_dz/cc**2/aa - _GG*zv*xv*dl_dz/cc/aa**2
                    + (2*H+2*H_)*dl_dz*xv*yv/(aa*bb)
                    - _HH*xv*yv*dl_dz/aa**2/bb - _HH*xv*yv*dl_dz/aa/bb**2)

        dQc_dx = dQ_common_dx(); dQc_dy = dQ_common_dy(); dQc_dz = dQ_common_dz()

        dQ_dx = (dQc_dx + (2*B-2*A)*yv**2*dl_dx/bb**2 - 2*dl_dx*_WW*yv**2/bb**3
                         + (2*C_-2*A)*zv**2*dl_dx/cc**2 + 2*dl_dx*_VV*zv**2/cc**3)
        dQ_dy = (dQc_dy + _WW*2*yv/bb**2 + (2*B-2*A)*yv**2*dl_dy/bb**2
                         - 2*dl_dy*_WW*yv**2/bb**3 + (2*C_-2*A)*zv**2*dl_dy/cc**2
                         + 2*dl_dy*_VV*zv**2/cc**3)
        dQ_dz = (dQc_dz + (2*B-2*A)*yv**2*dl_dz/bb**2 - 2*dl_dz*_WW*yv**2/bb**3
                         - 2*_VV*zv/cc**2 + (2*C_-2*A)*zv**2*dl_dz/cc**2
                         + 2*dl_dz*_VV*zv**2/cc**3)

        dQv_dx = (dQc_dx + (2*C_-2*B)*zv**2*dl_dx/cc**2 - 2*dl_dx*_UU*zv**2/cc**3
                          + (2*A-2*B)*xv**2*dl_dx/aa**2 + 2*dl_dx*_WW*xv**2/aa**3
                          - 2*_WW*xv/aa**2)
        dQv_dy = (dQc_dy + (2*C_-2*B)*zv**2*dl_dy/cc**2 - 2*dl_dy*_UU*zv**2/cc**3
                          + (2*A-2*B)*xv**2*dl_dy/aa**2 + 2*dl_dy*_WW*xv**2/aa**3)
        dQv_dz = (dQc_dz + 2*_UU*zv/cc**2 + (2*C_-2*B)*zv**2*dl_dz/cc**2
                          - 2*dl_dz*_UU*zv**2/cc**3 + (2*A-2*B)*xv**2*dl_dz/aa**2
                          + 2*dl_dz*_WW*xv**2/aa**3)

        dQw_dx = (dQc_dx + (2*A-2*C_)*xv**2*dl_dx/aa**2 - 2*dl_dx*_VV*xv**2/aa**3
                          + 2*_VV*xv/aa**2 + (2*B-2*C_)*yv**2*dl_dx/bb**2
                          + 2*dl_dx*_UU*yv**2/bb**3)
        dQw_dy = (dQc_dy + (2*A-2*C_)*xv**2*dl_dy/aa**2 - 2*dl_dy*_VV*xv**2/aa**3
                          + (2*B-2*C_)*yv**2*dl_dy/bb**2 - 2*_UU*yv/bb**2
                          + 2*dl_dy*_UU*yv**2/bb**3)
        dQw_dz = (dQc_dz + (2*A-2*C_)*xv**2*dl_dz/aa**2 - 2*dl_dz*_VV*xv**2/aa**3
                          + (2*B-2*C_)*yv**2*dl_dz/bb**2 + 2*dl_dz*_UU*yv**2/bb**3)

        # --- prefix factors px, py, pz and their derivatives ---
        px = 2*xv*P2 / (aa * Delta_val)
        py = 2*yv*P2 / (bb * Delta_val)
        pz = 2*zv*P2 / (cc * Delta_val)

        dpx_dx = (-2*P2/(aa*Delta_val) - 4*xv*P_*dP_dx/(aa*Delta_val)
                  + 2*xv*P2*dl_dx/aa**2/Delta_val - 2*xv*P2*dDm_dx/aa)
        dpx_dy = (-4*xv*P_*dP_dy/(aa*Delta_val)
                  + 2*xv*P2*dl_dy/aa**2/Delta_val - 2*xv*P2*dDm_dy/aa)
        dpx_dz = (-4*xv*P_*dP_dz/(aa*Delta_val)
                  + 2*xv*P2*dl_dz/aa**2/Delta_val - 2*xv*P2*dDm_dz/aa)

        dpy_dx = (-4*yv*P_*dP_dx/(bb*Delta_val)
                  + 2*yv*P2*dl_dx/bb**2/Delta_val - 2*yv*P2*dDm_dx/bb)
        dpy_dy = (-2*P2/(bb*Delta_val) - 4*yv*P_*dP_dy/(bb*Delta_val)
                  + 2*yv*P2*dl_dy/bb**2/Delta_val - 2*yv*P2*dDm_dy/bb)
        dpy_dz = (-4*yv*P_*dP_dz/(bb*Delta_val)
                  + 2*yv*P2*dl_dz/bb**2/Delta_val - 2*yv*P2*dDm_dz/bb)

        dpz_dx = (-4*zv*P_*dP_dx/(cc*Delta_val)
                  + 2*zv*P2*dl_dx/cc**2/Delta_val - 2*zv*P2*dDm_dx/cc)
        dpz_dy = (-4*zv*P_*dP_dy/(cc*Delta_val)
                  + 2*zv*P2*dl_dy/cc**2/Delta_val - 2*zv*P2*dDm_dy/cc)
        dpz_dz = (-2*P2/(cc*Delta_val) - 4*zv*P_*dP_dz/(cc*Delta_val)
                  + 2*zv*P2*dl_dz/cc**2/Delta_val - 2*zv*P2*dDm_dz/cc)

        # --- velocity gradients (N,) each ---
        u_x = (a_bold + ga*W - bep*V - 2*(al+be+ga)*A
               + xv*W*dgap_dx - xv*V*dbep_dx - 2*A*xv*(dal_dx+dbe_dx+dga_dx)
               + yv*T*dgap_dx - 2*yv*H*dbe_dx + 2*yv*H_*dal_dx
               + zv*S*dbep_dx - 2*zv*G_*dga_dx + 2*zv*G*dal_dx
               + dpx_dx*Q - px*dQ_dx)

        u_y = (xv*W*dgap_dy - xv*V*dbep_dy - 2*A*xv*(dal_dy+dbe_dy+dga_dy)
               + h_bold - xi + ga*T - 2*H*be + 2*al*H_
               + yv*T*dgap_dy - 2*yv*H*dbe_dy + 2*yv*H_*dal_dy
               + zv*S*dbep_dy - 2*zv*G_*dga_dy + 2*zv*G*dal_dy
               + dpx_dy*Q - px*dQ_dy)

        u_z = (xv*W*dgap_dz - xv*V*dbep_dz - 2*A*xv*(dal_dz+dbe_dz+dga_dz)
               + yv*T*dgap_dz - 2*yv*H*dbe_dz + 2*yv*H_*dal_dz
               + g_bold + eta + bep*S - 2*ga*G_ + 2*al*G
               + zv*S*dbep_dz - 2*zv*G_*dga_dz + 2*zv*G*dal_dz
               + dpx_dz*Q - px*dQ_dz)

        v_x = (h_bold + xi + ga*T + 2*H*be - 2*H_*al
               + xv*T*dgap_dx + 2*xv*H*dbe_dx - 2*xv*H_*dal_dx
               + yv*U*dalp_dx - yv*W*dgap_dx - 2*yv*B*(dal_dx+dbe_dx+dga_dx)
               + zv*R*dalp_dx - 2*zv*F*dga_dx + 2*zv*F_*dbe_dx
               + dpy_dx*Qv - py*dQv_dx)

        v_y = (xv*T*dgap_dy + 2*xv*H*dbe_dy - 2*xv*H_*dal_dy
               + b_bold + alp*U - ga*W - 2*(al+be+ga)*B
               + yv*U*dalp_dy - yv*W*dgap_dy - 2*yv*B*(dal_dy+dbe_dy+dga_dy)
               + zv*R*dalp_dy - 2*zv*F*dga_dy + 2*zv*F_*dbe_dy
               + dpy_dy*Qv - py*dQv_dy)

        v_z = (xv*T*dgap_dz + 2*xv*H*dbe_dz - 2*xv*H_*dal_dz
               + yv*U*dalp_dz - yv*W*dgap_dz - 2*yv*B*(dal_dz+dbe_dz+dga_dz)
               + f_bold - xii + alp*R - 2*ga*F + 2*be*F_
               + zv*R*dalp_dz - 2*zv*F*dga_dz + 2*zv*F_*dbe_dz
               + dpy_dz*Qv - py*dQv_dz)

        w_x = (g_bold - eta + bep*S - 2*al*G + 2*ga*G_
               + xv*S*dbep_dx - 2*xv*G*dal_dx + 2*xv*G_*dga_dx
               + yv*R*dalp_dx + 2*yv*F*dga_dx - 2*yv*F_*dbe_dx
               + zv*V*dbep_dx - zv*U*dalp_dx - 2*zv*C_*(dal_dx+dbe_dx+dga_dx)
               + dpz_dx*Qw - pz*dQw_dx)

        w_y = (xv*S*dbep_dy - 2*xv*G*dal_dy + 2*xv*G_*dga_dy
               + f_bold + xii + alp*R + 2*F*ga - 2*be*F_
               + yv*R*dalp_dy + 2*yv*F*dga_dy - 2*yv*F_*dbe_dy
               + zv*V*dbep_dy - zv*U*dalp_dy - 2*zv*C_*(dal_dy+dbe_dy+dga_dy)
               + dpz_dy*Qw - pz*dQw_dy)

        w_z = (xv*S*dbep_dz - 2*xv*G*dal_dz + 2*xv*G_*dga_dz
               + yv*R*dalp_dz + 2*yv*F*dga_dz - 2*yv*F_*dbe_dz
               + c_bold + bep*V - alp*U - 2*C_*(al+be+ga)
               + zv*V*dbep_dz - zv*U*dalp_dz - 2*zv*C_*(dal_dz+dbe_dz+dga_dz)
               + dpz_dz*Qw - pz*dQw_dz)

        # --- pressure (surface mode: lam=0, I integrals are constants) ---
        # p = 2*mu * sum_i ABC[i] * d²Omega/dxi²
        # On the surface d²Omega/dxi² = 2*I0[i] - 4xi²/(ai²·Delta²)·(P²/Delta)
        # which we can write via the existing p(x) formula evaluated at lam=0.
        # For the batch we replicate the d2Om_dx2 diagonal and p formula:
        Delta0 = float(np.sqrt(self.a[0]**2 * self.a[1]**2 * self.a[2]**2))  # lam=0
        # P² at each point (already computed above, but lam=0 here matches Delta_val=Delta0)
        # press = 2*mu*( A*(2*I0[0] - 4x²/(a0²·Delta0²)·P²/Delta0)
        #              + B*(...)  + C*(...) )
        inv_D0 = 1.0 / Delta_val   # Delta_val already computed at lam=0 via aa=a0², etc.
        d2Om_diag = np.stack([
            2*al - 4*xv**2 * P2 / (aa**2 * Delta_val),
            2*be - 4*yv**2 * P2 / (bb**2 * Delta_val),
            2*ga - 4*zv**2 * P2 / (cc**2 * Delta_val),
        ], axis=1)   # (N, 3)
        press = 2*mu * (A*d2Om_diag[:,0] + B*d2Om_diag[:,1] + C_*d2Om_diag[:,2])

        # assemble gradu (N,3,3), then sigma = -p*I + mu*(gradu + gradu^T)
        gradu = np.stack([
            np.stack([u_x, u_y, u_z], axis=1),
            np.stack([v_x, v_y, v_z], axis=1),
            np.stack([w_x, w_y, w_z], axis=1),
        ], axis=1)   # (N, 3, 3)

        # sigma_ij = -press * delta_ij + mu*(gradu_ij + gradu_ji)
        eye3 = np.eye(3)[np.newaxis]           # (1, 3, 3)
        sig3 = -press[:, np.newaxis, np.newaxis] * eye3 + mu * (gradu + gradu.transpose(0,2,1))
        return sig3.reshape(-1, 9)             # (N, 9)

    def build_transfer_matrix(self, x):
        """
        9×9 stress transfer matrix M at a single surface point x.

        vec(sigma) = M @ vec(epsilon)   (epsilon = far-field velocity gradient)

        This replaces the old `build_stress_transfer_matrix` helper in extra.py.
        It calls sigma_batch with a (1, 3) array for each of the 9 basis strains,
        avoiding any Ellipsoid re-construction or set_coefs() calls.

        Parameters
        ----------
        x : array-like, shape (3,)

        Returns
        -------
        M : np.ndarray, shape (9, 9)
        """
        X = np.asarray(x, dtype=float).reshape(1, 3)
        M = np.zeros((9, 9))
        for q in range(9):
            A_basis = np.zeros((3, 3))
            A_basis[q//3, q%3] = 1.0
            eps_basis = 0.5 * (A_basis + A_basis.T)
            M[:, q] = self.sigma_batch(X, eps_basis, lam=0.0)[0]
        return M

    def build_transfer_matrices_batch(self, X):
        """
        9×9 stress transfer matrices M at N surface points simultaneously.

        For each of the 9 basis strains the geometry loop over points is a
        single vectorised numpy call (sigma_batch), so the total cost is
        9 × O(N) array operations instead of N × 9 × O(1) Python loops.

        Parameters
        ----------
        X : array-like, shape (N, 3)
            Surface point coordinates (all assumed to lie on the ellipsoid
            surface so that lam = 0).

        Returns
        -------
        M_all : np.ndarray, shape (N, 9, 9)
            M_all[i] is the 9×9 transfer matrix at X[i].
        """
        X = np.asarray(X, dtype=float)
        N = len(X)
        M_all = np.zeros((N, 9, 9))
        for q in range(9):
            A_basis = np.zeros((3, 3))
            A_basis[q//3, q%3] = 1.0
            eps_basis = 0.5 * (A_basis + A_basis.T)
            # sigma_batch returns (N, 9); column q of M is sigma for basis q
            M_all[:, :, q] = self.sigma_batch(X, eps_basis, lam=0.0)
        return M_all

    def transfer(self, x, field='u'):
        """Compute transfer matrix for field response to far-field forcing"""
        n = self.n
        m = n**2
        T = []
        for i in range(m):
            basis = np.zeros(m)
            basis[i] = 1.0
            epsilon = self.unpack(basis)
            self.set_strain(epsilon)
            self.set_coefs()
            T.append(getattr(self, field)(x))
        return np.array(T).T