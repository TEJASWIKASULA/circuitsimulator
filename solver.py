"""
CircuitForge SPICE Engine
=========================
Real Modified Nodal Analysis (MNA) solver using numpy.
Implements:
  - DC operating point analysis
  - Transient analysis (backward Euler)
  - Newton-Raphson for non-linear devices
  - Diode / LED / Zener (Shockley model)
  - BJT NPN/PNP (Ebers-Moll linearized)
  - MOSFET NMOS (square-law)
  - Capacitor / Inductor companion models
"""

import numpy as np
import math
import logging

logger = logging.getLogger(__name__)

VT = 0.02585      # Thermal voltage at 300K
MAX_ITER = 50     # Newton-Raphson max iterations
NR_TOL = 1e-6     # Convergence tolerance
MAX_EXP = 40.0    # Clamp exp argument


def safe_exp(x):
    return math.exp(min(x, MAX_EXP))


class Node:
    GND = 0


class SPICESolver:
    """
    Full MNA SPICE solver.

    Matrix equation: [G | B] [v]   [i]
                     [C | D] [j] = [e]

    G = conductance sub-matrix (n×n)
    B = voltage source columns  (n×k)
    C = voltage source rows     (k×n)
    D = zero sub-matrix         (k×k)
    v = node voltages
    j = branch currents (V sources)
    i = current injections
    e = voltage source values
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.node_map = {}      # name -> index (0=GND excluded)
        self.n_nodes = 0        # count excluding GND
        self.vsrc_list = []     # (comp, branch_index)
        self.solution = None    # last solution vector
        self.time = 0.0
        self.dt = 1e-5          # 10µs default timestep
        # State for reactive elements
        self._cap_vc = {}       # cap_id -> voltage
        self._ind_il = {}       # ind_id -> current
        # NR state
        self._diode_vd = {}     # diode_id -> Vd
        self._bjt_vbe = {}
        self._bjt_vbc = {}
        self._mos_vgs = {}
        self._mos_vds = {}

    def node_index(self, name):
        """Return 1-based node index. GND variants return 0."""
        if name in ('GND', '0', None, ''):
            return 0
        if name not in self.node_map:
            self.n_nodes += 1
            self.node_map[name] = self.n_nodes
        return self.node_map[name]

    def solve_dc(self, components, nets, sim_time=0.0):
        """
        Solve operating point.
        Returns dict: {net_name: voltage, '_currents': {comp_id: current}}
        """
        self.node_map = {}
        self.n_nodes = 0
        self.vsrc_list = []
        self.time = sim_time

        # First pass: register all nodes
        for comp in components:
            ctype = comp['type']
            props = comp.get('props', {})
            for pin_id, net in comp.get('_nets', {}).items():
                self.node_index(net)

        # Collect voltage sources (need extra MNA rows)
        vsrc_types = {'vdc', 'vac', 'vmeter', 'amm'}
        for comp in components:
            if comp['type'] in vsrc_types:
                self.vsrc_list.append(comp)

        N = self.n_nodes
        K = len(self.vsrc_list)
        size = N + K

        if size == 0:
            return {}

        # Newton-Raphson loop
        v_prev = np.zeros(N)

        for nr_iter in range(MAX_ITER):
            G = np.zeros((size, size))
            b = np.zeros(size)

            # Stamp all elements
            for comp in components:
                self._stamp(G, b, comp, v_prev, sim_time)

            # Stamp voltage sources
            for k, vsrc in enumerate(self.vsrc_list):
                self._stamp_vsource(G, b, vsrc, k, N, sim_time)

            # Solve
            try:
                x = np.linalg.solve(G, b)
            except np.linalg.LinAlgError:
                # Singular - try pseudoinverse
                try:
                    x = np.linalg.lstsq(G, b, rcond=None)[0]
                except Exception:
                    logger.warning("Singular matrix, returning zeros")
                    return {}

            v_new = x[:N]

            # Check convergence
            err = np.max(np.abs(v_new - v_prev)) if N > 0 else 0
            v_prev = v_new.copy()

            if err < NR_TOL:
                break

        # Update reactive element states
        self.solution = x
        results = {'GND': 0.0, '_time': sim_time}

        for name, idx in self.node_map.items():
            results[name] = float(x[idx - 1])

        # Compute branch currents for each component
        currents = {}
        for comp in components:
            cid = comp['id']
            nets_map = comp.get('_nets', {})
            ctype = comp['type']
            pins = list(nets_map.keys())

            if len(pins) >= 2:
                na = nets_map.get(pins[0], 'GND')
                nb = nets_map.get(pins[1], 'GND')
                va = results.get(na, 0.0)
                vb = results.get(nb, 0.0)
                vd = va - vb

                if ctype == 'res':
                    R = max(comp['props'].get('R', 1000), 1e-6)
                    currents[cid] = vd / R
                elif ctype in ('diode', 'led', 'zener'):
                    Is = 1e-14
                    currents[cid] = Is * (safe_exp(vd / VT) - 1)
                elif ctype == 'cap':
                    vc_prev = self._cap_vc.get(cid, 0.0)
                    C = comp['props'].get('C', 1e-4)
                    currents[cid] = C * (vd - vc_prev) / self.dt
                    self._cap_vc[cid] = vd
                elif ctype == 'ind':
                    self._ind_il[cid] = self._ind_il.get(cid, 0.0) + \
                                        vd * self.dt / max(comp['props'].get('L', 1e-3), 1e-9)
                    currents[cid] = self._ind_il.get(cid, 0.0)
                elif ctype == 'sw':
                    if comp['props'].get('closed', False):
                        Ron = 0.001
                        currents[cid] = vd / Ron
                    else:
                        currents[cid] = 0.0
                else:
                    currents[cid] = 0.0

            # Update NR state
            if ctype in ('diode', 'led', 'zener'):
                na = nets_map.get('A', 'GND')
                nb = nets_map.get('B', 'GND')
                self._diode_vd[cid] = results.get(na, 0) - results.get(nb, 0)
            elif ctype in ('npn', 'pnp'):
                nb_ = nets_map.get('B', 'GND')
                nc_ = nets_map.get('C', 'GND')
                ne_ = nets_map.get('E', 'GND')
                self._bjt_vbe[cid] = results.get(nb_, 0) - results.get(ne_, 0)
                self._bjt_vbc[cid] = results.get(nb_, 0) - results.get(nc_, 0)
            elif ctype == 'nmos':
                ng = nets_map.get('G', 'GND')
                nd = nets_map.get('D', 'GND')
                ns = nets_map.get('S', 'GND')
                self._mos_vgs[cid] = results.get(ng, 0) - results.get(ns, 0)
                self._mos_vds[cid] = results.get(nd, 0) - results.get(ns, 0)

        results['_currents'] = currents
        return results

    def _net_v(self, v_vec, net):
        """Get voltage of net from solution vector."""
        if net in ('GND', '0', None, ''):
            return 0.0
        idx = self.node_map.get(net)
        if idx is None:
            return 0.0
        return float(v_vec[idx - 1]) if idx - 1 < len(v_vec) else 0.0

    def _stamp(self, G, b, comp, v_vec, sim_time):
        """Stamp component into MNA matrix."""
        ctype = comp['type']
        props = comp.get('props', {})
        nets_map = comp.get('_nets', {})
        cid = comp['id']
        N = self.n_nodes

        def ni(net):
            """Node index (1-based), or 0 for GND."""
            if net in ('GND', '0', None, ''):
                return 0
            return self.node_map.get(net, 0)

        def nv(net):
            return self._net_v(v_vec, net)

        def stamp_g(na, nb, g):
            """Stamp conductance between nodes na and nb."""
            p, q = ni(na), ni(nb)
            if p > 0: G[p-1][p-1] += g
            if q > 0: G[q-1][q-1] += g
            if p > 0 and q > 0:
                G[p-1][q-1] -= g
                G[q-1][p-1] -= g

        def stamp_i(na, nb, I):
            """Stamp current source from nb to na."""
            p, q = ni(na), ni(nb)
            if p > 0: b[p-1] += I
            if q > 0: b[q-1] -= I

        if ctype == 'res':
            R = max(props.get('R', 1000), 1e-9)
            na, nb = nets_map.get('A', 'GND'), nets_map.get('B', 'GND')
            stamp_g(na, nb, 1.0 / R)

        elif ctype == 'cap':
            C = max(props.get('C', 1e-4), 1e-15)
            na, nb = nets_map.get('A', 'GND'), nets_map.get('B', 'GND')
            g_eq = C / self.dt
            vc = self._cap_vc.get(cid, 0.0)
            I_eq = g_eq * vc
            stamp_g(na, nb, g_eq)
            stamp_i(na, nb, -I_eq)

        elif ctype == 'ind':
            L = max(props.get('L', 1e-3), 1e-12)
            na, nb = nets_map.get('A', 'GND'), nets_map.get('B', 'GND')
            g_eq = self.dt / L
            il = self._ind_il.get(cid, 0.0)
            stamp_g(na, nb, g_eq)
            stamp_i(na, nb, il)

        elif ctype in ('diode', 'led'):
            na, nb = nets_map.get('A', 'GND'), nets_map.get('B', 'GND')
            Is = 1e-14 if ctype == 'diode' else 1e-15
            Vd = self._diode_vd.get(cid, 0.0)
            Vd = max(min(Vd, 1.5), -5.0)
            exp_vd = safe_exp(Vd / VT)
            Id = Is * (exp_vd - 1)
            gd = Is * exp_vd / VT
            I_eq = Id - gd * Vd
            stamp_g(na, nb, gd)
            stamp_i(na, nb, -I_eq)

        elif ctype == 'zener':
            na, nb = nets_map.get('A', 'GND'), nets_map.get('B', 'GND')
            Is = 1e-14
            Vz = props.get('Vz', 5.1)
            Vd = self._diode_vd.get(cid, 0.0)
            Vd = max(min(Vd, 1.5), -(Vz + 1))
            # Forward diode
            exp_f = safe_exp(Vd / VT)
            Id_f = Is * (exp_f - 1)
            gd_f = Is * exp_f / VT
            Ieq_f = Id_f - gd_f * Vd
            stamp_g(na, nb, gd_f)
            stamp_i(na, nb, -Ieq_f)
            # Reverse zener breakdown (simplified)
            Vr = -(Vd + Vz)
            if Vr > 0:
                exp_r = safe_exp(Vr / VT)
                Id_r = Is * (exp_r - 1)
                gd_r = Is * exp_r / VT
                Ieq_r = Id_r - gd_r * Vr
                stamp_g(nb, na, gd_r)
                stamp_i(nb, na, -Ieq_r)

        elif ctype in ('npn', 'pnp'):
            sign = 1 if ctype == 'npn' else -1
            Bf = max(props.get('Hfe', 100), 1)
            Is = 1e-14
            nb_ = nets_map.get('B', 'GND')
            nc_ = nets_map.get('C', 'GND')
            ne_ = nets_map.get('E', 'GND')
            Vbe = sign * self._bjt_vbe.get(cid, 0.6)
            Vbc = sign * self._bjt_vbc.get(cid, 0.0)
            Vbe = max(min(Vbe, 1.2), -5.0)
            Vbc = max(min(Vbc, 1.2), -5.0)

            exp_be = safe_exp(Vbe / VT)
            exp_bc = safe_exp(Vbc / VT)

            # Base-emitter conductance
            gbe = Is / Bf / VT * exp_be
            Ibe = Is / Bf * (exp_be - 1)
            Ieq_be = Ibe - gbe * Vbe

            # Collector current (transconductance)
            gm = Is / VT * exp_be
            Ic = Is * (exp_be - exp_bc)
            Icq = Ic - gm * Vbe

            p_b, p_c, p_e = ni(nb_), ni(nc_), ni(ne_)

            # Stamp gbe between B and E
            if p_b > 0: G[p_b-1][p_b-1] += gbe
            if p_e > 0: G[p_e-1][p_e-1] += gbe
            if p_b > 0 and p_e > 0:
                G[p_b-1][p_e-1] -= gbe
                G[p_e-1][p_b-1] -= gbe
            if p_b > 0: b[p_b-1] -= sign * Ieq_be
            if p_e > 0: b[p_e-1] += sign * Ieq_be

            # Stamp gm*Vbe as VCCS from C to E
            if p_c > 0 and p_b > 0: G[p_c-1][p_b-1] += sign * gm
            if p_c > 0 and p_e > 0: G[p_c-1][p_e-1] -= sign * gm
            if p_e > 0 and p_b > 0: G[p_e-1][p_b-1] -= sign * gm
            if p_e > 0 and p_e > 0: G[p_e-1][p_e-1] += sign * gm
            if p_c > 0: b[p_c-1] -= sign * Icq
            if p_e > 0: b[p_e-1] += sign * Icq

        elif ctype == 'nmos':
            Vth = props.get('Vth', 2.0)
            Kn = props.get('K', 0.01)
            ng = nets_map.get('G', 'GND')
            nd = nets_map.get('D', 'GND')
            ns = nets_map.get('S', 'GND')
            Vgs = self._mos_vgs.get(cid, 0.0)
            Vds = self._mos_vds.get(cid, 0.0)

            Vov = Vgs - Vth
            gm, gds, Id = 0.0, 1e-9, 0.0

            if Vov > 0:
                if Vds >= Vov:  # Saturation
                    Id = 0.5 * Kn * Vov ** 2
                    gm = Kn * Vov
                    gds = 1e-6
                else:           # Linear/Triode
                    Id = Kn * (Vov * Vds - 0.5 * Vds ** 2)
                    gm = Kn * Vds
                    gds = Kn * (Vov - Vds)

            Icq = Id - gm * Vgs - gds * Vds

            p_g, p_d, p_s = ni(ng), ni(nd), ni(ns)

            # gds between D and S
            if p_d > 0: G[p_d-1][p_d-1] += gds
            if p_s > 0: G[p_s-1][p_s-1] += gds
            if p_d > 0 and p_s > 0:
                G[p_d-1][p_s-1] -= gds
                G[p_s-1][p_d-1] -= gds

            # gm VCCS
            if p_d > 0 and p_g > 0: G[p_d-1][p_g-1] += gm
            if p_d > 0 and p_s > 0: G[p_d-1][p_s-1] -= gm
            if p_s > 0 and p_g > 0: G[p_s-1][p_g-1] -= gm
            if p_s > 0 and p_s > 0: G[p_s-1][p_s-1] += gm

            if p_d > 0: b[p_d-1] -= Icq
            if p_s > 0: b[p_s-1] += Icq

        elif ctype == 'sw':
            if props.get('closed', False):
                na, nb = nets_map.get('A', 'GND'), nets_map.get('B', 'GND')
                stamp_g(na, nb, 1.0 / 0.001)
            else:
                na, nb = nets_map.get('A', 'GND'), nets_map.get('B', 'GND')
                stamp_g(na, nb, 1e-9)  # High resistance open switch

        elif ctype == 'bulb':
            W = props.get('W', 5)
            V_nominal = 5.0
            R = V_nominal ** 2 / max(W, 0.001)
            na, nb = nets_map.get('A', 'GND'), nets_map.get('B', 'GND')
            stamp_g(na, nb, 1.0 / R)

        elif ctype == 'gnd':
            # Pure ground node - no stamp needed, handled by node_index
            pass

    def _stamp_vsource(self, G, b, vsrc, k, N, sim_time):
        """Stamp voltage source using MNA extra rows/cols."""
        row = N + k
        ctype = vsrc['type']
        props = vsrc.get('props', {})
        nets_map = vsrc.get('_nets', {})

        # Voltage value
        if ctype == 'vmeter' or ctype == 'amm':
            V = 0.0  # ideal meter = 0V source
        elif ctype == 'vac':
            Vamp = props.get('V', 10.0)
            freq = props.get('f', 60.0)
            V = Vamp * math.sin(2 * math.pi * freq * sim_time)
        else:  # vdc
            V = props.get('V', 5.0)

        # Pin A = positive, Pin B = negative
        na = nets_map.get('A', 'GND')
        nb = nets_map.get('B', 'GND')

        def ni(net):
            if net in ('GND', '0', None, ''):
                return 0
            return self.node_map.get(net, 0)

        p, q = ni(na), ni(nb)
        size = G.shape[0]

        if row < size:
            if p > 0 and p - 1 < size:
                G[row][p-1] = 1.0
                G[p-1][row] = 1.0
            if q > 0 and q - 1 < size:
                G[row][q-1] = -1.0
                G[q-1][row] = -1.0
            b[row] = V
