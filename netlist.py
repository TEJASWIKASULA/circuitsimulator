"""
NetList Builder
===============
Converts a circuit graph (components + wires) into net assignments
using Union-Find for connectivity resolution.
"""


class UnionFind:
    def __init__(self):
        self._parent = {}

    def find(self, x):
        if x not in self._parent:
            self._parent[x] = x
        if self._parent[x] != x:
            self._parent[x] = self.find(self._parent[x])
        return self._parent[x]

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            # GND always wins as root
            if rb == 'GND':
                self._parent[ra] = rb
            else:
                self._parent[rb] = ra

    def same(self, a, b):
        return self.find(a) == self.find(b)


# Component pin definitions
COMP_PINS = {
    'vdc':    ['A', 'B'],
    'vac':    ['A', 'B'],
    'gnd':    ['A'],
    'res':    ['A', 'B'],
    'cap':    ['A', 'B'],
    'ind':    ['A', 'B'],
    'diode':  ['A', 'B'],
    'zener':  ['A', 'B'],
    'led':    ['A', 'B'],
    'npn':    ['B', 'C', 'E'],
    'pnp':    ['B', 'C', 'E'],
    'nmos':   ['G', 'D', 'S'],
    'bulb':   ['A', 'B'],
    'vmeter': ['A', 'B'],
    'amm':    ['A', 'B'],
    'sw':     ['A', 'B'],
}

# Pin world positions (relative to component origin, before rotation)
COMP_PIN_POS = {
    'vdc':    {'A': (70, 25), 'B': (0, 25)},
    'vac':    {'A': (70, 25), 'B': (0, 25)},
    'gnd':    {'A': (20, 0)},
    'res':    {'A': (0, 15),  'B': (70, 15)},
    'cap':    {'A': (0, 20),  'B': (60, 20)},
    'ind':    {'A': (0, 15),  'B': (70, 15)},
    'diode':  {'A': (0, 20),  'B': (60, 20)},
    'zener':  {'A': (0, 20),  'B': (60, 20)},
    'led':    {'A': (0, 20),  'B': (60, 20)},
    'npn':    {'B': (0, 35),  'C': (50, 5),  'E': (50, 65)},
    'pnp':    {'B': (0, 35),  'C': (50, 5),  'E': (50, 65)},
    'nmos':   {'G': (0, 35),  'D': (60, 5),  'S': (60, 65)},
    'bulb':   {'A': (0, 30),  'B': (60, 30)},
    'vmeter': {'A': (0, 30),  'B': (60, 30)},
    'amm':    {'A': (0, 30),  'B': (60, 30)},
    'sw':     {'A': (0, 15),  'B': (60, 15)},
}

COMP_SIZE = {
    'vdc':    (70, 50),  'vac':    (70, 50),  'gnd':    (40, 40),
    'res':    (70, 30),  'cap':    (60, 40),  'ind':    (70, 30),
    'diode':  (60, 40),  'zener':  (60, 40),  'led':    (60, 40),
    'npn':    (60, 70),  'pnp':    (60, 70),  'nmos':   (60, 70),
    'bulb':   (60, 60),  'vmeter': (60, 60),  'amm':    (60, 60),
    'sw':     (60, 30),
}


def rotate_point(x, y, cx, cy, angle_deg):
    """Rotate (x,y) around (cx,cy) by angle_deg."""
    import math
    angle = math.radians(angle_deg)
    dx, dy = x - cx, y - cy
    nx = dx * math.cos(angle) - dy * math.sin(angle)
    ny = dx * math.sin(angle) + dy * math.cos(angle)
    return nx + cx, ny + cy


def pin_world_pos(comp, pin_id):
    """Get world position of a component pin, accounting for rotation."""
    ctype = comp['type']
    local_pins = COMP_PIN_POS.get(ctype, {})
    size = COMP_SIZE.get(ctype, (60, 40))

    if pin_id not in local_pins:
        return (comp['x'], comp['y'])

    lx, ly = local_pins[pin_id]
    cx = comp['x'] + size[0] / 2
    cy = comp['y'] + size[1] / 2
    rot = comp.get('rot', 0)

    wx, wy = comp['x'] + lx, comp['y'] + ly
    if rot:
        wx, wy = rotate_point(wx, wy, cx, cy, rot)

    return (round(wx, 1), round(wy, 1))


def snap(v, grid=20):
    return round(v / grid) * grid


def build_nets(components, wires):
    """
    Assign net names to all component pins.
    Returns components with '_nets' dict populated.
    Modifies wires with 'net' field.
    """
    uf = UnionFind()

    # Keys: "compid_pinid" or "wire_Wid_end"
    def comp_key(comp_id, pin_id):
        return f"C{comp_id}_{pin_id}"

    def wire_key(wire_id, end):
        return f"W{wire_id}_{end}"

    # Register all pin keys
    for comp in components:
        ctype = comp['type']
        for pin_id in COMP_PINS.get(ctype, []):
            uf.find(comp_key(comp['id'], pin_id))

        # GND component connects its pin to ground
        if ctype == 'gnd':
            uf.union(comp_key(comp['id'], 'A'), 'GND')

    # Register wire endpoints and connect each end to itself
    for wire in wires:
        uf.find(wire_key(wire['id'], 'A'))
        uf.find(wire_key(wire['id'], 'B'))
        # Wire internally connects A to B
        uf.union(wire_key(wire['id'], 'A'), wire_key(wire['id'], 'B'))

    # Connect pins at same world position
    # Build position -> key mapping
    pos_map = {}  # (x,y) -> list of keys

    for comp in components:
        ctype = comp['type']
        for pin_id in COMP_PINS.get(ctype, []):
            pos = pin_world_pos(comp, pin_id)
            px, py = round(pos[0]), round(pos[1])
            key = comp_key(comp['id'], pin_id)
            pt = (px, py)
            if pt not in pos_map:
                pos_map[pt] = []
            pos_map[pt].append(key)

    for wire in wires:
        for end, (wx, wy) in [('A', (wire['x1'], wire['y1'])),
                               ('B', (wire['x2'], wire['y2']))]:
            pt = (round(wx), round(wy))
            key = wire_key(wire['id'], end)
            if pt not in pos_map:
                pos_map[pt] = []
            pos_map[pt].append(key)

    # Union everything at the same position
    for pt, keys in pos_map.items():
        for i in range(1, len(keys)):
            uf.union(keys[0], keys[i])

    # Assign net names
    root_to_net = {}
    net_counter = [0]

    def get_net(key):
        root = uf.find(key)
        if root == 'GND' or uf.find('GND') == root:
            return 'GND'
        if root not in root_to_net:
            net_counter[0] += 1
            root_to_net[root] = f'N{net_counter[0]}'
        return root_to_net[root]

    # Assign to components
    result_comps = []
    for comp in components:
        ctype = comp['type']
        nets = {}
        for pin_id in COMP_PINS.get(ctype, []):
            key = comp_key(comp['id'], pin_id)
            nets[pin_id] = get_net(key)
        comp = dict(comp)
        comp['_nets'] = nets
        result_comps.append(comp)

    # Assign to wires
    result_wires = []
    for wire in wires:
        wire = dict(wire)
        net_a = get_net(wire_key(wire['id'], 'A'))
        net_b = get_net(wire_key(wire['id'], 'B'))
        wire['netA'] = net_a
        wire['netB'] = net_b
        wire['net'] = net_a if net_a != 'GND' else net_b
        result_wires.append(wire)

    return result_comps, result_wires


def get_all_nets(components, wires):
    """Return set of all net names in circuit."""
    nets = set()
    for comp in components:
        for net in comp.get('_nets', {}).values():
            nets.add(net)
    for wire in wires:
        if 'netA' in wire:
            nets.add(wire['netA'])
            nets.add(wire['netB'])
    return nets
