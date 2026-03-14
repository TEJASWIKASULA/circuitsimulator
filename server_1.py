"""
CircuitForge Backend Server
============================
WebSocket + HTTP server. No external deps beyond numpy.

Protocol (JSON messages):
  Client -> Server:
    {type: "circuit", components: [...], wires: [...]}
    {type: "step", dt: 1e-5}
    {type: "dc"}
    {type: "transient", duration: 0.1, dt: 1e-5}
    {type: "ping"}

  Server -> Client:
    {type: "result", voltages: {net: V}, currents: {id: A}, time: t, converged: bool}
    {type: "error", message: "..."}
    {type: "pong"}
    {type: "ready"}
"""

import json
import logging
import hashlib
import base64
import struct
import threading
import time
import traceback
import sys
import os
from http.server import HTTPServer, BaseHTTPRequestHandler
from http import HTTPStatus

# Ensure backend/ directory is on the path regardless of where script is run from
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from solver import SPICESolver
from netlist import build_nets

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger('circuitforge')

# ─────────────────────────────────────────
#  WebSocket handshake + framing (RFC 6455)
# ─────────────────────────────────────────
WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"


def ws_accept_key(key):
    combined = key.strip() + WS_GUID
    sha1 = hashlib.sha1(combined.encode()).digest()
    return base64.b64encode(sha1).decode()


def ws_decode(data):
    """Decode a WebSocket frame. Returns (opcode, payload_bytes)."""
    if len(data) < 2:
        return None, None
    b0, b1 = data[0], data[1]
    opcode = b0 & 0x0F
    masked = (b1 & 0x80) != 0
    length = b1 & 0x7F

    idx = 2
    if length == 126:
        if len(data) < idx + 2: return None, None
        length = struct.unpack('>H', data[idx:idx+2])[0]
        idx += 2
    elif length == 127:
        if len(data) < idx + 8: return None, None
        length = struct.unpack('>Q', data[idx:idx+8])[0]
        idx += 8

    mask_key = None
    if masked:
        if len(data) < idx + 4: return None, None
        mask_key = data[idx:idx+4]
        idx += 4

    if len(data) < idx + length:
        return None, None

    payload = bytearray(data[idx:idx+length])
    if masked and mask_key:
        for i in range(len(payload)):
            payload[i] ^= mask_key[i % 4]

    return opcode, bytes(payload)


def ws_encode(payload, opcode=0x01):
    """Encode a WebSocket text frame."""
    if isinstance(payload, str):
        payload = payload.encode('utf-8')
    length = len(payload)
    header = bytearray()
    header.append(0x80 | opcode)  # FIN + opcode
    if length < 126:
        header.append(length)
    elif length < 65536:
        header.append(126)
        header.extend(struct.pack('>H', length))
    else:
        header.append(127)
        header.extend(struct.pack('>Q', length))
    return bytes(header) + payload


# ─────────────────────────────────────────
#  Circuit simulation session
# ─────────────────────────────────────────
class SimSession:
    def __init__(self):
        self.solver = SPICESolver()
        self.components = []
        self.wires = []
        self.sim_time = 0.0
        self.running = False
        self.dt = 1e-5
        self._lock = threading.Lock()

    def load_circuit(self, components, wires):
        with self._lock:
            self.components = components
            self.wires = wires
            self.sim_time = 0.0
            self.solver.reset()
            # Reset cap/ind state
            self.solver._cap_vc.clear()
            self.solver._ind_il.clear()

    def step(self, dt=None):
        with self._lock:
            if dt:
                self.solver.dt = dt
            try:
                comps, wires = build_nets(self.components, self.wires)
                result = self.solver.solve_dc(comps, wires, self.sim_time)
                self.sim_time += self.solver.dt
                # Push back net assignments
                for i, c in enumerate(comps):
                    if i < len(self.components):
                        self.components[i]['_nets'] = c.get('_nets', {})
                return result
            except Exception as e:
                logger.error(f"Solver error: {e}")
                traceback.print_exc()
                return None


# ─────────────────────────────────────────
#  WebSocket connection handler
# ─────────────────────────────────────────
class WSHandler:
    def __init__(self, conn, addr):
        self.conn = conn
        self.addr = addr
        self.session = SimSession()
        self.sim_thread = None
        self.sim_running = False
        self._send_lock = threading.Lock()

    def send(self, obj):
        try:
            payload = json.dumps(obj)
            frame = ws_encode(payload)
            with self._send_lock:
                self.conn.sendall(frame)
        except Exception as e:
            logger.warning(f"Send error: {e}")

    def handle(self):
        buf = b''
        self.send({'type': 'ready', 'version': '1.0'})
        while True:
            try:
                chunk = self.conn.recv(65536)
                if not chunk:
                    break
                buf += chunk
                opcode, payload = ws_decode(buf)
                if payload is None:
                    continue
                # Consume frame from buffer
                buf = b''

                if opcode == 0x08:  # Close
                    break
                if opcode == 0x09:  # Ping
                    self.conn.sendall(ws_encode(b'', opcode=0x0A))
                    continue
                if opcode in (0x01, 0x02):  # Text/Binary
                    self._process(payload.decode('utf-8', errors='replace'))

            except ConnectionResetError:
                break
            except Exception as e:
                logger.error(f"WS error: {e}")
                break

        self.sim_running = False
        logger.info(f"WS disconnected: {self.addr}")

    def _process(self, raw):
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            self.send({'type': 'error', 'message': 'Invalid JSON'})
            return

        mtype = msg.get('type')

        if mtype == 'ping':
            self.send({'type': 'pong'})

        elif mtype == 'circuit':
            comps = msg.get('components', [])
            wires = msg.get('wires', [])
            self.session.load_circuit(comps, wires)
            self.send({'type': 'ack', 'message': f'Loaded {len(comps)} components, {len(wires)} wires'})

        elif mtype == 'dc':
            result = self.session.step()
            self._send_result(result)

        elif mtype == 'step':
            dt = msg.get('dt', 1e-5)
            result = self.session.step(dt)
            self._send_result(result)

        elif mtype == 'start_transient':
            if self.sim_running:
                self.send({'type': 'error', 'message': 'Already running'})
                return
            dt = msg.get('dt', 1e-5)
            self.session.solver.dt = dt
            self.sim_running = True
            self.sim_thread = threading.Thread(
                target=self._transient_loop,
                args=(dt,),
                daemon=True
            )
            self.sim_thread.start()

        elif mtype == 'stop_transient':
            self.sim_running = False
            self.send({'type': 'stopped'})

        elif mtype == 'update_component':
            comp_id = msg.get('id')
            props = msg.get('props', {})
            for c in self.session.components:
                if c['id'] == comp_id:
                    c['props'].update(props)
                    break
            self.send({'type': 'ack', 'message': f'Updated {comp_id}'})

        elif mtype == 'toggle_switch':
            comp_id = msg.get('id')
            for c in self.session.components:
                if c['id'] == comp_id and c['type'] == 'sw':
                    c['props']['closed'] = not c['props'].get('closed', False)
                    logger.info(f"Switch {comp_id} -> {c['props']['closed']}")
                    break
            self.send({'type': 'ack', 'message': f'Toggled {comp_id}'})

        else:
            self.send({'type': 'error', 'message': f'Unknown message type: {mtype}'})

    def _transient_loop(self, dt):
        """Continuously simulate and stream results."""
        STEPS_PER_SEND = 5
        interval = dt * STEPS_PER_SEND * 1000  # ms
        interval = max(20, min(interval, 100))  # 20-100ms send rate

        last_send = 0
        while self.sim_running:
            result = self.session.step(dt)
            now = time.time() * 1000
            if now - last_send >= interval:
                self._send_result(result)
                last_send = now
            time.sleep(dt * STEPS_PER_SEND * 0.5)

    def _send_result(self, result):
        if result is None:
            self.send({'type': 'result', 'converged': False, 'voltages': {}, 'currents': {}, 'time': self.session.sim_time})
            return

        voltages = {k: round(float(v), 6) for k, v in result.items()
                    if k not in ('_currents', '_time')}
        currents = {str(k): round(float(v), 9)
                    for k, v in result.get('_currents', {}).items()}

        # Compute power for components
        powers = {}
        for comp in self.session.components:
            cid = comp['id']
            nets = comp.get('_nets', {})
            pins = list(nets.keys())
            if len(pins) >= 2:
                va = voltages.get(nets[pins[0]], 0)
                vb = voltages.get(nets[pins[1]], 0)
                I = currents.get(str(cid), 0)
                powers[str(cid)] = round(abs((va - vb) * I), 9)

        self.send({
            'type': 'result',
            'converged': True,
            'voltages': voltages,
            'currents': currents,
            'powers': powers,
            'time': round(self.session.sim_time * 1000, 4),  # ms
            'nets': {
                comp['id']: comp.get('_nets', {})
                for comp in self.session.components
            }
        })


# ─────────────────────────────────────────
#  HTTP + WS combined server
# ─────────────────────────────────────────
class CircuitHandler(BaseHTTPRequestHandler):
    """Handles both HTTP (serve frontend) and WS upgrade."""

    # Class-level state
    frontend_html = ''
    log_message = lambda self, *a: None  # suppress access log

    def do_GET(self):
        path = self.path.split('?')[0]

        if path == '/ws':
            self._upgrade_ws()
            return

        if path in ('/', '/index.html'):
            content = self.frontend_html.encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', len(content))
            self.send_header('Cache-Control', 'no-cache')
            self.end_headers()
            self.wfile.write(content)
            return

        if path == '/api/status':
            body = json.dumps({'status': 'ok', 'version': '1.0'}).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', len(body))
            self.end_headers()
            self.wfile.write(body)
            return

        self.send_response(404)
        self.end_headers()

    def _upgrade_ws(self):
        key = self.headers.get('Sec-WebSocket-Key', '')
        if not key:
            self.send_response(400)
            self.end_headers()
            return

        accept = ws_accept_key(key)
        self.send_response(101, 'Switching Protocols')
        self.send_header('Upgrade', 'websocket')
        self.send_header('Connection', 'Upgrade')
        self.send_header('Sec-WebSocket-Accept', accept)
        self.end_headers()

        logger.info(f"WS connected: {self.client_address}")
        handler = WSHandler(self.connection, self.client_address)
        handler.handle()


def load_frontend():
    base = os.path.dirname(os.path.abspath(__file__))
    path = os.path.normpath(os.path.join(base, '..', 'frontend_html', 'index.html'))
    if not os.path.exists(path):
          raise FileNotFoundError(f'Frontend not found at: {path}')
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


def run(host='127.0.0.1', port=7777):
    CircuitHandler.frontend_html = load_frontend()
    server = HTTPServer((host, port), CircuitHandler)
    logger.info(f"CircuitForge running at http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down")


if __name__ == '__main__':
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 7777
    run(port=port)
