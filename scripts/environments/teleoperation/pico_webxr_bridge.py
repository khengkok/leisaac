# /// script
# requires-python = ">=3.10"
# dependencies = ["websockets", "pyzmq"]
# ///
"""Bridge a Pico 4 Ultra controller to ZMQ PUB using WebXR -- no SteamVR/PICO
Connect/Windows PC needed.

Run this on whatever machine you'd otherwise run so101_joint_state_server.py
on (e.g. your Mac, over Wi-Fi with the headset). It serves a small WebXR page
over HTTPS that you open directly in the Pico's own browser; the page reads
controller pose + trigger/grip via the WebXR Device API in JS and streams it
back over a WebSocket, which this script re-publishes over ZMQ in the same
10-float wire format leisaac's SO101PicoController expects -- so the EC2 side
(teleop_device=pico) needs no changes regardless of which bridge you use.

Prerequisites:
    - The Pico 4 Ultra and this machine on the same Wi-Fi network.
    - pip install websockets pyzmq
    - openssl CLI available (ships with macOS/Linux) to generate a one-time
      self-signed TLS cert -- WebXR requires a secure (https/wss) context.

Usage:
    python pico_webxr_bridge.py --hand right
    # Then, in the Pico's browser, open the printed https://<lan-ip>:8443/
    # URL, accept the self-signed certificate warning, tap "Enter VR", and
    # hold GRIP to drive the arm.
"""

import argparse
import asyncio
import http.server
import json
import os
import socket
import ssl
import struct
import subprocess
import threading

import websockets
import zmq

CACHE_DIR = os.path.join(os.path.dirname(__file__), ".cache", "pico_webxr")
CERT_PATH = os.path.join(CACHE_DIR, "cert.pem")
KEY_PATH = os.path.join(CACHE_DIR, "key.pem")

_CLIENT_PAGE = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LeIsaac Pico Bridge</title>
<style>
  body { font-family: sans-serif; background: #111; color: #eee; text-align: center; padding-top: 40px; }
  button { font-size: 1.5em; padding: 0.5em 1.5em; }
  #status { margin-top: 20px; font-size: 1.1em; white-space: pre-wrap; }
</style>
</head>
<body>
  <h2>LeIsaac Pico Controller Bridge</h2>
  <p>Hand: __HAND__</p>
  <button id="startBtn">Enter VR</button>
  <div id="status">not started</div>
<script>
const HAND = "__HAND__";
const WS_URL = "wss://" + location.hostname + ":__WS_PORT__/";
const statusEl = document.getElementById("status");
let ws = null;

function setStatus(msg) { statusEl.textContent = msg; }

function connectWS() {
  ws = new WebSocket(WS_URL);
  ws.onopen = () => setStatus("ws: connected");
  ws.onclose = () => { setStatus("ws: disconnected, retrying..."); setTimeout(connectWS, 1000); };
  ws.onerror = () => {};
}
connectWS();

document.getElementById("startBtn").addEventListener("click", async () => {
  if (!navigator.xr) { setStatus("WebXR not available in this browser"); return; }
  const supported = await navigator.xr.isSessionSupported("immersive-vr");
  if (!supported) { setStatus("immersive-vr not supported"); return; }

  const session = await navigator.xr.requestSession("immersive-vr", { requiredFeatures: ["local-floor"] });
  let refSpace;
  try {
    refSpace = await session.requestReferenceSpace("local-floor");
  } catch (e) {
    refSpace = await session.requestReferenceSpace("local");
  }

  const canvas = document.createElement("canvas");
  const gl = canvas.getContext("webgl", { xrCompatible: true });
  session.updateRenderState({ baseLayer: new XRWebGLLayer(session, gl) });
  session.addEventListener("end", () => setStatus("xr session ended"));

  function onFrame(time, frame) {
    session.requestAnimationFrame(onFrame);

    let tracked = false;
    for (const source of session.inputSources) {
      if (source.handedness !== HAND) continue;
      const space = source.gripSpace || source.targetRaySpace;
      if (!space) continue;
      const pose = frame.getPose(space, refSpace);
      if (!pose) continue;

      const p = pose.transform.position;
      const o = pose.transform.orientation;
      const gp = source.gamepad;
      const trigger = (gp && gp.buttons[0]) ? gp.buttons[0].value : 0.0;
      const grip = (gp && gp.buttons[1]) ? gp.buttons[1].value : 0.0;

      const msg = {
        px: p.x, py: p.y, pz: p.z,
        qx: o.x, qy: o.y, qz: o.z, qw: o.w,
        trigger: trigger, grip: grip, valid: 1.0,
      };
      if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(msg));
      tracked = true;
    }
    if (!tracked && ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({px:0,py:0,pz:0,qx:0,qy:0,qz:0,qw:1,trigger:0,grip:0,valid:0.0}));
    }
    setStatus("xr: active, hand=" + HAND + (tracked ? " (tracked)" : " (NOT tracked)"));
  }
  session.requestAnimationFrame(onFrame);
  setStatus("xr: session started");
});
</script>
</body>
</html>
"""


def ensure_self_signed_cert():
    if os.path.exists(CERT_PATH) and os.path.exists(KEY_PATH):
        return
    os.makedirs(CACHE_DIR, exist_ok=True)
    print("Generating a one-time self-signed TLS cert (WebXR requires a secure context)...")
    subprocess.run(
        [
            "openssl",
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-nodes",
            "-keyout",
            KEY_PATH,
            "-out",
            CERT_PATH,
            "-days",
            "3650",
            "-subj",
            "/CN=leisaac-pico-bridge",
        ],
        check=True,
    )


def local_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


def make_https_server(page: str, http_port: int, ssl_context: ssl.SSLContext) -> http.server.ThreadingHTTPServer:
    page_bytes = page.encode("utf-8")

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(page_bytes)))
            self.end_headers()
            self.wfile.write(page_bytes)

        def log_message(self, format, *args):
            pass

    server = http.server.ThreadingHTTPServer(("0.0.0.0", http_port), Handler)
    server.socket = ssl_context.wrap_socket(server.socket, server_side=True)
    return server


async def run_ws_server(ws_port: int, ssl_context: ssl.SSLContext, pub: zmq.Socket):
    async def handler(websocket):
        async for raw in websocket:
            try:
                data = json.loads(raw)
                msg = struct.pack(
                    "<10f",
                    data["px"],
                    data["py"],
                    data["pz"],
                    data["qx"],
                    data["qy"],
                    data["qz"],
                    data["qw"],
                    data["trigger"],
                    data["grip"],
                    data["valid"],
                )
                pub.send(msg, zmq.NOBLOCK)
            except (json.JSONDecodeError, KeyError):
                continue

    async with websockets.serve(handler, "0.0.0.0", ws_port, ssl=ssl_context):
        await asyncio.Future()  # run forever


def main():
    parser = argparse.ArgumentParser(description="Pico 4 Ultra WebXR-to-ZMQ bridge (no SteamVR needed)")
    parser.add_argument("--hand", choices=["left", "right"], default="right", help="Which controller to publish")
    parser.add_argument("--bind", default="tcp://0.0.0.0:5557", help="ZMQ PUB bind address")
    parser.add_argument("--http_port", type=int, default=8443, help="HTTPS port serving the WebXR page")
    parser.add_argument("--ws_port", type=int, default=8444, help="WSS port receiving pose data from the page")
    args = parser.parse_args()

    ensure_self_signed_cert()
    ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ssl_context.load_cert_chain(CERT_PATH, KEY_PATH)

    ctx = zmq.Context()
    pub = ctx.socket(zmq.PUB)
    pub.setsockopt(zmq.CONFLATE, 1)
    pub.bind(args.bind)

    page = _CLIENT_PAGE.replace("__HAND__", args.hand).replace("__WS_PORT__", str(args.ws_port))
    https_server = make_https_server(page, args.http_port, ssl_context)
    threading.Thread(target=https_server.serve_forever, daemon=True).start()

    ip = local_ip()
    print(f"Publishing controller data on {args.bind}")
    print(f"On the Pico headset, open: https://{ip}:{args.http_port}/")
    print("Accept the self-signed certificate warning, tap 'Enter VR', then hold GRIP to drive the arm.")

    try:
        asyncio.run(run_ws_server(args.ws_port, ssl_context, pub))
    except KeyboardInterrupt:
        print("\nDone.")
    finally:
        https_server.shutdown()
        pub.close()
        ctx.term()


if __name__ == "__main__":
    main()
