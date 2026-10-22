#!/usr/bin/env python3
"""Relay localhost:<port> -> TOOLATHLON_MCP_HOST:<port> inside the task
container.

Why: official preprocess/eval code hardcodes http://localhost:<port> for the
host-published backends (canvas 10001/20001, woocommerce 10003, poste
10005/1143/1587/2525) via per-task other_key.py / setup scripts -- strings the
TOOLATHLON_MCP_HOST remap never touches (its consumption points are task text,
token sessions, canvas api_client and email configs only). On the official
single-machine run the task container is --network host so localhost resolves
to the host; VTB split-MCP containers sit on a per-task bridge, so the same
code dies with "Connection refused" (canvas-art-quiz et al., preprocess).
This relay restores the localhost view. stdlib only -- the task image ships
no socat/nc, but does ship python3.

Each candidate port is probed against the upstream FIRST; ports nothing
listens on (canvas redis 6379 lives unpublished inside the canvas container,
k8s NodePorts 30123/30124 only exist while a kind cluster runs) stay unbound,
preserving the official "connection refused" semantics instead of installing
a fake listener. Bind failures on loopback are skipped, not fatal -- the
relay is best-effort by design.
"""
import os
import socket
import sys
import threading

HOST = (os.environ.get("TOOLATHLON_MCP_HOST") or "172.17.0.1").strip()
PORTS = [int(p) for p in sys.argv[1:]]


def _pump(src, dst):
    """One direction of the bidirectional relay, with half-close: EOF on
    src propagates as shutdown(WR) on dst so the peer sees it, while the
    opposite direction keeps flowing until its own EOF (IMAP-friendly)."""
    try:
        while True:
            data = src.recv(65536)
            if not data:
                break
            dst.sendall(data)
    except OSError:
        pass
    finally:
        try:
            dst.shutdown(socket.SHUT_WR)
        except OSError:
            pass


def _serve(listener, port):
    while True:
        try:
            client, _ = listener.accept()
        except OSError:
            return
        try:
            upstream = socket.create_connection((HOST, port), timeout=10)
        except OSError:
            client.close()
            continue
        threading.Thread(target=_pump, args=(client, upstream),
                         daemon=True).start()
        threading.Thread(target=_pump, args=(upstream, client),
                         daemon=True).start()


def _bind(port):
    """Listen on both loopback families; requests/urllib may resolve
    localhost to ::1 first. Returns the successfully bound sockets."""
    bound = []
    for family, addr in ((socket.AF_INET, ("127.0.0.1", port)),
                         (socket.AF_INET6, ("::1", port))):
        try:
            s = socket.socket(family, socket.SOCK_STREAM)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(addr)
            s.listen(64)
            bound.append(s)
            threading.Thread(target=_serve, args=(s, port),
                             daemon=True).start()
        except OSError:
            pass  # no v6 loopback, or something already owns the port
    return bound


def main():
    listening = []
    for port in PORTS:
        try:  # upstream reachable? don't fake-listen otherwise
            socket.create_connection((HOST, port), timeout=2).close()
        except OSError:
            print(f"skip {port}: {HOST}:{port} unreachable", flush=True)
            continue
        if _bind(port):
            listening.append(port)
        else:
            print(f"skip {port}: loopback bind failed", flush=True)
    print(f"localhost_forwarder: relaying {listening} -> {HOST}", flush=True)
    threading.Event().wait()  # park; daemon threads carry the relays


if __name__ == "__main__":
    main()
