"""HTTP/3 (QUIC) + HTTP/2 + HTTP/1.1 server via Hypercorn + aioquic.

HTTP/3 runs over QUIC, which mandates TLS — so this binds a TLS listener for
HTTP/1.1 + HTTP/2 and a UDP/QUIC listener for HTTP/3 on the same port. Clients
that support it upgrade to h3 via the advertised ``alt-svc`` header.

Run:
    python run_http3.py                # self-signed cert auto-generated
    GRAVITY_TLS_CERT=cert.pem GRAVITY_TLS_KEY=key.pem python run_http3.py

Env:
    GRAVITY_HOST   (default 0.0.0.0)
    GRAVITY_PORT   (default 6969)         # TCP (h1/h2) + UDP (h3)
    GRAVITY_TLS_CERT / GRAVITY_TLS_KEY    # PEM paths; self-signed if unset
"""

import os
import asyncio
import datetime
import ipaddress

CERT_DIR = os.path.expanduser("~/nodataishere/tls")
os.makedirs(CERT_DIR, exist_ok=True)
DEFAULT_CERT = os.path.join(CERT_DIR, "cert.pem")
DEFAULT_KEY = os.path.join(CERT_DIR, "key.pem")


def ensure_cert(cert_path: str, key_path: str) -> None:
    """Generate a self-signed cert/key if they don't exist (dev/LAN use)."""
    if os.path.exists(cert_path) and os.path.exists(key_path):
        return
    from cryptography import x509
    from cryptography.x509.oid import NameOID
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "gravity-api.local")])
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(name).issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(days=1))
        .not_valid_after(now + datetime.timedelta(days=825))
        .add_extension(x509.SubjectAlternativeName([
            x509.DNSName("localhost"),
            x509.DNSName("gravity-api.local"),
            x509.IPAddress(ipaddress.ip_address("127.0.0.1")),
        ]), critical=False)
        .sign(key, hashes.SHA256())
    )
    with open(key_path, "wb") as f:
        f.write(key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.TraditionalOpenSSL,
            serialization.NoEncryption()))
    with open(cert_path, "wb") as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))
    print(f"[http3] generated self-signed cert at {cert_path}")


def main() -> None:
    import hypercorn.asyncio
    from hypercorn.config import Config

    host = os.environ.get("GRAVITY_HOST", "0.0.0.0")
    port = int(os.environ.get("GRAVITY_PORT", "6969"))
    cert = os.environ.get("GRAVITY_TLS_CERT", DEFAULT_CERT)
    key = os.environ.get("GRAVITY_TLS_KEY", DEFAULT_KEY)
    ensure_cert(cert, key)

    cfg = Config()
    cfg.bind = [f"{host}:{port}"]            # TCP: HTTP/1.1 + HTTP/2 (ALPN)
    cfg.quic_bind = [f"{host}:{port}"]       # UDP: HTTP/3 (QUIC)
    cfg.certfile = cert
    cfg.keyfile = key
    cfg.alpn_protocols = ["h3", "h2", "http/1.1"]
    cfg.workers = 1                          # shared in-proc stores → single proc
    cfg.accesslog = "-"

    from main import app
    print(f"[http3] serving h3/h2/h1 on https://{host}:{port}  (QUIC udp/{port})")
    asyncio.run(hypercorn.asyncio.serve(app, cfg))


if __name__ == "__main__":
    main()
