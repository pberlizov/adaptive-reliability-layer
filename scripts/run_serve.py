#!/usr/bin/env python3
"""Start the ARL production HTTP sidecar."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve Adaptive Reliability Layer over HTTP.")
    parser.add_argument("--config", default="configs/serving_pilot_fraud_torch.yaml")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument(
        "--model-bundle",
        default=None,
        help="Override serving.model_bundle (e.g. paysim_fraud_torch, paysim_fraud)",
    )
    parser.add_argument(
        "--force-shadow",
        action="store_true",
        help="Set ARL_FORCE_SHADOW=1 for zero-mutation serving",
    )
    parser.add_argument("--ssl-certfile", default=None, help="TLS certificate for mTLS/HTTPS")
    parser.add_argument("--ssl-keyfile", default=None, help="TLS private key")
    parser.add_argument(
        "--ssl-ca-certs",
        default=None,
        help="CA bundle for client certificate verification (mTLS)",
    )
    args = parser.parse_args()

    if args.force_shadow:
        os.environ["ARL_FORCE_SHADOW"] = "1"

    try:
        import uvicorn
    except ImportError as exc:
        raise SystemExit("Install serving extras: pip install -e '.[serving,prometheus]'") from exc

    from adaptive_reliability_layer.serving.app import create_app
    from adaptive_reliability_layer.serving.config import ServingConfig, load_serving_config_from_yaml
    from adaptive_reliability_layer.runtime.config import RuntimeConfig

    raw, serving = load_serving_config_from_yaml(args.config)
    if args.model_bundle:
        serving = ServingConfig(
            **{
                **serving.__dict__,
                "model_bundle": args.model_bundle,
            }
        )
    runtime_config = RuntimeConfig.from_mapping(raw)
    from adaptive_reliability_layer.serving.loader import build_layer_for_serving

    layer = build_layer_for_serving(runtime_config, serving)
    app = create_app(config_path=args.config, layer=layer, serving=serving, runtime_config=runtime_config)

    ssl_kwargs: dict[str, object] = {}
    certfile = args.ssl_certfile or os.environ.get("ARL_SSL_CERTFILE")
    keyfile = args.ssl_keyfile or os.environ.get("ARL_SSL_KEYFILE")
    ca_certs = args.ssl_ca_certs or os.environ.get("ARL_SSL_CA_CERTS")
    if certfile:
        import ssl

        ssl_kwargs["ssl_certfile"] = certfile
        ssl_kwargs["ssl_keyfile"] = keyfile
        if ca_certs:
            ssl_kwargs["ssl_ca_certs"] = ca_certs
            ssl_kwargs["ssl_cert_reqs"] = ssl.CERT_REQUIRED

    uvicorn.run(app, host=args.host, port=args.port, **ssl_kwargs)


if __name__ == "__main__":
    main()
