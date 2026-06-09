# TLS / mTLS Sidecar Setup

*How to encrypt ARL sidecar connections and enforce mutual TLS for zero-trust deployments.*  
*Last updated: 2026-06-05*

---

## Why TLS/mTLS matters for ARL

The ARL sidecar receives raw feature vectors and emits model predictions.  In regulated
environments (credit, medical, payments), the connection between the model inference
service and the ARL sidecar must be encrypted **and** mutually authenticated:

- **TLS**: encrypts the channel so features are not exposed in transit
- **mTLS**: ensures the sidecar only accepts connections from authorized callers (your
  model server), preventing rogue requests that could trigger adversarial adaptation

---

## Option A: Nginx reverse proxy (recommended for production)

Place an Nginx instance in front of the ARL sidecar.  All TLS termination happens at Nginx;
the sidecar itself runs on localhost plaintext.

```nginx
# /etc/nginx/conf.d/arl.conf
server {
    listen 443 ssl;
    server_name arl.internal;

    # Server certificate
    ssl_certificate     /etc/ssl/arl/server.crt;
    ssl_certificate_key /etc/ssl/arl/server.key;

    # Client certificate (mTLS)
    ssl_client_certificate /etc/ssl/arl/ca.crt;
    ssl_verify_client      on;

    ssl_protocols      TLSv1.2 TLSv1.3;
    ssl_ciphers        HIGH:!aNULL:!MD5;

    location / {
        proxy_pass http://127.0.0.1:8080;
        proxy_set_header X-Forwarded-For $remote_addr;
    }
}
```

Generate a self-signed CA + client/server certificates:

```bash
# CA
openssl genrsa -out ca.key 4096
openssl req -new -x509 -days 1825 -key ca.key -out ca.crt -subj "/CN=ARL CA"

# Server certificate
openssl genrsa -out server.key 2048
openssl req -new -key server.key -out server.csr -subj "/CN=arl.internal"
openssl x509 -req -days 365 -in server.csr -CA ca.crt -CAkey ca.key -CAcreateserial -out server.crt

# Client certificate (your model server)
openssl genrsa -out client.key 2048
openssl req -new -key client.key -out client.csr -subj "/CN=model-server"
openssl x509 -req -days 365 -in client.csr -CA ca.crt -CAkey ca.key -CAcreateserial -out client.crt
```

---

## Option B: Uvicorn with SSL (simple, not mTLS)

Pass `--ssl-keyfile` and `--ssl-certfile` to uvicorn when starting the sidecar:

```bash
arl-serve --config configs/serving_pilot_fraud_torch.yaml \
    --host 0.0.0.0 --port 8443 \
    -- --ssl-keyfile /etc/ssl/arl/server.key \
       --ssl-certfile /etc/ssl/arl/server.crt
```

Or programmatically:

```python
import uvicorn
from adaptive_reliability_layer.serving.app import create_app

app = create_app(config_path="configs/default.yaml")
uvicorn.run(
    app,
    host="0.0.0.0",
    port=8443,
    ssl_keyfile="/etc/ssl/arl/server.key",
    ssl_certfile="/etc/ssl/arl/server.crt",
)
```

---

## Option C: Kubernetes Istio service mesh

If running in Kubernetes with Istio, enable mTLS at the mesh level:

```yaml
# PeerAuthentication — enforce strict mTLS for the arl namespace
apiVersion: security.istio.io/v1beta1
kind: PeerAuthentication
metadata:
  name: arl-mtls
  namespace: arl
spec:
  mtls:
    mode: STRICT
```

No application code changes needed — Istio handles the certificate rotation automatically.

---

## Option D: Application-layer certificate header check

When TLS termination happens upstream (load balancer, Nginx) and you want the ARL
sidecar to additionally verify the client certificate CN, use the built-in header check:

```yaml
# In your serving config YAML
serving:
  trusted_client_cn: model-server   # only requests forwarded from this CN are accepted
  client_cn_header: X-Client-Cert-CN  # header injected by nginx ($ssl_client_s_dn_cn)
```

The sidecar will reject requests where the header is missing or the CN does not match.

---

## Certificate rotation

Certificates should rotate at least annually.  The ARL sidecar supports zero-downtime rotation:

1. Generate new certificate
2. Deploy new certificate to Nginx (reload, not restart): `nginx -s reload`
3. Update `ARL_TLS_CERT` and `ARL_TLS_KEY` environment variables
4. Restart ARL sidecar (policy state is preserved if Redis backend is configured)

---

## Security checklist

- [ ] TLS 1.2 minimum (`ssl_protocols TLSv1.2 TLSv1.3`)
- [ ] Strong cipher suites (`HIGH:!aNULL:!MD5`)
- [ ] Certificate pinning for model server → sidecar connection
- [ ] API key authentication in addition to TLS (defense in depth)
- [ ] Audit log confirms no plaintext connections (`ssl_access_log /var/log/nginx/arl_tls.log`)
- [ ] Certificate expiry alerting (PagerDuty or equivalent)
