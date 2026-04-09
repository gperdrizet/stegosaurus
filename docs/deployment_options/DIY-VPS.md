# Deploying Stegosaurus: DIY (Local GPU + VPS)

## Concept

Run the model on your local machine using your own GPU. A cheap VPS acts as a public reverse proxy, connected to the local machine via a WireGuard tunnel. The VPS never touches the model - it just forwards traffic.

```
User → VPS (nginx + TLS, public IP) → WireGuard tunnel → Local machine (Gradio + GPU)
```

- **Compute cost:** ~$3–6/mo (VPS only) - local GPU is free
- **Encode latency:** ~2s (depends on local GPU)
- **Availability:** Local machine must be on and connected

## 1. Provision a VPS

Any small VPS works - 1 vCPU / 1 GB RAM is sufficient. The model doesn't run here.

| Provider | Plan | Cost |
|---|---|---|
| Hetzner | CX22 | ~€3.29/mo |
| DigitalOcean | Basic Droplet | ~$6/mo |
| Vultr / Linode | Nanode / Shared | ~$5/mo |

Open ports in the VPS firewall: **80/TCP**, **443/TCP**, **51820/UDP** (WireGuard).

## 2. Set up WireGuard

**On the VPS:**
```bash
apt install wireguard

# Generate keypair
wg genkey | tee /etc/wireguard/server_private.key | wg pubkey > /etc/wireguard/server_public.key

cat > /etc/wireguard/wg0.conf << 'WGEOF'
[Interface]
Address = 10.0.0.1/24
ListenPort = 51820
PrivateKey = <contents of server_private.key>

[Peer]
PublicKey = <contents of local_public.key>
AllowedIPs = 10.0.0.2/32
WGEOF

systemctl enable --now wg-quick@wg0
```

**On your local machine:**
```bash
apt install wireguard

# Generate keypair
wg genkey | tee /etc/wireguard/local_private.key | wg pubkey > /etc/wireguard/local_public.key

cat > /etc/wireguard/wg0.conf << 'WGEOF'
[Interface]
Address = 10.0.0.2/24
PrivateKey = <contents of local_private.key>

[Peer]
PublicKey = <contents of server_public.key>
Endpoint = <vps_public_ip>:51820
AllowedIPs = 10.0.0.1/32
PersistentKeepalive = 25
WGEOF

systemctl enable --now wg-quick@wg0
```

`PersistentKeepalive = 25` keeps the tunnel alive through home NAT/firewalls.

**Verify the tunnel:**
```bash
# On either machine:
ping 10.0.0.1   # from local → VPS
ping 10.0.0.2   # from VPS → local
```

## 3. Run the app locally

`demo/app.py` already binds on all interfaces and reads the port from the `PORT` environment variable (default `8080`). Start the app:

```bash
cd /workspaces/stegosaurus
python demo/app.py
```

## 4. Configure nginx on the VPS

```bash
apt install nginx certbot python3-certbot-nginx
```

Create `/etc/nginx/sites-available/stegosaurus`:
```nginx
server {
    listen 80;
    server_name yourdomain.com;

    location / {
        proxy_pass http://10.0.0.2:8080;

        # Required for Gradio WebSocket connections
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;

        proxy_read_timeout 300s;
        proxy_send_timeout 300s;
    }
}
```

```bash
ln -s /etc/nginx/sites-available/stegosaurus /etc/nginx/sites-enabled/
nginx -t && systemctl reload nginx
```

**Add TLS with Certbot:**
```bash
certbot --nginx -d yourdomain.com
```

Certbot modifies the nginx config automatically and sets up auto-renewal. Point your domain's A record at the VPS public IP before running this.

## Common pitfalls

| Symptom | Cause | Fix |
|---|---|---|
| 502 Bad Gateway | App not running locally, or tunnel down | Check `wg show` and `python demo/app.py` |
| WebSocket errors in browser | Missing `Upgrade`/`Connection` headers | Verify nginx config includes both headers |
| Wrong port | `PORT` env var not set, nginx proxying wrong port | Ensure nginx `proxy_pass` uses port 8080 |
| Tunnel drops after idle | Missing `PersistentKeepalive` | Add `PersistentKeepalive = 25` to local peer config |
| Certbot fails | Domain A record not yet propagated | Wait for DNS propagation, then retry |
