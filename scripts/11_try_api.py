"""Envoie une image a l'API et affiche le JSON. Usage :
    python scripts/11_try_api.py chemin/recu.jpg --method docai
    python scripts/11_try_api.py chemin/recu.jpg --method rules
    python scripts/11_try_api.py chemin/recu.jpg --method docai --dataset cord
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import sys
import urllib.error
import urllib.request
import uuid
from pathlib import Path


def post_image(url: str, path: Path, fields: dict[str, str]) -> tuple[int, dict]:
    boundary = uuid.uuid4().hex
    parts = [f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode() for k, v in fields.items()]
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{path.name}"\r\n'
                 f"Content-Type: {mime}\r\n\r\n".encode() + path.read_bytes() + b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())
    req = urllib.request.Request(url.rstrip("/") + "/extract", data=b"".join(parts),
                                 headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))      # API locale : jamais de proxy
    try:
        with opener.open(req, timeout=300) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("image", type=Path)
    ap.add_argument("--method", choices=["rules", "llm", "docai"], default="docai")
    ap.add_argument("--dataset", choices=["sroie", "cord"], default="sroie")
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    args = ap.parse_args(argv)
    if not args.image.exists():
        print(f"Image introuvable : {args.image}")
        return 2
    try:
        status, body = post_image(args.url, args.image, {"method": args.method, "dataset": args.dataset})
    except urllib.error.URLError as e:
        print(f"API injoignable sur {args.url} ({e.reason}). Lance-la : uvicorn docextract.api:app --port 8000")
        return 1
    print(status, json.dumps(body, indent=2, ensure_ascii=False))
    return 0 if status == 200 else 1


if __name__ == "__main__":
    sys.exit(main())
