"""Fetch a reproducible small COCO natural-image pool, without downloading image archives."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import time
import urllib.request
import zipfile

from PIL import Image
from wmq_fixture_utils import fixture_lock


BASE = "https://s3.amazonaws.com/images.cocodataset.org"
INDEX_URL = BASE + "/annotations/image_info_test2017.zip"
# COCO archive identity, also the upstream S3 object's ETag (single-part MD5).
INDEX_MD5 = "85da7065e5e600ebfee8af1edb634eb5"


def digest(path, algorithm="sha256"):
    h = hashlib.new(algorithm)
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url, path):
    path = Path(path)
    part = path.with_suffix(path.suffix + ".part")
    for attempt in range(3):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "WMQ-research/1.0"})
            with urllib.request.urlopen(request, timeout=60) as source, part.open("wb") as target:
                while chunk := source.read(1024 * 1024):
                    target.write(chunk)
            part.replace(path)
            return
        except (OSError, TimeoutError):
            part.unlink(missing_ok=True)
            if attempt == 2:
                raise
            time.sleep(attempt + 1)


def prepare(output, count, seed):
    if count < 2:
        raise ValueError("Need at least two images for disjoint train/search")
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with fixture_lock(output):
        output.mkdir(exist_ok=True)
        marker = output / "dataset_provenance.json"
        if marker.exists():
            data = json.loads(marker.read_text(encoding="utf-8"))
            if data["count"] != count or data["seed"] != seed:
                raise ValueError("Dataset cache count/seed differ; choose a fresh output directory")
            if len(data["images"]) != count:
                raise ValueError("Incomplete dataset provenance")
            for entry in data["images"]:
                name = entry["file_name"]
                if Path(name).name != name or digest(output / name) != entry["sha256"]:
                    raise ValueError("Cached natural dataset changed; refusing silent replacement")
            if len(list(output.glob("*.jpg"))) != count:
                raise ValueError("Unexpected image files in natural cache")
            print(f"Reusing verified COCO pool: {output}", flush=True)
            return
        index = output.parent / "image_info_test2017.zip"
        with fixture_lock(index):
            if not index.exists():
                print("Downloading COCO image metadata (no full image archive)...", flush=True)
                download(INDEX_URL, index)
            if digest(index, "md5") != INDEX_MD5:
                raise ValueError("COCO image metadata checksum mismatch; remove the corrupt cached ZIP and retry")
            with zipfile.ZipFile(index) as archive:
                metadata = json.loads(archive.read("annotations/image_info_test2017.json"))
        candidates = sorted(metadata["images"], key=lambda x: x["id"])
        chosen = random.Random(seed).sample(candidates, count)
        expected_names = {f"{int(item['id']):012d}.jpg" for item in chosen}
        if any(p.name not in expected_names for p in output.glob("*.jpg")):
            raise ValueError("Unexpected images in incomplete dataset cache; choose a fresh directory")
        entries = []
        hashes = set()
        for i, item in enumerate(chosen):
            name = f"{int(item['id']):012d}.jpg"
            url = BASE + "/test2017/" + name
            path = output / name
            # Without a completion marker, re-download rather than trusting partial data.
            download(url, path)
            with Image.open(path) as im:
                im.verify()
            checksum = digest(path)
            if checksum in hashes:
                raise ValueError("Duplicate image content in chosen COCO pool; choose another dataset seed")
            hashes.add(checksum)
            entries.append({"file_name": name, "id": item["id"], "url": url,
                            "sha256": checksum, "license": item.get("license"),
                            "flickr_url": item.get("flickr_url")})
            print(f"Natural image {i + 1}/{count}: {name}", flush=True)
        data = {"dataset": "COCO 2017 test images used ONLY as external calibration data",
                "count": count, "seed": seed, "index_url": INDEX_URL,
                "index_md5": INDEX_MD5, "index_sha256": digest(index),
                "image_hash_policy": "SHA256 recorded at first HTTPS download and verified on reuse; not publisher-signed",
                "terms": "https://cocodataset.org/#termsofuse", "licenses": metadata.get("licenses"),
                "assumption": "Natural photographs; not certified absent arbitrary watermarks; no Stable Signature detector used",
                "images": entries}
        tmp = marker.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        tmp.replace(marker)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    p.add_argument("--count", type=int, default=52)
    p.add_argument("--seed", type=int, default=3407)
    args = p.parse_args()
    prepare(args.output, args.count, args.seed)


if __name__ == "__main__":
    main()
