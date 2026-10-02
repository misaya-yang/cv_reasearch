#!/usr/bin/env python3
"""Download the frozen ThinObject5K subset with stdlib ZIP range reads only.

No full-archive download, model execution, or dataset evaluation. Planned lists
are distinct from the receipt's verified ready pairs. Run again to resume.
"""
import argparse
import concurrent.futures
import hashlib
import http.client
import json
import os
from pathlib import Path, PurePosixPath
import re
import struct
import threading
import time
import urllib.error
import urllib.request
import uuid
import zlib

ROOT = Path(__file__).resolve().parents[1]
CHUNK = 64 * 1024


def atomic_bytes(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    try:
        with temporary.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def file_identity(path):
    digest, crc, size = hashlib.sha256(), 0, 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(CHUNK), b""):
            size += len(block)
            crc = zlib.crc32(block, crc)
            digest.update(block)
    return {"bytes": size, "crc32_hex": f"{crc & 0xffffffff:08x}",
            "sha256": digest.hexdigest()}


def matches_entry(identity, entry):
    return (identity["bytes"] == entry["uncompressed_bytes"]
            and identity["crc32_hex"] == entry["crc32_hex"].lower())


def safe_error(error):
    # HTTP exceptions may carry temporary signed redirect URLs. Never log them.
    if isinstance(error, urllib.error.HTTPError):
        return f"HTTPError: status={error.code}"
    return re.sub(r"https?://\S+", "[URL redacted]",
                  f"{type(error).__name__}: {error}")[:500]


def validate_range_response(response, start, length, archive_size):
    if response.status != 206:
        raise ValueError(f"Expected HTTP206; got {response.status}; body not read")
    expected = f"bytes {start}-{start + length - 1}/{archive_size}"
    if response.headers.get("Content-Range") != expected:
        raise ValueError("Content-Range does not match requested range/archive total")
    content_length = response.headers.get("Content-Length")
    if content_length is not None and int(content_length) != length:
        raise ValueError("Content-Length does not match requested range")
    if "text/html" in response.headers.get("Content-Type", "").lower():
        raise ValueError("HTML confirmation/quota response rejected")


def inflate_atomic(part, target, entry):
    temporary = target.with_name(target.name + ".tmp-" + uuid.uuid4().hex)
    target.parent.mkdir(parents=True, exist_ok=True)
    decoder = zlib.decompressobj(-15) if entry["compression_method"] == 8 else None
    digest, crc, size = hashlib.sha256(), 0, 0
    try:
        with part.open("rb") as source, temporary.open("xb") as output:
            def write(block):
                nonlocal crc, size
                size += len(block)
                if size > entry["uncompressed_bytes"]:
                    raise ValueError("Inflated data exceeds manifest size")
                output.write(block)
                crc = zlib.crc32(block, crc)
                digest.update(block)

            for block in iter(lambda: source.read(CHUNK), b""):
                write(decoder.decompress(block) if decoder else block)
            if decoder:
                write(decoder.flush())
                if not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:
                    raise ValueError("Incomplete or trailing raw-DEFLATE stream")
            identity = {"bytes": size, "crc32_hex": f"{crc & 0xffffffff:08x}",
                        "sha256": digest.hexdigest()}
            if not matches_entry(identity, entry):
                raise ValueError("Inflated CRC32/size does not match manifest")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, target)
        return identity
    finally:
        if temporary.exists():
            temporary.unlink()


def load_plan(path):
    raw = path.read_bytes()
    manifest = json.loads(raw)
    url = manifest["canonical_public_download_url"]
    if not url.startswith("https://drive.usercontent.google.com/download?"):
        raise ValueError("Expected the manifest's official Google download endpoint")
    archive_size = manifest["archive_bytes"]
    entries, roles = [], {}
    for subset in manifest["subsets"]:
        role = subset["role"]
        if role not in ("development", "heldout_report") or role in roles:
            raise ValueError("Unexpected or duplicated subset role")
        ids = subset["mask_ids"]
        if len(ids) != subset["count"] or len(ids) != len(set(ids)):
            raise ValueError("Invalid subset ID count")
        expected_names = set()
        for mask_id in ids:
            if PurePosixPath(mask_id).name != mask_id or not mask_id.endswith(".png"):
                raise ValueError("Invalid frozen mask ID")
            expected_names.update((f"ThinObject5K/images/{mask_id[:-4]}.jpg",
                                   f"ThinObject5K/masks/{mask_id}"))
        actual_names = {entry["name"] for entry in subset["zip_entries"]}
        if actual_names != expected_names or len(actual_names) != len(subset["zip_entries"]):
            raise ValueError("Frozen IDs do not match archive entry names")
        roles[role] = subset
        for entry in subset["zip_entries"]:
            name = PurePosixPath(entry["name"])
            if name.is_absolute() or ".." in name.parts or len(name.parts) != 3:
                raise ValueError("Unsafe archive path")
            if entry["flags"] & 1 or entry["compression_method"] not in (0, 8):
                raise ValueError("Encrypted/unsupported ZIP entry")
            if (entry["compressed_bytes"] <= 0 or entry["uncompressed_bytes"] <= 0
                    or entry["local_header_offset"] < 0
                    or entry["local_header_offset"] + 30 + entry["compressed_bytes"] > archive_size):
                raise ValueError("Invalid ZIP entry bounds")
            entries.append(entry)
    if len(roles) != 2 or set(roles["development"]["mask_ids"]) & set(roles["heldout_report"]["mask_ids"]):
        raise ValueError("Missing/overlapping development and report subsets")
    return manifest, entries, roles, hashlib.sha256(raw).hexdigest()


class RangeDownloader:
    def __init__(self, manifest, output, retries, timeout):
        self.url = manifest["canonical_public_download_url"]
        self.archive_size = manifest["archive_bytes"]
        self.output, self.retries, self.timeout = output, retries, timeout
        self.lock = threading.Lock()
        self.received_bytes = 0
        self.completed = 0
        self.progress = output / "progress.jsonl"
        self.partial = output / ".partial"
        self.partial.mkdir(parents=True, exist_ok=True)

    def received(self, count):
        with self.lock:
            self.received_bytes += count

    def range_to(self, start, length, destination):
        if start < 0 or length <= 0 or start + length > self.archive_size:
            raise ValueError("Requested range outside archive")
        request = urllib.request.Request(self.url, headers={
            "Range": f"bytes={start}-{start + length - 1}",
            "Accept-Encoding": "identity"})
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            validate_range_response(response, start, length, self.archive_size)
            remaining = length
            while remaining:
                try:
                    block = response.read(min(CHUNK, remaining))
                except http.client.IncompleteRead as error:
                    block = error.partial
                    self.received(len(block))
                    destination.write(block)
                    destination.flush()
                    raise
                self.received(len(block))
                if not block:
                    raise EOFError("Short range response; partial payload retained")
                destination.write(block)
                remaining -= len(block)
                destination.flush()

    def range_bytes(self, start, length):
        import io
        buffer = io.BytesIO()
        self.range_to(start, length, buffer)
        return buffer.getvalue()

    def payload_offset(self, entry):
        codec = "utf-8" if entry["flags"] & 0x800 else "cp437"
        expected_name = entry["name"].encode(codec)
        header = self.range_bytes(entry["local_header_offset"], 30 + len(expected_name))
        values = struct.unpack("<4s5H3L2H", header[:30])
        if (values[0] != b"PK\x03\x04" or values[2] != entry["flags"]
                or values[3] != entry["compression_method"]
                or values[-2] != len(expected_name) or header[30:] != expected_name):
            raise ValueError("Local ZIP header/filename differs from manifest")
        return entry["local_header_offset"] + 30 + values[-2] + values[-1]

    def record(self, result):
        with self.lock:
            self.completed += 1
            event = dict(result, completed=self.completed,
                         received_body_bytes_this_run=self.received_bytes)
            with self.progress.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(event, ensure_ascii=False) + "\n")
            if result["status"] == "failed" or self.completed % 25 == 0:
                print(json.dumps(event, ensure_ascii=False), flush=True)
        return result

    def download(self, entry):
        target = self.output.joinpath(*PurePosixPath(entry["name"]).parts)
        if target.is_file():
            try:
                identity = file_identity(target)
            except OSError as error:
                return self.record(dict(name=entry["name"], status="failed", attempts=0,
                                        error=safe_error(error)))
            if matches_entry(identity, entry):
                return self.record(dict(name=entry["name"], status="verified_existing",
                                        attempts=0, **identity))
        key = hashlib.sha256(f"{self.archive_size}:{entry['local_header_offset']}:"
                             f"{entry['crc32_hex']}:{entry['name']}".encode()).hexdigest()
        part = self.partial / (key + ".compressed.part")
        last_error = None
        for attempt in range(1, self.retries + 1):
            try:
                # One combined metadata request plus one payload request.
                payload_start = self.payload_offset(entry)
                current = part.stat().st_size if part.exists() else 0
                if current > entry["compressed_bytes"]:
                    part.unlink()
                    current = 0
                if current < entry["compressed_bytes"]:
                    with part.open("ab") as payload:
                        self.range_to(payload_start + current,
                                      entry["compressed_bytes"] - current, payload)
                try:
                    identity = inflate_atomic(part, target, entry)
                except (ValueError, zlib.error):
                    # Complete but invalid payload must be refetched; partial
                    # network payloads remain available across retries/runs.
                    part.unlink()
                    raise
                part.unlink()
                return self.record(dict(name=entry["name"], status="downloaded_verified",
                                        attempts=attempt, **identity))
            except Exception as error:
                last_error = safe_error(error)
                if attempt < self.retries:
                    time.sleep(min(2 ** (attempt - 1), 4))
        return self.record(dict(name=entry["name"], status="failed",
                                attempts=self.retries, error=last_error))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path,
                        default=ROOT / "assets/manifests/thinobject5k_subset_ranges.json")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "assets/datasets/thinobject5k_core_v1")
    parser.add_argument("--workers", type=int, choices=range(1, 9), default=4)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--limit-files", type=int,
                        help="Asset smoke check only; marks the full subset incomplete")
    args = parser.parse_args()
    if args.retries < 1 or args.timeout <= 0 or (args.limit_files is not None and args.limit_files < 1):
        parser.error("Retries, timeout and optional limit must be positive")
    manifest, entries, roles, manifest_sha = load_plan(args.manifest)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    previous_bytes = 0
    receipt_path = output / "receipt.json"
    if receipt_path.exists():
        previous = json.loads(receipt_path.read_text())
        if previous["manifest_sha256"] != manifest_sha:
            raise ValueError("Output already belongs to a different frozen manifest")
        previous_bytes = previous.get("received_body_bytes_total_recorded_runs",
                                      previous["received_body_bytes_this_run"])
    for role, subset in roles.items():
        list_name = "development" if role == "development" else "report"
        atomic_bytes(output / f"ThinObject5K/list/{list_name}.txt",
                     ("\n".join(subset["mask_ids"]) + "\n").encode())
    chosen = entries[:args.limit_files] if args.limit_files else entries
    downloader = RangeDownloader(manifest, output, args.retries, args.timeout)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        results = list(executor.map(downloader.download, chosen))
    successful = {r["name"] for r in results if r["status"] != "failed"}
    ready = {}
    for role, subset in roles.items():
        ready[role] = [mid for mid in subset["mask_ids"] if
                      f"ThinObject5K/images/{mid[:-4]}.jpg" in successful
                      and f"ThinObject5K/masks/{mid}" in successful]
        list_name = "development" if role == "development" else "report"
        atomic_bytes(output / f"ThinObject5K/list/{list_name}_ready.txt",
                     ("\n".join(ready[role]) + ("\n" if ready[role] else "")).encode())
    failed = sum(r["status"] == "failed" for r in results)
    receipt = {
        "manifest_sha256": manifest_sha, "official_file_id": manifest["official_file_id"],
        "archive_bytes": manifest["archive_bytes"], "workers": args.workers,
        "planned_files": len(entries), "requested_files_this_run": len(chosen),
        "verified_files_this_run": len(successful), "failed_files_this_run": failed,
        "received_body_bytes_this_run": downloader.received_bytes,
        "received_body_bytes_total_recorded_runs": previous_bytes + downloader.received_bytes,
        "byte_counter_scope": "Bytes read from HTTP response bodies, including headers-as-ZIP-metadata and retries; excludes HTTP/TCP overhead.",
        "full_subset_ready": len(successful) == len(entries),
        "scope": "Frozen asset subset only; report200/official-test500 is a subset, not the full benchmark. No model/quality/performance evaluation.",
        "planned_pair_counts": {role: subset["count"] for role, subset in roles.items()},
        "verified_ready_pair_counts": {role: len(ids) for role, ids in ready.items()},
        "list_contract": "development.txt/report.txt freeze planned IDs; *_ready.txt lists verified pairs in this invocation. Full use requires full_subset_ready=true.",
        "files": results,
    }
    atomic_bytes(output / "receipt.json", (json.dumps(receipt, ensure_ascii=False, indent=2) + "\n").encode())
    print(json.dumps({k: v for k, v in receipt.items() if k != "files"}, ensure_ascii=False), flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(json.dumps({"status": "failed", "error": safe_error(error)}), flush=True)
        raise SystemExit(1)
