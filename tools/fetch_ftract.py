#!/usr/bin/env python3
"""Fetch the few files Brainviewer uses from the F-TRACT atlas zip on Zenodo (915 MB) with HTTP range requests:
the zip's central directory, then only the wanted members. Writes raw/ftract/extracted/<member path>.

usage: python3 -I tools/fetch_ftract.py            (stdlib only; about 140 MB of transfer)

F-TRACT is free for research use only, so the files stay in raw/ and the private `dynamic` branch.
"""
import http.client
import io
import os
import sys
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor

URL = "https://zenodo.org/api/records/7015415/files/f-tract_v2112.zip/content"
BASE = "ages_15_100/sr_8.40/seg_None_None/pl_200/Lausanne2008-33/"
FEATURES = ["probability", "onset_latency", "peak_latency", "axonal_conduction_delay", "amplitude"]
WANT = [f"{BASE}export/{f}/{f}.csv" for f in FEATURES] + \
       [f"{BASE}atlas__ages_15_100__sr_8.40__seg_None_None__pl_200__Lausanne2008-33.pkl.gz",
        "scripts/readme.txt", "scripts/license.txt"]
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "raw", "ftract", "extracted")
CHUNK = 4 << 20


def get(a, b):
    """bytes a..b inclusive, with retries"""
    for attempt in range(8):
        try:
            req = urllib.request.Request(URL, headers={"Range": f"bytes={a}-{b}"})
            with urllib.request.urlopen(req, timeout=120) as r:
                data = r.read()
            if len(data) == b - a + 1:
                return data
        except (OSError, http.client.HTTPException) as e:  # includes truncated bodies
            print("retry", a, e, file=sys.stderr)
    raise SystemExit(f"failed to fetch bytes {a}-{b}")


class Sparse(io.RawIOBase):
    """A read-only view of the remote zip holding only the byte ranges fetched so far."""
    def __init__(self, size):
        self.size, self.pos, self.parts = size, 0, {}

    def fetch(self, a, b, jobs=8):
        starts = [s for s in range(a - a % CHUNK, b, CHUNK) if s not in self.parts]
        with ThreadPoolExecutor(jobs) as ex:
            for s, data in zip(starts, ex.map(lambda s: get(s, min(self.size, s + CHUNK) - 1), starts)):
                self.parts[s] = data

    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos

    def readinto(self, buf):
        n = min(len(buf), self.size - self.pos)
        if n <= 0:
            return 0
        s = self.pos - self.pos % CHUNK
        part = self.parts[s]  # KeyError = a range that wasn't fetched
        k = min(n, s + len(part) - self.pos)
        buf[:k] = part[self.pos - s:self.pos - s + k]
        self.pos += k
        return k


def main():
    with urllib.request.urlopen(urllib.request.Request(URL, method="HEAD")) as r:
        size = int(r.headers["Content-Length"])
    f = Sparse(size)
    tail = get(size - 65536, size - 1)  # zip64 end records, then the central directory they point at
    i = tail.rfind(b"PK\x06\x06")
    cd_size = int.from_bytes(tail[i + 40:i + 48], "little")
    cd_off = int.from_bytes(tail[i + 48:i + 56], "little")
    print(f"central directory: {cd_size / 1e6:.0f} MB", flush=True)
    f.fetch(cd_off, size)
    z = zipfile.ZipFile(io.BufferedReader(f, CHUNK))  # buffered: zipfile expects whole reads
    for info in (z.getinfo(m) for m in WANT):
        f.fetch(info.header_offset, info.header_offset + info.compress_size + 1024, jobs=2)
        dest = os.path.realpath(os.path.join(OUT, info.filename))
        assert dest.startswith(os.path.realpath(OUT) + os.sep), dest
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest, "wb") as fh:
            fh.write(z.read(info))  # checks the CRC
        print("got", info.filename, flush=True)


if __name__ == "__main__":
    main()
