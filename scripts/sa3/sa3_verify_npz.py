"""Verify every array in a Beat This spectrogram bundle actually loads.

A SIGTERM during spectrogram creation leaves partially-written .npy files. Some
are empty, but others are truncated mid-array -- those pass a file-size sanity
check and only fail at training time, deep inside a DataLoader worker, hours in.
This reads each entry's npy header and compares the declared shape against the
real payload size, which is cheap (headers only) and catches both cases.
"""
import argparse, io, zipfile
import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("npz")
    ap.add_argument("--delete-bad-pieces", action="store_true",
                    help="print the piece dirs to remove, one per line, to stdout")
    a = ap.parse_args()

    zf = zipfile.ZipFile(a.npz)
    names = zf.namelist()
    bad = []
    for n in names:
        info = zf.getinfo(n)
        try:
            with zf.open(n) as f:
                version = np.lib.format.read_magic(f)
                shape, _, dtype = np.lib.format._read_array_header(f, version)
                need = int(np.prod(shape)) * dtype.itemsize
                have = info.file_size - f.tell()
                if have != need:
                    bad.append((n, shape, need, have))
        except Exception as e:
            bad.append((n, None, None, repr(e)[:60]))

    pieces = sorted({n.split("/")[0] for n, *_ in bad})
    if a.delete_bad_pieces:
        for p in pieces:
            print(p)
        return
    print(f"entries: {len(names)}   corrupt: {len(bad)}   affected pieces: {len(pieces)}")
    for n, shape, need, have in bad[:15]:
        print(f"  {n}  shape={shape} need={need} have={have}")
    if len(bad) > 15:
        print(f"  ... and {len(bad)-15} more")


if __name__ == "__main__":
    main()
