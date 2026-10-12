"""Atomic file writes that tolerate the temporary file locks of Windows."""
import json
import os
import sys
import tempfile
import time


def _replace(source, destination, attempts=41, delay=0.25):
    for attempt in range(attempts):
        try:
            os.replace(source, destination)
            return
        except OSError as error:
            if getattr(error, "winerror", None) not in (5, 32, 33) or attempt == attempts - 1:
                raise
            if attempt == 0:
                print("[io] Windows denied replacement of %s; retrying" % destination, file=sys.stderr, flush=True)
            time.sleep(delay)


def durable_write(path, write, mode):
    """Write to a temporary file beside `path`, flush, fsync, then rename it over `path`.

    tempfile uses its own random generator, so training random streams are untouched.
    """
    path = os.fspath(path)
    fd, temporary = tempfile.mkstemp(prefix=os.path.basename(path) + ".", suffix=".tmp",
                                     dir=os.path.dirname(os.path.abspath(path)))
    with os.fdopen(fd, mode) as stream:
        write(stream)
        stream.flush()
        os.fsync(stream.fileno())
    _replace(temporary, path)


def write_json(path, data, indent=2):
    durable_write(path, lambda f: json.dump(data, f, indent=indent), "w")


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)
