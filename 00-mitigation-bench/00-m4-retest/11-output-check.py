"""M4's original-artifact performance policy: file integrity and finite state.

No cross-run numerical equivalence or physical-correctness claim is made.
"""
import math
from pathlib import Path
import struct

POLICY='m4-execution-integrity-v1'
PARTICLES=36504
OUTPUT_BYTES=8+PARTICLES*9*4

def inspect_output(path):
    path=Path(path)
    if path.stat().st_size!=OUTPUT_BYTES:
        raise ValueError('fluid output length mismatch')
    data=path.read_bytes()
    if len(data)!=OUTPUT_BYTES:
        raise ValueError('fluid output length changed while reading')
    rest,count=struct.unpack_from('<fi',data)
    if rest!=392 or count!=PARTICLES:
        raise ValueError('unexpected fluid output header')
    if not all(math.isfinite(value) for row in struct.iter_unpack('<9f',data[8:]) for value in row):
        raise ValueError('non-finite particle state')
    return {'kind':POLICY,'particles':count,'bytes':len(data),
            'rest_particles_per_meter':rest,'finite_components':PARTICLES*9}
