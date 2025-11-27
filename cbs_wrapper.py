import os
import json
import ctypes


_DEFAULT_LIB_PATH = os.environ.get(
    "CBS_LIB_PATH",
    os.path.join(os.path.dirname(__file__), "CBS", "libcbs.so"),
)

_cbs_lib = ctypes.CDLL(_DEFAULT_LIB_PATH)

_cbs_lib.solve_cbs.argtypes = [ctypes.c_char_p]
_cbs_lib.solve_cbs.restype = ctypes.c_char_p


def solve_cbs(grid_size, obstacles, starts, goals):
    payload = {
        "grid_size": int(grid_size),
        "obstacles": [[int(x), int(y)] for x, y in obstacles],
        "starts": [[int(x), int(y)] for x, y in starts],
        "goals": [[int(x), int(y)] for x, y in goals],
    }

    in_str = json.dumps(payload).encode("utf-8")
    res_ptr = _cbs_lib.solve_cbs(in_str)

    if not res_ptr:
        raise RuntimeError("solve_cbs returned NULL")

    res_str = ctypes.cast(res_ptr, ctypes.c_char_p).value.decode("utf-8")
    data = json.loads(res_str)

    if "error" in data:
        raise RuntimeError(f"CBS error: {data['error']}")

    paths_raw = data["paths"]
    return {int(k): v for k, v in paths_raw.items()}