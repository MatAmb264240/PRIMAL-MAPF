import os
import json
import ctypes

print("cbs_wrapper")

_DEFAULT_LIB_PATH = os.environ.get(
    "CBS_LIB_PATH",
    os.path.join(os.path.dirname(__file__), "CBS", "libcbs.so"),
)

_cbs_lib = ctypes.CDLL(_DEFAULT_LIB_PATH)

_cbs_lib.solve_cbs.argtypes = [ctypes.c_char_p]
_cbs_lib.solve_cbs.restype = ctypes.c_char_p

def solve_cbs(grid_size, obstacles, starts, goals):
    print(f"Grid size: {grid_size}")
    print(f"Obstacles: {obstacles}")
    print(f"Starts: {starts}")
    print(f"Goals: {goals}")
    payload = {
        "grid_size": int(grid_size),
        "obstacles": obstacles,
        "starts": [[int(x), int(y)] for x, y in starts],
        "goals": [[int(x), int(y)] for x, y in goals],
    }

    in_str = json.dumps(payload).encode("utf-8")
    res_ptr = _cbs_lib.solve_cbs(in_str)

    if not res_ptr:
        raise RuntimeError("solve_cbs returned NULL")

    res_str = ctypes.cast(res_ptr, ctypes.c_char_p).value.decode("utf-8")
    # print(f"Raw response from solve_cbs: {res_str}")  # Debug print the raw response
    data = json.loads(res_str)
    # print(f"Parsed response data: {data}")  # Zapisz sparsowane dane JSON

    if "error" in data:
        raise RuntimeError(f"CBS error: {data['error']}")

    paths_raw = data.get("paths", None)  # Safely access paths
    if paths_raw is None:
        raise RuntimeError("Paths not found in response")
    return {int(k): v for k, v in paths_raw.items()}

    payload = {
        "grid_size": int(grid_size),
        "obstacles": obstacles,
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