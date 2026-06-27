"""navigate.py - coordinate-feedback overworld movement. The verifiable nav core.

walk_to() steps toward a target tile using real_coords() feedback with stuck
detection - no blind button counting. A route is just a list of waypoints fed
through this. ZERO Kira imports.
"""
import firered_ram as ram


def coords(bridge):
    """In-game (x,y) or None (rejects the boot zeroed-SaveBlock1 state)."""
    co = ram.read_player_coords(bridge)
    if co is None or co == (0, 0) or not (0 <= co[0] < 1000 and 0 <= co[1] < 1000):
        return None
    return co


def _step(bridge, key, hold, render):
    bridge.press(key, hold, hold, render)


def walk_to(bridge, target, hold=8, max_steps=40, render=lambda: None, log=print):
    """Greedily walk toward target=(tx,ty). Returns (reached, final_coords, steps).
    On a blocked axis it tries the other axis; gives up after repeated no-progress."""
    tx, ty = target
    stuck = 0
    last = coords(bridge)
    for step in range(max_steps):
        cur = coords(bridge)
        if cur is None:
            log(f"   [Nav] lost coords at step {step} (left overworld?)")
            return False, cur, step
        if cur == (tx, ty):
            return True, cur, step
        dx, dy = tx - cur[0], ty - cur[1]
        # prefer the axis with the larger remaining distance; flip on being stuck
        order = ["x", "y"] if abs(dx) >= abs(dy) else ["y", "x"]
        if stuck % 2 == 1:
            order.reverse()
        key = None
        for ax in order:
            if ax == "x" and dx != 0:
                key = "RIGHT" if dx > 0 else "LEFT"
                break
            if ax == "y" and dy != 0:
                key = "DOWN" if dy > 0 else "UP"
                break
        if key is None:
            return cur == (tx, ty), cur, step
        _step(bridge, key, hold, render)
        nxt = coords(bridge)
        if nxt == last:                 # no progress this step
            stuck += 1
            if stuck >= 6:
                log(f"   [Nav] STUCK near {cur}, target {target} (blocked)")
                return False, cur, step
        else:
            stuck = 0
        last = nxt
    return coords(bridge) == (tx, ty), coords(bridge), max_steps
