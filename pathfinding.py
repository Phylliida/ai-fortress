"""
pathfinding.py — a small A* for navigating the colony grid.

8-connected grid (octile distance). `passable(x, y) -> bool` lets callers mark obstacles; the default
is an open grid (everything walkable), where the octile heuristic is exact, so A* expands essentially
only the cells along an optimal path — fast even across the full 500x500 map. Returns the path as a
list of (x, y) cells from start to goal inclusive ([] if unreachable; [start] if already there).
"""
import heapq

_SQRT2 = 1.4142135623730951
_ORTHO = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_DIAG = [(-1, -1), (-1, 1), (1, -1), (1, 1)]


def octile(a, b):
    dx, dy = abs(a[0] - b[0]), abs(a[1] - b[1])
    return (dx + dy) + (_SQRT2 - 2) * min(dx, dy)   # = max(dx,dy) + (sqrt2-1)*min(dx,dy)


def astar(start, goal, passable=None, w=500, h=500, diagonal=True):
    """Shortest path on a w x h grid from start to goal (both (x, y) tuples). passable(x,y)->bool marks
    walkable cells (default all). Diagonal moves cost sqrt(2). Returns [start..goal] or []."""
    start = (int(start[0]), int(start[1]))
    goal = (int(goal[0]), int(goal[1]))
    if start == goal:
        return [start]
    walk = passable or (lambda x, y: True)
    if not (0 <= goal[0] < w and 0 <= goal[1] < h and walk(*goal)):
        return []
    steps = _ORTHO + (_DIAG if diagonal else [])
    openh = [(octile(start, goal), 0.0, start)]
    came, g = {}, {start: 0.0}
    while openh:
        _, gc, cur = heapq.heappop(openh)
        if cur == goal:
            path = [cur]
            while cur in came:
                cur = came[cur]
                path.append(cur)
            return path[::-1]
        if gc > g.get(cur, 1e18):
            continue                                 # stale heap entry
        cx, cy = cur
        for dx, dy in steps:
            nx, ny = cx + dx, cy + dy
            if not (0 <= nx < w and 0 <= ny < h) or not walk(nx, ny):
                continue
            ng = gc + (_SQRT2 if dx and dy else 1.0)
            if ng < g.get((nx, ny), 1e18):
                g[(nx, ny)] = ng
                came[(nx, ny)] = cur
                heapq.heappush(openh, (ng + octile((nx, ny), goal), ng, (nx, ny)))
    return []


def next_steps(start, goal, n, passable=None, w=500, h=500):
    """The first `n` cells to walk from start toward goal (excludes start). [] if no path/at goal."""
    path = astar(start, goal, passable, w, h)
    return path[1:1 + n] if len(path) > 1 else []
