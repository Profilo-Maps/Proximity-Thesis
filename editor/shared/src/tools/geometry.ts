/**
 * Pure geometry helpers for tool operations.
 * No platform dependencies — works in web, RN, Node, etc.
 */

/**
 * Project a point onto a line segment and return the closest point + distance.
 * All coordinates in the same CRS (typically WGS84 lng/lat).
 */
export function projectPointOnSegment(
  px: number, py: number,
  ax: number, ay: number,
  bx: number, by: number,
): { x: number; y: number; dist: number } {
  const dx = bx - ax, dy = by - ay;
  const len2 = dx * dx + dy * dy;
  if (len2 === 0) return { x: ax, y: ay, dist: Math.sqrt((px - ax) ** 2 + (py - ay) ** 2) };
  let t = ((px - ax) * dx + (py - ay) * dy) / len2;
  t = Math.max(0, Math.min(1, t));
  const cx = ax + t * dx, cy = ay + t * dy;
  return { x: cx, y: cy, dist: Math.sqrt((px - cx) ** 2 + (py - cy) ** 2) };
}

/**
 * Find the index after which a new vertex should be inserted into a polygon ring
 * to minimise the distance from the ring to the new point.
 * Returns the insertion index (splice at that position).
 */
export function findNearestEdgeInsertIndex(
  ring: [number, number][],
  px: number,
  py: number,
): number {
  let bestIdx = 0;
  let bestDist = Infinity;
  const n = ring.length;
  for (let i = 0; i < n; i++) {
    const [ax, ay] = ring[i];
    const [bx, by] = ring[(i + 1) % n];
    const { dist } = projectPointOnSegment(px, py, ax, ay, bx, by);
    if (dist < bestDist) {
      bestDist = dist;
      bestIdx = i + 1; // insert after vertex i
    }
  }
  return bestIdx;
}

/**
 * Find the nearest point on a LineString or MultiLineString to a given point.
 * Returns null if the geometry has no segments.
 */
export function nearestPointOnLine(
  px: number, py: number,
  coordinates: number[][] | number[][][],
  isMulti: boolean,
): { x: number; y: number; dist: number } | null {
  const lines = isMulti ? (coordinates as number[][][]) : [coordinates as number[][]];
  let best: { x: number; y: number; dist: number } | null = null;

  for (const line of lines) {
    for (let i = 0; i < line.length - 1; i++) {
      const [ax, ay] = line[i];
      const [bx, by] = line[i + 1];
      const result = projectPointOnSegment(px, py, ax, ay, bx, by);
      if (!best || result.dist < best.dist) {
        best = result;
      }
    }
  }
  return best;
}
