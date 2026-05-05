import type { NetworkSegment } from '../types/NetworkSegment';

const EARTH_RADIUS_M = 6371000;

function haversineDistance(lat1: number, lon1: number, lat2: number, lon2: number): number {
  const phi1 = (lat1 * Math.PI) / 180;
  const phi2 = (lat2 * Math.PI) / 180;
  const dPhi = ((lat2 - lat1) * Math.PI) / 180;
  const dLambda = ((lon2 - lon1) * Math.PI) / 180;
  const a2 = Math.sin(dPhi / 2) * Math.sin(dPhi / 2) +
    Math.cos(phi1) * Math.cos(phi2) * Math.sin(dLambda / 2) * Math.sin(dLambda / 2);
  const c = 2 * Math.atan2(Math.sqrt(a2), Math.sqrt(1 - a2));
  return EARTH_RADIUS_M * c;
}

function pointToLineStringDistance(lat: number, lon: number, lineCoords: [number, number][]): number {
  let minDist = Infinity;
  for (let i = 0; i < lineCoords.length - 1; i++) {
    const [lon1, lat1] = lineCoords[i];
    const [lon2, lat2] = lineCoords[i + 1];
    const dist = pointToSegmentDistance(lat, lon, lat1, lon1, lat2, lon2);
    if (dist < minDist) minDist = dist;
  }
  if (lineCoords.length === 1) {
    const [lon1, lat1] = lineCoords[0];
    minDist = haversineDistance(lat, lon, lat1, lon1);
  }
  return minDist;
}

function pointToSegmentDistance(pLat: number, pLon: number, aLat: number, aLon: number, bLat: number, bLon: number): number {
  const dx = bLon - aLon;
  const dy = bLat - aLat;
  const lenSq = dx * dx + dy * dy;
  if (lenSq === 0) return haversineDistance(pLat, pLon, aLat, aLon);
  let t = ((pLon - aLon) * dx + (pLat - aLat) * dy) / lenSq;
  t = Math.max(0, Math.min(1, t));
  const projLon = aLon + t * dx;
  const projLat = aLat + t * dy;
  return haversineDistance(pLat, pLon, projLat, projLon);
}

export class SegmentSpatialGrid {
  private grid: Map<string, Set<string>> = new Map();
  private segmentMap: Map<string, NetworkSegment> = new Map();
  private cellSize: number;

  constructor(segments: NetworkSegment[], cellSize: number = 0.001) {
    this.cellSize = cellSize;
    this.buildIndex(segments);
  }

  private buildIndex(segments: NetworkSegment[]): void {
    for (const segment of segments) {
      this.segmentMap.set(segment.streetGridId, segment);
      this.indexSegment(segment);
    }
  }

  private indexSegment(segment: NetworkSegment): void {
    const coords = segment.street.geometry.coordinates;
    const visitedCells = new Set<string>();
    for (const [lon, lat] of coords) {
      const key = this.getKey(lat, lon);
      if (visitedCells.has(key)) continue;
      visitedCells.add(key);
      let cell = this.grid.get(key);
      if (!cell) { cell = new Set(); this.grid.set(key, cell); }
      cell.add(segment.streetGridId);
    }
  }

  private getKey(lat: number, lon: number): string {
    return `${Math.floor(lat / this.cellSize)}_${Math.floor(lon / this.cellSize)}`;
  }

  queryNearby(lat: number, lon: number, radiusM: number = 50, maxResults: number = 50): NetworkSegment[] {
    const cellsToSearch = Math.ceil(radiusM / (this.cellSize * 111000)) + 1;
    const centerCellLat = Math.floor(lat / this.cellSize);
    const centerCellLon = Math.floor(lon / this.cellSize);
    const candidateIds = new Set<string>();
    for (let dLat = -cellsToSearch; dLat <= cellsToSearch; dLat++) {
      for (let dLon = -cellsToSearch; dLon <= cellsToSearch; dLon++) {
        const key = `${centerCellLat + dLat}_${centerCellLon + dLon}`;
        const cell = this.grid.get(key);
        if (cell) { for (const id of cell) candidateIds.add(id); }
      }
    }
    const results: { segment: NetworkSegment; distance: number }[] = [];
    for (const id of candidateIds) {
      const segment = this.segmentMap.get(id);
      if (!segment) continue;
      const distance = pointToLineStringDistance(lat, lon, segment.street.geometry.coordinates as [number, number][]);
      if (distance <= radiusM) results.push({ segment, distance });
    }
    results.sort((a2, b2) => a2.distance - b2.distance);
    return results.slice(0, maxResults).map(r => r.segment);
  }

  getById(streetGridId: string): NetworkSegment | null {
    return this.segmentMap.get(streetGridId) ?? null;
  }

  updateSegment(segment: NetworkSegment): void {
    this.segmentMap.set(segment.streetGridId, segment);
  }

  getStats(): { totalCells: number; totalSegments: number } {
    return { totalCells: this.grid.size, totalSegments: this.segmentMap.size };
  }
}
