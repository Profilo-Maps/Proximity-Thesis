import type { FeatureCollection, Feature } from 'geojson';
import type { SaveRequest, SaveResponse } from '@proximity/shared/types/changeset';

const BASE = '/api';

export async function fetchFeatures(
  parquet: string,
  bbox: [number, number, number, number],
  zoom: number
): Promise<FeatureCollection> {
  const [minLon, minLat, maxLon, maxLat] = bbox;
  const params = new URLSearchParams({
    zoom: String(zoom),
    min_lon: String(minLon),
    min_lat: String(minLat),
    max_lon: String(maxLon),
    max_lat: String(maxLat),
  });
  const res = await fetch(`${BASE}/features/${parquet}?${params}`);
  if (!res.ok) throw new Error(`Failed to fetch features: ${res.status}`);
  return res.json();
}

export async function fetchRows(
  parquet: string,
  bbox: [number, number, number, number]
): Promise<Record<string, unknown>[]> {
  const [minLon, minLat, maxLon, maxLat] = bbox;
  const params = new URLSearchParams({
    min_lon: String(minLon),
    min_lat: String(minLat),
    max_lon: String(maxLon),
    max_lat: String(maxLat),
  });
  const res = await fetch(`${BASE}/rows/${parquet}?${params}`);
  if (!res.ok) throw new Error(`Failed to fetch rows: ${res.status}`);
  const data = await res.json();
  return data.rows;
}

export async function fetchHulls(
  parquet: string,
  bbox: [number, number, number, number]
): Promise<{ hulls: Feature[]; slots: Feature[] }> {
  const [minLon, minLat, maxLon, maxLat] = bbox;
  const params = new URLSearchParams({
    min_lon: String(minLon),
    min_lat: String(minLat),
    max_lon: String(maxLon),
    max_lat: String(maxLat),
  });
  const res = await fetch(`${BASE}/hulls/${parquet}?${params}`);
  if (!res.ok) throw new Error(`Failed to fetch hulls: ${res.status}`);
  return res.json();
}

export async function fetchParquetList(): Promise<string[]> {
  const res = await fetch(`${BASE}/parquets`);
  if (!res.ok) throw new Error(`Failed to fetch parquet list: ${res.status}`);
  const data = await res.json();
  return data.files;
}

export async function fetchConfig(parquet: string): Promise<{
  fields: { name: string; type: string; default: unknown }[];
  global: Record<string, unknown>;
  city: Record<string, unknown>;
}> {
  const res = await fetch(`${BASE}/config/${parquet}`);
  if (!res.ok) throw new Error(`Failed to fetch config: ${res.status}`);
  return res.json();
}

export async function saveConfig(
  parquet: string,
  tier: 'global' | 'city',
  values: Record<string, unknown>
): Promise<void> {
  const res = await fetch(`${BASE}/config/${parquet}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ tier, values }),
  });
  if (!res.ok) throw new Error(`Failed to save config: ${res.status}`);
}

export async function saveChangeset(request: SaveRequest): Promise<SaveResponse> {
  const res = await fetch(`${BASE}/save`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  });
  if (!res.ok) throw new Error(`Failed to save: ${res.status}`);
  return res.json();
}

export interface SaveProgress {
  stage: string;
  pct: number;
  stages_run?: number[];
  rows_affected?: number;
  message?: string;
}

/**
 * Stream save progress from /save/stream.
 * Calls onProgress with each update; resolves with the final SaveResponse on "done".
 */
export async function saveChangesetStream(
  request: SaveRequest,
  onProgress: (p: SaveProgress) => void,
): Promise<SaveResponse> {
  const res = await fetch(`${BASE}/save/stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  });
  if (!res.ok) throw new Error(`Failed to save: ${res.status}`);
  if (!res.body) throw new Error('No response body');

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let finalResult: SaveResponse | null = null;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n');
    buffer = lines.pop() ?? '';
    for (const line of lines) {
      if (!line.trim()) continue;
      const msg: SaveProgress = JSON.parse(line);
      onProgress(msg);
      if (msg.stage === 'error') throw new Error(msg.message ?? 'Save failed');
      if (msg.stage === 'done') {
        finalResult = { status: 'ok', stages_run: msg.stages_run ?? [], rows_affected: msg.rows_affected ?? 0, bbox_used: request.bbox };
      }
    }
  }

  if (!finalResult) throw new Error('Save stream ended without completion');
  return finalResult;
}
