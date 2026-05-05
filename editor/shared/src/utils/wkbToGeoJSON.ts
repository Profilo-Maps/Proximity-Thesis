import type { Geometry, Point, LineString, MultiPoint, MultiLineString } from 'geojson';

const WKB_POINT = 1;
const WKB_LINESTRING = 2;
const WKB_MULTIPOINT = 4;
const WKB_MULTILINESTRING = 5;

class WKBReader {
  private data: DataView;
  private offset: number = 0;
  private littleEndian: boolean = true;

  constructor(buffer: ArrayBuffer) {
    this.data = new DataView(buffer);
  }

  readByte(): number {
    const val = this.data.getUint8(this.offset);
    this.offset += 1;
    return val;
  }

  readUint32(): number {
    const val = this.data.getUint32(this.offset, this.littleEndian);
    this.offset += 4;
    return val;
  }

  readFloat64(): number {
    const val = this.data.getFloat64(this.offset, this.littleEndian);
    this.offset += 8;
    return val;
  }

  readCoord(): [number, number] {
    const x = this.readFloat64();
    const y = this.readFloat64();
    return [x, y];
  }

  readCoordArray(count: number): [number, number][] {
    const coords: [number, number][] = [];
    for (let i = 0; i < count; i++) {
      coords.push(this.readCoord());
    }
    return coords;
  }

  readGeometry(): Geometry {
    const byteOrder = this.readByte();
    this.littleEndian = byteOrder === 1;
    const typeCode = this.readUint32();
    const baseType = typeCode & 0xFF;

    switch (baseType) {
      case WKB_POINT: return this.readPoint();
      case WKB_LINESTRING: return this.readLineString();
      case WKB_MULTIPOINT: return this.readMultiPoint();
      case WKB_MULTILINESTRING: return this.readMultiLineString();
      default: throw new Error(`Unsupported WKB geometry type: ${baseType}`);
    }
  }

  private readPoint(): Point {
    return { type: 'Point', coordinates: this.readCoord() };
  }

  private readLineString(): LineString {
    const numPoints = this.readUint32();
    return { type: 'LineString', coordinates: this.readCoordArray(numPoints) };
  }

  private readMultiPoint(): MultiPoint {
    const numPoints = this.readUint32();
    const coordinates: [number, number][] = [];
    for (let i = 0; i < numPoints; i++) {
      this.readByte();
      this.readUint32();
      coordinates.push(this.readCoord());
    }
    return { type: 'MultiPoint', coordinates };
  }

  private readMultiLineString(): MultiLineString {
    const numLines = this.readUint32();
    const coordinates: [number, number][][] = [];
    for (let i = 0; i < numLines; i++) {
      this.readByte();
      this.readUint32();
      const numPoints = this.readUint32();
      coordinates.push(this.readCoordArray(numPoints));
    }
    return { type: 'MultiLineString', coordinates };
  }
}

function hexToBuffer(hex: string): ArrayBuffer {
  const bytes = new Uint8Array(hex.length / 2);
  for (let i = 0; i < hex.length; i += 2) {
    bytes[i / 2] = parseInt(hex.substring(i, i + 2), 16);
  }
  return bytes.buffer;
}

export function wkbToGeoJSON(hexString: string | null | undefined): Geometry | null {
  if (!hexString || hexString.length === 0) return null;
  try {
    const buffer = hexToBuffer(hexString);
    const reader = new WKBReader(buffer);
    return reader.readGeometry();
  } catch (error) {
    console.warn('[wkbToGeoJSON] Failed to decode WKB hex:', error);
    return null;
  }
}
