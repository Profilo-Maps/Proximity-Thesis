import type { Geometry, Position } from 'geojson';

const a = 6378137.0;
const f = 1 / 298.257223563;
const b = a * (1 - f);
const e = Math.sqrt(1 - (b * b) / (a * a));
const e2 = e * e;
const ePrime2 = e2 / (1 - e2);

const k0 = 0.9996;
const lon0 = -123 * (Math.PI / 180);
const falseEasting = 500000;
const falseNorthing = 0;

function utmToLatLon(easting: number, northing: number): [number, number] {
  const x = easting - falseEasting;
  const y = northing - falseNorthing;

  const M = y / k0;
  const mu = M / (a * (1 - e2 / 4 - 3 * e2 * e2 / 64 - 5 * e2 * e2 * e2 / 256));

  const e1 = (1 - Math.sqrt(1 - e2)) / (1 + Math.sqrt(1 - e2));
  const phi1 = mu
    + (3 * e1 / 2 - 27 * e1 * e1 * e1 / 32) * Math.sin(2 * mu)
    + (21 * e1 * e1 / 16 - 55 * e1 * e1 * e1 * e1 / 32) * Math.sin(4 * mu)
    + (151 * e1 * e1 * e1 / 96) * Math.sin(6 * mu)
    + (1097 * e1 * e1 * e1 * e1 / 512) * Math.sin(8 * mu);

  const sinPhi1 = Math.sin(phi1);
  const cosPhi1 = Math.cos(phi1);
  const tanPhi1 = sinPhi1 / cosPhi1;

  const N1 = a / Math.sqrt(1 - e2 * sinPhi1 * sinPhi1);
  const T1 = tanPhi1 * tanPhi1;
  const C1 = ePrime2 * cosPhi1 * cosPhi1;
  const R1 = a * (1 - e2) / Math.pow(1 - e2 * sinPhi1 * sinPhi1, 1.5);
  const D = x / (N1 * k0);

  const lat = phi1 - (N1 * tanPhi1 / R1) * (
    D * D / 2
    - (5 + 3 * T1 + 10 * C1 - 4 * C1 * C1 - 9 * ePrime2) * D * D * D * D / 24
    + (61 + 90 * T1 + 298 * C1 + 45 * T1 * T1 - 252 * ePrime2 - 3 * C1 * C1) * D * D * D * D * D * D / 720
  );

  const lon = lon0 + (1 / cosPhi1) * (
    D
    - (1 + 2 * T1 + C1) * D * D * D / 6
    + (5 - 2 * C1 + 28 * T1 - 3 * C1 * C1 + 8 * ePrime2 + 24 * T1 * T1) * D * D * D * D * D / 120
  );

  return [lon * (180 / Math.PI), lat * (180 / Math.PI)];
}

function convertPosition(pos: Position): Position {
  const [lon, lat] = utmToLatLon(pos[0], pos[1]);
  return [lon, lat];
}

export function convertGeometryToWgs84(geom: Geometry): Geometry {
  switch (geom.type) {
    case 'Point':
      return { type: 'Point', coordinates: convertPosition(geom.coordinates) };
    case 'LineString':
      return { type: 'LineString', coordinates: geom.coordinates.map(convertPosition) };
    case 'MultiPoint':
      return { type: 'MultiPoint', coordinates: geom.coordinates.map(convertPosition) };
    case 'MultiLineString':
      return { type: 'MultiLineString', coordinates: geom.coordinates.map(ring => ring.map(convertPosition)) };
    case 'Polygon':
      return { type: 'Polygon', coordinates: geom.coordinates.map(ring => ring.map(convertPosition)) };
    case 'MultiPolygon':
      return { type: 'MultiPolygon', coordinates: geom.coordinates.map(poly => poly.map(ring => ring.map(convertPosition))) };
    case 'GeometryCollection':
      return { type: 'GeometryCollection', geometries: geom.geometries.map(convertGeometryToWgs84) };
    default:
      return geom;
  }
}
