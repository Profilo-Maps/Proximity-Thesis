/**
 * Platform-agnostic tool click handlers.
 * Pure functions that take a tap event + adapter + callbacks + session state,
 * and produce side effects only through the callbacks.
 *
 * No DOM, no MapLibre, no React — portable to web and React Native.
 */

import type {
  ToolId,
  ToolSubtype,
  MapTapEvent,
  MapAdapter,
  ToolCallbacks,
  ToolSessionState,
} from './types';
import { findNearestEdgeInsertIndex } from './geometry';

/** Status hint text per tool */
export const TOOL_STATUS: Record<string, string> = {
  move_point: 'Move: drag vertex | Snap: tap node to snap to nearest segment',
  add_node: 'Select subtype, then tap to place point',
  delete_point: 'Tap a point to delete it',
  highlight_point: 'Tap a feature to highlight it',
  merge_segments: 'Tap first segment, then second segment to merge',
  split_segment: 'Tap on a segment to split at that point',
  draw_segment: 'Select subtype (Crosswalk: tap two ramps), tap points to draw, Enter to complete',
  add_polygon: 'Tap vertices to draw polygon, Enter to complete',
  delete_polygon: 'Tap a hull polygon to delete it',
  edit_polygon_face: 'Tap a hull to select it, then choose an action from the panel',
};

/**
 * Handle a single tap event for the active tool.
 * Mutates `session` in place for multi-tap tools.
 */
export function handleToolTap(
  tool: ToolId,
  tap: MapTapEvent,
  adapter: MapAdapter,
  cb: ToolCallbacks,
  session: ToolSessionState,
  subtype?: ToolSubtype | null,
): void {
  const { lng, lat } = tap;

  switch (tool) {
    case 'add_node': {
      const st = subtype ?? 'node';

      if (st === 'ramp') {
        // Two-step: first select a sidewalk, then click to place the ramp
        if (!session.rampTarget) {
          const sw = adapter.queryFeatures(tap, ['sidewalk']);
          if (!sw) {
            cb.setStatusText('Tap a sidewalk to attach a curb ramp to');
            break;
          }
          const segId = (sw.properties._seg_id ?? '') as string;
          const side = (sw.properties._side ?? 'left') as string;
          // Determine position (start/end) based on which end is closer
          const swCoords = sw.geometry.type === 'LineString' ? sw.geometry.coordinates : [];
          let position = 'start';
          if (swCoords.length >= 2) {
            const dStart = Math.hypot(swCoords[0][0] - lng, swCoords[0][1] - lat);
            const dEnd = Math.hypot(swCoords[swCoords.length - 1][0] - lng, swCoords[swCoords.length - 1][1] - lat);
            position = dStart <= dEnd ? 'start' : 'end';
          }
          session.rampTarget = { segId, side, position };
          cb.addEditPreview(sw.geometry, 'ghost');
          cb.setStatusText(`Sidewalk ${segId} (${side}/${position}) selected — tap to place curb ramp`);
        } else {
          const rt = session.rampTarget;
          const rampId = `ramp_${Date.now()}`;
          cb.recordAddedNode({ x: lng, y: lat });
          cb.mutateSource({
            type: 'add',
            feature: {
              type: 'Feature',
              geometry: { type: 'Point', coordinates: [lng, lat] },
              properties: {
                _t: 'ramp', _fid: rampId,
                _seg_id: rt.segId, _side: rt.side,
                _position: rt.position, _index: 1,
                _ramp_disabled: 'no',
              },
            },
          });
          cb.setStatusText(`Curb ramp placed on ${rt.segId} (${rt.side}/${rt.position})`);
          session.rampTarget = null;
        }
        break;
      }

      if (st === 'calm') {
        const calmId = `calm_${Date.now()}`;
        cb.recordAddedNode({ x: lng, y: lat });
        cb.mutateSource({
          type: 'add',
          feature: {
            type: 'Feature',
            geometry: { type: 'Point', coordinates: [lng, lat] },
            properties: { _t: 'calm', _fid: calmId, _seg_id: '', _calm_type: 'traffic_calming' },
          },
        });
        cb.setStatusText(`Traffic calming point at ${lng.toFixed(5)}, ${lat.toFixed(5)}`);
        break;
      }

      // Default: intersection node
      cb.recordAddedNode({ x: lng, y: lat });
      cb.mutateSource({
        type: 'add',
        feature: {
          type: 'Feature',
          geometry: { type: 'Point', coordinates: [lng, lat] },
          properties: { _t: 'node', _node_id: `added_${Date.now()}`, _seg_id: '' },
        },
      });
      cb.setStatusText(`Node added at ${lng.toFixed(5)}, ${lat.toFixed(5)}`);
      break;
    }

    case 'delete_point': {
      const pt = adapter.queryFeatures(tap, ['node', 'ramp', 'calm']);
      if (pt) {
        const t = (pt.properties._t ?? 'node') as string;
        const coords = pt.geometry.type === 'Point' ? pt.geometry.coordinates : [lng, lat];
        if (t === 'node') {
          const nid = (pt.properties._node_id ?? '') as string;
          cb.recordDeletedPoint({ node_id: nid, x: coords[0], y: coords[1] });
          cb.mutateSource({ type: 'delete', matchId: nid, matchKey: '_node_id' });
          cb.setStatusText(`Deleted intersection node ${nid}`);
        } else {
          // ramp or calm — identified by _fid
          const fid = (pt.properties._fid ?? '') as string;
          cb.recordDeletedPoint({ node_id: fid, x: coords[0], y: coords[1] });
          cb.mutateSource({ type: 'delete', matchId: fid, matchKey: '_fid' });
          cb.setStatusText(`Deleted ${t} ${fid}`);
        }
      } else {
        cb.setStatusText('No point found — tap closer to a node, ramp, or calming point');
      }
      break;
    }

    case 'highlight_point': {
      const feat = adapter.queryFeatures(tap, ['node', 'ramp', 'calm', 'street']);
      if (feat) {
        const fid = (feat.properties._fid ?? feat.properties._node_id ?? feat.properties._seg_id ?? '') as string;
        cb.setSelectedFeatureId(fid);
        cb.setStatusText(`Selected: ${fid}`);
      } else {
        cb.setStatusText('No feature found at tap');
      }
      break;
    }

    case 'move_point': {
      if (subtype === 'snap') {
        // Snap subtype: tap a node and snap it to the nearest segment point
        const node = adapter.queryFeatures(tap, ['node']);
        if (!node) {
          cb.setStatusText('No node found — tap closer to a node');
          break;
        }
        const nid = (node.properties._node_id ?? '') as string;
        const nearest = adapter.findNearestSegmentPoint(tap);
        if (!nearest) {
          cb.setStatusText('No nearby segment to snap to');
          break;
        }
        cb.recordMovedEndpoint({
          node_id: nid,
          new_x: nearest.lng,
          new_y: nearest.lat,
          rubber_band_segments: [nearest.segId],
        });
        cb.addEditPreview(node.geometry, 'ghost');
        cb.addEditPreview({ type: 'Point', coordinates: [nearest.lng, nearest.lat] }, 'moved');
        cb.setStatusText(`Snapped node ${nid} to segment ${nearest.segId}`);
      } else {
        // Move subtype (default): two-tap — select node, then tap new position
        if (!session.moveTarget) {
          const node = adapter.queryFeatures(tap, ['node']);
          if (node) {
            const nid = (node.properties._node_id ?? '') as string;
            session.moveTarget = nid;
            cb.setSelectedFeatureId(nid);
            cb.setStatusText(`Node ${nid} selected — tap new position`);
            cb.addEditPreview(node.geometry, 'ghost');
          } else {
            cb.setStatusText('No intersection node found — tap closer');
          }
        } else {
          const nodeId = session.moveTarget;
          cb.recordMovedEndpoint({
            node_id: nodeId,
            new_x: lng,
            new_y: lat,
            rubber_band_segments: [],
          });
          cb.addEditPreview({ type: 'Point', coordinates: [lng, lat] }, 'moved');
          cb.setStatusText(`Moved node ${nodeId} to ${lng.toFixed(5)}, ${lat.toFixed(5)}`);
          cb.setSelectedFeatureId(null);
          session.moveTarget = null;
        }
      }
      break;
    }

    case 'split_segment': {
      const nearest = adapter.findNearestSegmentPoint(tap);
      if (!nearest) {
        cb.setStatusText('No segment found — tap closer to a street');
        break;
      }
      cb.recordAddedNode({ x: nearest.lng, y: nearest.lat });
      cb.addEditPreview({ type: 'Point', coordinates: [nearest.lng, nearest.lat] }, 'node');
      cb.setStatusText(`Split point added on segment ${nearest.segId}`);
      break;
    }

    case 'merge_segments': {
      const seg = adapter.queryFeatures(tap, ['street']);
      if (!seg) {
        cb.setStatusText('No segment found — tap on a street');
        break;
      }
      const segId = (seg.properties._seg_id ?? '') as string;
      if (!session.mergeFirst) {
        session.mergeFirst = segId;
        cb.addEditPreview(seg.geometry, 'ghost');
        cb.setStatusText(`First segment: ${segId} — now tap second segment`);
      } else {
        if (segId === session.mergeFirst) {
          cb.setStatusText('Same segment — tap a different segment');
          break;
        }
        const pairInfo = adapter.getSegmentPairInfo(session.mergeFirst, segId);
        if (!pairInfo || !pairInfo.sharedNodeId) {
          cb.setStatusText(`Segments ${session.mergeFirst} and ${segId} do not share a node`);
          session.mergeFirst = null;
          break;
        }
        // Earlier sequential ID survives
        const id1 = session.mergeFirst;
        const id2 = segId;
        const surviving = id1 < id2 ? id1 : id2;
        const consumed = id1 < id2 ? id2 : id1;
        cb.recordMergedSegments({
          surviving_seg_id: surviving,
          consumed_seg_id: consumed,
          shared_node_id: pairInfo.sharedNodeId,
        });
        cb.addEditPreview(seg.geometry, 'deleted');
        cb.setStatusText(`Merged ${consumed} into ${surviving}`);
        session.mergeFirst = null;
      }
      break;
    }

    case 'draw_segment': {
      if (subtype === 'crosswalk') {
        // Crosswalk subtype: two-tap ramp selection
        const ramp = adapter.queryFeatures(tap, ['ramp']);
        if (!ramp) {
          cb.setStatusText('No curb ramp found — tap closer to a ramp point');
          break;
        }
        const rampInfo = {
          segId: (ramp.properties._seg_id ?? '') as string,
          side: (ramp.properties._side ?? '') as string,
          position: (ramp.properties._position ?? '') as string,
          index: Number(ramp.properties._index ?? 1),
        };
        if (!session.crosswalkFirst) {
          session.crosswalkFirst = rampInfo;
          cb.addEditPreview(ramp.geometry, 'ghost');
          cb.setStatusText(`First ramp selected (${rampInfo.side}/${rampInfo.position}) — tap second ramp`);
        } else {
          const first = session.crosswalkFirst;
          cb.recordDrawnCrosswalk({
            ramp_a: { segment_id: first.segId, side: first.side, position: first.position, index: first.index },
            ramp_b: { segment_id: rampInfo.segId, side: rampInfo.side, position: rampInfo.position, index: rampInfo.index },
          });
          if (ramp.geometry.type === 'Point') {
            cb.addEditPreview(
              { type: 'LineString', coordinates: [ramp.geometry.coordinates, ramp.geometry.coordinates] },
              'added',
            );
          }
          cb.setStatusText('Crosswalk drawn between ramps');
          session.crosswalkFirst = null;
        }
      } else {
        // Default: tap points to draw a line segment, Enter to complete
        session.drawPoints.push([lng, lat]);
        cb.addEditPreview({ type: 'Point', coordinates: [lng, lat] }, 'node');
        if (session.drawPoints.length >= 2) {
          const coords = [...session.drawPoints];
          cb.replaceEditPreviews(
            (et) => et === 'line-preview',
            { geometry: { type: 'LineString', coordinates: coords }, editType: 'line-preview' },
          );
        }
        cb.setStatusText(`${session.drawPoints.length} points — Enter to complete`);
      }
      break;
    }

    case 'add_polygon': {
      session.drawPoints.push([lng, lat]);
      cb.addEditPreview({ type: 'Point', coordinates: [lng, lat] }, 'node');
      if (session.drawPoints.length >= 3) {
        const coords = [...session.drawPoints, session.drawPoints[0]];
        cb.replaceEditPreviews(
          (et) => et === 'polygon-preview',
          { geometry: { type: 'Polygon', coordinates: [coords] }, editType: 'polygon-preview' },
        );
      }
      cb.setStatusText(`${session.drawPoints.length} vertices — Enter to complete`);
      break;
    }

    case 'delete_polygon': {
      const hull = adapter.queryHulls(tap);
      if (hull) {
        const nodeId = (hull.properties.node_id ?? '') as string;
        cb.recordDeletedHull({ node_id: nodeId });
        // Mark the node as non-intersection so hull won't regenerate
        cb.mutateSource({ type: 'delete', matchId: nodeId, matchKey: '_node_id' });
        cb.setStatusText(`Deleted hull at node ${nodeId}`);
      } else {
        cb.setStatusText('No hull polygon found at tap');
      }
      break;
    }

    case 'edit_polygon_face': {
      if (!session.editingHullId) {
        // Select a hull — drag handling is done at the platform level (web: useToolHandler)
        const hull = adapter.queryHulls(tap);
        if (hull) {
          const nodeId = (hull.properties.node_id ?? '') as string;
          const verts = adapter.getHullVertices(nodeId);
          if (verts && verts.length >= 3) {
            session.editingHullId = nodeId;
            session.editingHullVertices = verts.map(v => [...v] as [number, number]);
            cb.setSelectedFeatureId(nodeId);
            // Show vertex handles with index so the platform drag layer can hit-test them
            for (let i = 0; i < verts.length; i++) {
              cb.addEditPreview({ type: 'Point', coordinates: verts[i] }, 'vertex-handle', { _vi: i });
            }
            const ring: [number, number][] = [...verts, verts[0]];
            cb.addEditPreview({ type: 'Polygon', coordinates: [ring] }, 'polygon-preview');
            cb.setStatusText(`Hull ${nodeId} — drag vertices, Enter to confirm`);
          } else {
            cb.setStatusText(`Hull ${nodeId} has no editable vertices`);
          }
        } else {
          cb.setStatusText('No hull polygon found — tap a hull polygon');
        }
      } else {
        // Hull selected — action depends on active subtype
        const verts = session.editingHullVertices;
        if (!verts) break;
        const action = (subtype ?? 'move_vertex') as string;

        const redraw = () => {
          cb.replaceEditPreviews(
            (et) => et === 'vertex-handle' || et === 'polygon-preview',
            null,
          );
          for (let i = 0; i < verts.length; i++) {
            cb.addEditPreview({ type: 'Point', coordinates: verts[i] }, 'vertex-handle', { _vi: i });
          }
          const ring: [number, number][] = [...verts, verts[0]];
          cb.addEditPreview({ type: 'Polygon', coordinates: [ring] }, 'polygon-preview');
        };

        if (action === 'add_ramp') {
          const ramp = adapter.queryFeatures(tap, ['ramp']);
          if (!ramp || ramp.geometry.type !== 'Point') {
            cb.setStatusText('Tap a curb ramp to add it to the hull');
            break;
          }
          const [rx, ry] = ramp.geometry.coordinates as [number, number];
          const insertIdx = findNearestEdgeInsertIndex(verts, rx, ry);
          verts.splice(insertIdx, 0, [rx, ry]);
          redraw();
          cb.setStatusText(`Ramp added (vertex ${insertIdx}) — Enter to confirm`);

        } else if (action === 'delete_vertex') {
          // Find nearest vertex handle to tap
          let bestIdx = -1;
          let bestDist = Infinity;
          for (let i = 0; i < verts.length; i++) {
            const dx = verts[i][0] - lng;
            const dy = verts[i][1] - lat;
            const d = Math.sqrt(dx * dx + dy * dy);
            if (d < bestDist) { bestDist = d; bestIdx = i; }
          }
          if (verts.length <= 3) {
            cb.setStatusText('Hull must have at least 3 vertices');
            break;
          }
          if (bestIdx >= 0 && bestDist < 0.0002) {
            verts.splice(bestIdx, 1);
            redraw();
            cb.setStatusText(`Vertex ${bestIdx} deleted — Enter to confirm`);
          } else {
            cb.setStatusText('Tap closer to a vertex to delete it');
          }

        } else {
          // move_vertex — drag is handled by the platform; click does nothing
          cb.setStatusText('Drag a vertex handle to move it, Enter to confirm');
        }
      }
      break;
    }
  }
}

/**
 * Handle the "finish" gesture for multi-tap tools (Enter on web, button on mobile).
 * Returns true if the gesture was consumed.
 */
export function handleToolFinish(
  tool: ToolId,
  cb: ToolCallbacks,
  session: ToolSessionState,
  subtype?: ToolSubtype | null,
): boolean {
  if (tool === 'draw_segment' && session.drawPoints.length >= 2) {
    const coords = [...session.drawPoints];
    const segId = `drawn_${Date.now()}`;
    const st = subtype ?? 'street';
    // Map subtypes to feature _t values and colors
    const SEGMENT_META: Record<string, { _t: string; _color: string }> = {
      street:      { _t: 'street',    _color: '#c0392b' },
      bikeway:     { _t: 'bikeway',   _color: '#2ed573' },
      sidewalk:    { _t: 'sidewalk',  _color: '#ffa502' },
      crosswalk:   { _t: 'crosswalk', _color: '#ff6b81' },
      curb_return: { _t: 'cret',      _color: '#a55eea' },
    };
    const meta = SEGMENT_META[st] ?? SEGMENT_META.street;
    cb.recordDrawnSegment({ coordinates: coords });
    cb.mutateSource({
      type: 'add',
      feature: {
        type: 'Feature',
        geometry: { type: 'LineString', coordinates: coords },
        properties: { _t: meta._t, _seg_id: segId, _fid: segId, _color: meta._color },
      },
    });
    cb.replaceEditPreviews((et) => et === 'line-preview' || et === 'node', null);
    cb.setStatusText(`${st} drawn with ${coords.length} points`);
    session.drawPoints = [];
    return true;
  }
  if (tool === 'add_polygon' && session.drawPoints.length >= 3) {
    const nodeId = `new_${Date.now()}`;
    const coords: [number, number][] = [...session.drawPoints, session.drawPoints[0]];
    const st = subtype ?? 'hull';
    cb.recordEditedHull({
      node_id: nodeId,
      geometry: { type: 'Polygon', coordinates: [coords] },
    });
    cb.replaceEditPreviews((et) => et === 'polygon-preview' || et === 'node', null);
    cb.setStatusText(`${st} polygon added with ${session.drawPoints.length} vertices`);
    session.drawPoints = [];
    return true;
  }
  if (tool === 'edit_polygon_face' && session.editingHullId && session.editingHullVertices) {
    const verts = session.editingHullVertices;
    const ring: [number, number][] = [...verts, verts[0]];
    cb.recordEditedHull({
      node_id: session.editingHullId,
      geometry: { type: 'Polygon', coordinates: [ring] },
    });
    cb.setStatusText(`Hull ${session.editingHullId} updated`);
    session.editingHullId = null;
    session.editingHullVertices = null;
    session.draggingVertexIndex = null;
    return true;
  }
  return false;
}
