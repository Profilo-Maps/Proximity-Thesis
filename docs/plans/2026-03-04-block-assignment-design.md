# Block Assignment Design

## Pipeline Integration

Block detection runs on the raw NetworkX graph `G` immediately after OSM load, before any GeoDataFrame schema population. This avoids creating a subgraph. The left/right facility swap is deferred until after bikelane/sidewalk tags are populated.

```
1. Load OSM graph -> G, edges_reset, nodes
2. detect_blocks(G, nodes) -> BlockResult
3. Create schema, populate street columns + block results
4. populate_base_bikelanes (tags)
5. populate_base_footlanes (tags)
6. swap_facilities_by_bearing(populated)
7. _populate_separate_facilities
8. Buffering pass
9. Export
```

## Data Structures

```python
@dataclass
class GridCache:
    origin_x: float      # SW corner x (EPSG:32610 metres)
    origin_y: float      # SW corner y
    cell_size: float     # 500.0
    n_cols: int
    n_rows: int

@dataclass
class BlockResult:
    edge_blocks: dict[tuple[int,int,int], tuple[str|None, str|None]]
      # (u, v, key) -> (left_block_id, right_block_id)
    edge_bearings: dict[tuple[int,int,int], float]
      # (u, v, key) -> normalized_bearing (0 deg N compass)
    node_block_faces: dict[int, set[str]]
      # node_id -> set of block IDs touching this node
    block_nodes: set[int]
      # nodes that are vertices of block polygons
    grid: GridCache
```

## Algorithm

### Grid Creation (Step 4)

- Bounding box from all node coordinates (EPSG:32610, metres)
- 500m square cells, origin at SW corner
- Block ID format: `"{col}_{row}_{seq}"`
- Serial traversal (no parallelization or stitching)
- Multi-cell blocks owned by vertex with lowest row, then lowest column

### Left-Turn Traversal (Step 5)

Pre-computation:
- Skip edges where `highway` in `{cycleway, footway, pedestrian, path, steps, corridor, bridleway}`
- For each node, collect outgoing road edges sorted by bearing (CCW)
- `node_out: dict[node_id, list[tuple[bearing, dest_node, edge_key]]]`

For each unvisited directed edge `(u0, v0, k0)`:
1. Initialize `face_edges`, `shoelace = 0.0`, mark visited
2. At each node, find reverse bearing, pick the next edge CCW (leftmost turn)
3. Accumulate shoelace cross-product at each step
4. Terminate on: return to start (cyclic), degree-1 node (terminal), safety guard
5. Classify: `shoelace > 0` = interior (assign block ID), `< 0` = exterior (discard), `~ 0` + terminal = dead-end
6. For each edge `(u, v, k)` in face: face is to LEFT of `u->v`
7. `normalized_bearing` = compass bearing of `u->v` (0 deg = North, clockwise)
8. Add block_id to `node_block_faces` for all nodes in face

### Node Classification (Step 6)

- `is_block_node = True` for nodes touched by at least one interior face
- `is_intersection_node = True` for nodes in 3+ distinct block faces
- Dead-end terminal nodes: block nodes but not intersection nodes

### Facility Swap (Step 6 continued)

Compare `normalized_bearing` vs raw geometry bearing (`coords[0] -> coords[-1]`). If they differ by ~180 deg (+/- 30 deg), swap all `sidewalk_left_* <-> sidewalk_right_*` and `bikeway_left_* <-> bikeway_right_*` columns.

### Block Side Labeling (Step 7)

- Group edges per block face by intersection nodes at each end
- Each group = one block side
- Label = circular mean of bearings, quantized to compass (N, NE, E, SE, S, SW, W, NW)
- Dead-end blocks: left slot gets label by default

## Edge Cases

- **Parallel edges (MultiDiGraph):** Handled via `(u, v, key)` tuples
- **Self-loops:** Skipped (undefined bearing)
- **Disconnected components:** Outer loop over unvisited edges handles naturally
- **Exterior-only edges:** Block ID remains None for edges only in exterior faces

## Diagnostics

Print after detection:
- Interior faces, exterior faces, terminal blocks
- Edges with/without block assignments
- Facility swaps performed
- Grid dimensions
