"""Diagnostic: examine how OSMnx represents Shattuck Ave (divided road) in the graph."""
import osmnx as ox
import networkx as nx

# Shattuck Ave area in Berkeley — small bbox around the provided coords
lat, lon = 37 + 52/60 + 16.4/3600, -(122 + 16/60 + 4.8/3600)
G = ox.graph_from_point((lat, lon), dist=400, network_type="all")

# Find edges on Shattuck Ave
shattuck_edges: list[tuple[int, int, int, dict]] = []
for u, v, k, d in G.edges(keys=True, data=True):
    name = d.get("name", "")
    if isinstance(name, list):
        name = " / ".join(str(n) for n in name)
    if "shattuck" in str(name).lower():
        shattuck_edges.append((u, v, k, d))

print(f"Shattuck edges: {len(shattuck_edges)}")
print()

# Group by (u,v) to find parallel keys
from collections import Counter, defaultdict
uv_counts = Counter((u, v) for u, v, k, d in shattuck_edges)
multi_key = {uv: c for uv, c in uv_counts.items() if c > 1}
print(f"Node pairs with multiple keys: {len(multi_key)}")
for (u, v), c in sorted(multi_key.items()):
    print(f"  ({u}, {v}): {c} keys")
print()

# Show node degrees (among ALL edges, not just Shattuck)
shattuck_nodes: set[int] = set()
for u, v, k, d in shattuck_edges:
    shattuck_nodes.add(u)
    shattuck_nodes.add(v)

print(f"Unique Shattuck nodes: {len(shattuck_nodes)}")
print()

# For each Shattuck node, show all outgoing edges (all streets, not just Shattuck)
for nid in sorted(shattuck_nodes):
    out_edges = list(G.out_edges(nid, keys=True, data=True))
    in_edges = list(G.in_edges(nid, keys=True, data=True))
    print(f"Node {nid}:  out_degree={len(out_edges)}  in_degree={len(in_edges)}")
    for u, v, k, d in out_edges:
        name = d.get("name", "?")
        if isinstance(name, list):
            name = " / ".join(str(n) for n in name)
        hw = d.get("highway", "?")
        oneway = d.get("oneway", "?")
        print(f"  -> {v} (k={k}) name={name!r}  hw={hw}  oneway={oneway}")
    print()

# Check if northbound and southbound Shattuck share nodes
print("=== Shared-node analysis ===")
by_oneway: dict[str, set[int]] = defaultdict(set)
for u, v, k, d in shattuck_edges:
    ow = str(d.get("oneway", False))
    by_oneway[ow].update([u, v])

for ow, nodes in sorted(by_oneway.items()):
    print(f"  oneway={ow}: {len(nodes)} nodes")

all_groups = list(by_oneway.values())
if len(all_groups) == 2:
    shared = all_groups[0] & all_groups[1]
    print(f"  Shared nodes between groups: {len(shared)}")
    for nid in sorted(shared):
        print(f"    node {nid}")
