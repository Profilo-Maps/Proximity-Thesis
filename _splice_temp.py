NEW_BODY = r"""        # ── Build slot zones ─────────────────────────────────────────────────
        # Each arm gets a slot zone: rectangle perpendicular to the arm,
        # clipped to the intersection hull.
        slots: list[tuple[int, str, BaseGeometry, LineString]] = []
        for st_row_s, st_pos_s in connections:
            far_col_s = "end_node_geometry" if st_pos_s == "start" else "start_node_geometry"
            far_geom_s: Any = populated.at[st_row_s, far_col_s] if far_col_s in populated.columns else None
            if not isinstance(far_geom_s, BaseGeometry) or far_geom_s.is_empty:
                continue
            far_pt_s = cast(Point, far_geom_s)
            dx_s = far_pt_s.x - node_pt.x
            dy_s = far_pt_s.y - node_pt.y
            arm_dist_s = (dx_s * dx_s + dy_s * dy_s) ** 0.5
            if arm_dist_s < 1e-6:
                continue
            ux_s, uy_s = dx_s / arm_dist_s, dy_s / arm_dist_s
            px_s, py_s = -uy_s, ux_s  # perpendicular unit vector
            strip_hw_s = default_lane_width_m / 2.0
            r_s = hull_fallback_r
            cx_s, cy_s = node_pt.x, node_pt.y
            rect_s = Polygon([
                (cx_s + ux_s * strip_hw_s + px_s * r_s, cy_s + uy_s * strip_hw_s + py_s * r_s),
                (cx_s + ux_s * strip_hw_s - px_s * r_s, cy_s + uy_s * strip_hw_s - py_s * r_s),
                (cx_s - ux_s * strip_hw_s - px_s * r_s, cy_s - uy_s * strip_hw_s - py_s * r_s),
                (cx_s - ux_s * strip_hw_s + px_s * r_s, cy_s - uy_s * strip_hw_s + py_s * r_s),
            ])
            slot_zone_s = rect_s.intersection(hull)
            if slot_zone_s.is_empty:
                continue
            arm_cl_s = LineString([(node_pt.x, node_pt.y), (far_pt_s.x, far_pt_s.y)])
            slots.append((st_row_s, st_pos_s, slot_zone_s, arm_cl_s))

        if not slots:
            continue

        # ── Sort by candidate count (most evidence first) ─────────────────────
        def _cand_count(slot_item: "tuple[int, str, BaseGeometry, LineString]") -> int:
            _, _, sz, _ = slot_item
            count = 0
            for _qp in street_tree.query(sz, predicate="intersects"):
                if _st_keys[_qp] in _fw_row_set:
                    count += 1
            if sw_tree is not None:
                count += len(sw_tree.query(sz, predicate="intersects"))
            return count

        slots.sort(key=_cand_count, reverse=True)

        # ── Pre-build footway adjacency maps for all candidates at this node ──
        all_fw_cands_node: set[int] = set()
        for _qp2 in street_tree.query(hull, predicate="intersects"):
            _rq2 = _st_keys[_qp2]
            if _rq2 not in _fw_row_set:
                continue
            fw_g_q = populated.geometry[_rq2]
            if not isinstance(fw_g_q, BaseGeometry) or fw_g_q.is_empty:
                continue
            fw_l_q = fw_g_q.length
            if fw_l_q < 1e-3 or fw_l_q > _FOOTWAY_CROSSING_MAX_LEN_M:
                continue
            inter_q = fw_g_q.intersection(hull).length
            if inter_q / fw_l_q >= _FOOTWAY_CROSSING_HULL_FRAC:
                all_fw_cands_node.add(_rq2)

        cand_sn: dict[int, int] = {}
        cand_en: dict[int, int] = {}
        nid_to_cands: dict[int, list[int]] = {}
        for _radj in all_fw_cands_node:
            _sn_r = populated.at[_radj, "start_node_id"]
            _en_r = populated.at[_radj, "end_node_id"]
            if pd.isna(_sn_r) or pd.isna(_en_r):
                continue
            _sni = int(float(str(_sn_r)))
            _eni = int(float(str(_en_r)))
            cand_sn[_radj] = _sni
            cand_en[_radj] = _eni
            nid_to_cands.setdefault(_sni, []).append(_radj)
            nid_to_cands.setdefault(_eni, []).append(_radj)

        # ── _chain_and_merge: BFS + degree-1 chain trace + coord merge ────────
        def _chain_and_merge(
            subset: "list[int]",
        ) -> "list[tuple[LineString, list[tuple[int, bool]]]]":
            subset_set = set(subset)
            vis: set[int] = set()
            result: list[tuple[LineString, list[tuple[int, bool]]]] = []
            for seed in subset:
                if seed in vis or seed not in cand_sn:
                    continue
                comp: list[int] = []
                bq: list[int] = [seed]
                while bq:
                    cur = bq.pop()
                    if cur in vis:
                        continue
                    vis.add(cur)
                    comp.append(cur)
                    for _nid in (cand_sn.get(cur), cand_en.get(cur)):
                        if _nid is None:
                            continue
                        for nb in nid_to_cands.get(_nid, []):
                            if nb in subset_set and nb not in vis:
                                bq.append(nb)
                nd: dict[int, int] = {}
                for r2 in comp:
                    for _nid in (cand_sn.get(r2), cand_en.get(r2)):
                        if _nid is not None:
                            nd[_nid] = nd.get(_nid, 0) + 1
                start_nid = next(
                    (n for n, d in nd.items() if d == 1), cand_sn.get(comp[0])
                )
                chain: list[tuple[int, bool]] = []
                chain_vis: set[int] = set()
                cur_nid = start_nid
                comp_set = set(comp)
                while cur_nid is not None:
                    found = False
                    for r2 in nid_to_cands.get(cur_nid, []):
                        if r2 not in comp_set or r2 in chain_vis:
                            continue
                        chain_vis.add(r2)
                        is_fwd = cand_sn.get(r2) == cur_nid
                        chain.append((r2, is_fwd))
                        cur_nid = cand_en[r2] if is_fwd else cand_sn[r2]
                        found = True
                        break
                    if not found:
                        break
                if not chain:
                    chain = [(comp[0], True)]
                coords: list[tuple[float, float]] = []
                for r2, is_fwd in chain:
                    g2 = populated.geometry[r2]
                    if not isinstance(g2, BaseGeometry):
                        continue
                    c2 = _flatten_coords(g2)
                    if not is_fwd:
                        c2 = list(reversed(c2))
                    if coords and c2:
                        if (abs(coords[-1][0] - c2[0][0]) < 1e-4
                                and abs(coords[-1][1] - c2[0][1]) < 1e-4):
                            c2 = c2[1:]
                    coords.extend(c2)
                if len(coords) >= 2:
                    result.append((LineString(coords), chain))
            return result

        # ── Process each slot ─────────────────────────────────────────────────
        for st_row, st_pos, slot_zone, arm_cl in slots:
            _slot_zones[(st_row, st_pos)] = slot_zone
            if (st_row, st_pos) in _assigned:
                continue

            st_geom = populated.geometry[st_row]
            if not isinstance(st_geom, BaseGeometry) or st_geom.is_empty:
                continue

            cw_geom_result: "BaseGeometry | None" = None
            cw_source_result: "str | None" = None

            # ── Case A: footway chains ────────────────────────────────────────
            slot_fw_cands: list[int] = []
            for _pa in street_tree.query(slot_zone, predicate="intersects"):
                _ra = _st_keys[_pa]
                if _ra not in all_fw_cands_node or _ra in _absorbed_footway_rows:
                    continue
                fw_g_a = populated.geometry[_ra]
                if isinstance(fw_g_a, BaseGeometry) and fw_g_a.intersects(slot_zone):
                    slot_fw_cands.append(_ra)

            crossing_fw: list[int] = []
            curb_return_fw: list[int] = []
            for _rcls in slot_fw_cands:
                _ic_cls = populated.geometry[_rcls].intersection(arm_cl)
                if isinstance(_ic_cls, Point) and not _ic_cls.is_empty:
                    crossing_fw.append(_rcls)
                else:
                    curb_return_fw.append(_rcls)

            # Curb returns: store in curb_return_* columns on nearest arm row
            for cr_geom_a, cr_chain_a in _chain_and_merge(curb_return_fw):
                cr_mid = cr_geom_a.centroid
                best_r_cr: "int | None" = None
                best_pos_cr: "str | None" = None
                best_d_cr = float("inf")
                for _str3 in local_street_rows_node:
                    _stp3 = _street_end_for_node(_str3, node_key)
                    if _stp3 is None:
                        continue
                    _stc3 = _flatten_coords(populated.geometry[_str3])
                    if not _stc3:
                        continue
                    _ep3 = Point(_stc3[0] if _stp3 == "start" else _stc3[-1])
                    _d3 = cr_mid.distance(_ep3)
                    if _d3 < best_d_cr:
                        best_d_cr, best_r_cr, best_pos_cr = _d3, _str3, _stp3
                if best_r_cr is not None and best_pos_cr is not None:
                    for _sncr in ("1", "2"):
                        _cr_col = f"curb_return_{best_pos_cr}_{_sncr}_geometry"
                        if _cr_col in populated.columns and pd.isna(populated.at[best_r_cr, _cr_col]):
                            populated.at[best_r_cr, _cr_col] = cr_geom_a  # type: ignore[index]
                            break
                for r2_cr, _ in cr_chain_a:
                    _absorbed_footway_rows.add(r2_cr)

            # Case A crossing: accept chain if merged LineString crosses arm_cl
            for merged_geom_a, chain_a in _chain_and_merge(crossing_fw):
                _ver_a = merged_geom_a.intersection(arm_cl)
                if _ver_a.is_empty or not isinstance(_ver_a, (Point, MultiPoint)):
                    continue
                cw_geom_result = merged_geom_a
                cw_source_result = "case_a"
                for r2_a, _ in chain_a:
                    _absorbed_footway_rows.add(r2_a)
                n_case_a += 1
                break

            if cw_geom_result is not None:
                populated.at[st_row, f"crosswalk_{st_pos}_geometry"] = cw_geom_result  # type: ignore[index]
                populated.at[st_row, f"crosswalk_{st_pos}_source"] = cw_source_result  # type: ignore[index]
                _assigned.add((st_row, st_pos))
                continue

            # ── Case B: sidewalk crossing ──────────────────────────────────────
            if sw_tree is not None:
                nearby_sw_b = [all_sw_keys[i] for i in sw_tree.query(slot_zone, predicate="intersects")]
                _found_b = False
                for sw_row, sw_side in nearby_sw_b:
                    if _found_b:
                        break
                    if sw_row == st_row:
                        continue
                    sw_col = f"sidewalk_{sw_side}_geometry"
                    sw_geom = populated.at[sw_row, sw_col]
                    if not isinstance(sw_geom, BaseGeometry) or sw_geom.is_empty:
                        continue
                    inter_b = sw_geom.intersection(arm_cl)
                    cross_pts_b: list[Point] = []
                    if isinstance(inter_b, Point) and not inter_b.is_empty:
                        cross_pts_b = [inter_b]
                    elif isinstance(inter_b, (MultiPoint, GeometryCollection)):
                        cross_pts_b = [p for p in inter_b.geoms if isinstance(p, Point)]
                    if not cross_pts_b:
                        continue
                    for cross_pt in cross_pts_b:
                        if not hull.covers(cross_pt):
                            continue
                        sw_coords_b = _flatten_coords(sw_geom)
                        if len(sw_coords_b) < 2:
                            continue
                        sw_line = sw_geom if isinstance(sw_geom, LineString) else LineString(sw_coords_b)
                        cross_dist = sw_line.project(cross_pt)
                        sw_len = sw_line.length

                        ramp_before: "tuple[float, Point] | None" = None
                        ramp_after: "tuple[float, Point] | None" = None

                        for ramp_pos in ("start", "end"):
                            ramp_col = f"sidewalk_{sw_side}_curbramp_{ramp_pos}_1_geometry"
                            ramp_val = populated.at[sw_row, ramp_col]
                            if not isinstance(ramp_val, Point):
                                continue
                            if not hull.covers(ramp_val):
                                continue
                            ramp_dist = sw_line.project(ramp_val)
                            if ramp_dist <= cross_dist:
                                if ramp_before is None or ramp_dist > ramp_before[0]:
                                    ramp_before = (ramp_dist, ramp_val)
                            else:
                                if ramp_after is None or ramp_dist < ramp_after[0]:
                                    ramp_after = (ramp_dist, ramp_val)

                        # Spanning sidewalk fallback (gated by slot zone midpoint)
                        if ramp_before is None and ramp_after is None:
                            _span_coords = _flatten_coords(sw_geom)
                            if (len(_span_coords) >= 2
                                    and _side_of_street(st_geom, Point(_span_coords[0]))
                                        != _side_of_street(st_geom, Point(_span_coords[-1]))):
                                for _rp_sp in ("start", "end"):
                                    _rc_sp = f"sidewalk_{sw_side}_curbramp_{_rp_sp}_1_geometry"
                                    _rv_sp = populated.at[sw_row, _rc_sp]
                                    if not isinstance(_rv_sp, Point):
                                        continue
                                    _rd_sp = sw_line.project(_rv_sp)
                                    if _rd_sp <= cross_dist:
                                        if ramp_before is None or _rd_sp > ramp_before[0]:
                                            ramp_before = (_rd_sp, _rv_sp)
                                    else:
                                        if ramp_after is None or _rd_sp < ramp_after[0]:
                                            ramp_after = (_rd_sp, _rv_sp)
                                if ramp_before is not None and ramp_after is None:
                                    ramp_after = (sw_len, Point(_span_coords[-1]))
                                elif ramp_after is not None and ramp_before is None:
                                    ramp_before = (0.0, Point(_span_coords[0]))

                        # Both ramps found — extract sub-segment
                        if ramp_before is not None and ramp_after is not None:
                            cw_cand_b = _sw_substring(sw_line, ramp_before[0], ramp_after[0])
                            if cw_cand_b.is_empty or cw_cand_b.length < 1e-3:
                                continue
                            # Gate: midpoint must be inside slot zone
                            if not slot_zone.covers(cw_cand_b.centroid):
                                continue
                            cw_geom_result = cw_cand_b
                            cw_source_result = "case_b"
                            if ramp_after[0] < sw_len - 1e-3:
                                trimmed = _sw_substring(sw_line, ramp_after[0], sw_len)
                                if not trimmed.is_empty:
                                    populated.at[sw_row, sw_col] = trimmed  # type: ignore[index]
                            elif ramp_before[0] > 1e-3:
                                trimmed = _sw_substring(sw_line, 0, ramp_before[0])
                                if not trimmed.is_empty:
                                    populated.at[sw_row, sw_col] = trimmed  # type: ignore[index]
                            n_case_b += 1
                            _found_b = True
                            break

                        # One ramp — attempt split
                        existing_ramp = ramp_before if ramp_before is not None else ramp_after
                        if existing_ramp is None:
                            continue

                        split_pt: "Point | None" = None
                        for sw_row_j, sw_side_j in nearby_sw_b:
                            if sw_row_j == sw_row and sw_side_j == sw_side:
                                continue
                            sw_geom_j = populated.at[sw_row_j, f"sidewalk_{sw_side_j}_geometry"]
                            if not isinstance(sw_geom_j, BaseGeometry) or sw_geom_j.is_empty:
                                continue
                            ij = sw_geom.intersection(sw_geom_j)
                            if ij.is_empty:
                                continue
                            ij_pts: list[Point] = []
                            if isinstance(ij, Point):
                                ij_pts = [ij]
                            elif isinstance(ij, (MultiPoint, GeometryCollection)):
                                ij_pts = [p for p in ij.geoms if isinstance(p, Point)]
                            for ip in ij_pts:
                                if not hull.covers(ip):
                                    continue
                                ip_dist = sw_line.project(ip)
                                if ramp_before is not None and ip_dist > cross_dist:
                                    split_pt = ip
                                    break
                                elif ramp_after is not None and ip_dist < cross_dist:
                                    split_pt = ip
                                    break
                            if split_pt is not None:
                                break

                        if split_pt is None:
                            for sw_row_k, sw_side_k in nearby_sw_b:
                                if sw_side_k == sw_side:
                                    continue
                                opp_geom = populated.at[sw_row_k, f"sidewalk_{sw_side_k}_geometry"]
                                if not isinstance(opp_geom, BaseGeometry) or opp_geom.is_empty:
                                    continue
                                opp_coords_k = _flatten_coords(opp_geom)
                                if not opp_coords_k:
                                    continue
                                opp_mid = Point(opp_coords_k[len(opp_coords_k) // 2])
                                if _side_of_street(st_geom, opp_mid) == _side_of_street(st_geom, cross_pt):
                                    continue
                                for ramp_pos_k in ("start", "end"):
                                    rk_col = f"sidewalk_{sw_side_k}_curbramp_{ramp_pos_k}_1_geometry"
                                    rk_val = populated.at[sw_row_k, rk_col]
                                    if not isinstance(rk_val, Point):
                                        continue
                                    if not hull.covers(rk_val):
                                        continue
                                    proj_dist = sw_line.project(rk_val)
                                    proj_pt = sw_line.interpolate(proj_dist)
                                    if ramp_before is not None and proj_dist > cross_dist:
                                        split_pt = proj_pt
                                        break
                                    elif ramp_after is not None and proj_dist < cross_dist:
                                        split_pt = proj_pt
                                        break
                                if split_pt is not None:
                                    break

                        if split_pt is None:
                            continue

                        split_dist = sw_line.project(split_pt)
                        new_ramp_pos = "end" if ramp_before is not None else "start"
                        new_ramp_col = f"sidewalk_{sw_side}_curbramp_{new_ramp_pos}_1_geometry"
                        if pd.isna(populated.at[sw_row, new_ramp_col]):
                            populated.at[sw_row, new_ramp_col] = split_pt  # type: ignore[index]
                            n_new_ramps += 1

                        cw_start_d = ramp_before[0] if ramp_before is not None else split_dist
                        cw_end_d = split_dist if ramp_before is not None else existing_ramp[0]
                        cw_cand_split = _sw_substring(sw_line, cw_start_d, cw_end_d)
                        if cw_cand_split.is_empty or cw_cand_split.length < 1e-3:
                            continue
                        if not slot_zone.covers(cw_cand_split.centroid):
                            continue
                        cw_geom_result = cw_cand_split
                        cw_source_result = "case_b"
                        if ramp_before is not None and split_dist < sw_len - 1e-3:
                            trimmed = _sw_substring(sw_line, split_dist, sw_len)
                            if not trimmed.is_empty:
                                populated.at[sw_row, sw_col] = trimmed  # type: ignore[index]
                        elif ramp_after is not None and split_dist > 1e-3:
                            trimmed = _sw_substring(sw_line, 0, split_dist)
                            if not trimmed.is_empty:
                                populated.at[sw_row, sw_col] = trimmed  # type: ignore[index]
                        n_case_b += 1
                        n_splits += 1
                        _found_b = True
                        break

            if cw_geom_result is not None:
                populated.at[st_row, f"crosswalk_{st_pos}_geometry"] = cw_geom_result  # type: ignore[index]
                populated.at[st_row, f"crosswalk_{st_pos}_source"] = cw_source_result  # type: ignore[index]
                _assigned.add((st_row, st_pos))
                continue

            # ── Case C: ramp pair within slot zone (exclude st_row) ───────────
            c_left: list[Point] = []
            c_right: list[Point] = []
            for c_row, _ in connections:
                if c_row == st_row:
                    continue
                for c_side in ("left", "right"):
                    for c_rp in ("start", "end"):
                        c_col = f"sidewalk_{c_side}_curbramp_{c_rp}_1_geometry"
                        c_val = populated.at[c_row, c_col]
                        if not isinstance(c_val, Point):
                            continue
                        if not slot_zone.covers(c_val):
                            continue
                        which_c = _side_of_street(st_geom, c_val)
                        (c_left if which_c == "left" else c_right).append(c_val)

            if c_left and c_right:
                c_l_pt = c_left[0]
                c_r_pt = min(c_right, key=lambda p: c_l_pt.distance(p))
                c_line = LineString([(c_l_pt.x, c_l_pt.y), (c_r_pt.x, c_r_pt.y)])
                _ic_c = c_line.intersection(arm_cl)
                if isinstance(_ic_c, Point) and not _ic_c.is_empty:
                    cw_geom_result = c_line
                    cw_source_result = "case_c"
                    n_case_c += 1

            if cw_geom_result is not None:
                populated.at[st_row, f"crosswalk_{st_pos}_geometry"] = cw_geom_result  # type: ignore[index]
                populated.at[st_row, f"crosswalk_{st_pos}_source"] = cw_source_result  # type: ignore[index]
                _assigned.add((st_row, st_pos))
                continue

            # ── Tier 1: ramp pair from all rows at this node ──────────────────
            t1_left: list[Point] = []
            t1_right: list[Point] = []
            for t1_row, _ in connections:
                for t1_side in ("left", "right"):
                    for t1_rp in ("start", "end"):
                        t1_col = f"sidewalk_{t1_side}_curbramp_{t1_rp}_1_geometry"
                        t1_val = populated.at[t1_row, t1_col]
                        if not isinstance(t1_val, Point):
                            continue
                        if not slot_zone.covers(t1_val):
                            continue
                        which_t1 = _side_of_street(st_geom, t1_val)
                        (t1_left if which_t1 == "left" else t1_right).append(t1_val)

            if t1_left and t1_right:
                t1_l = t1_left[0]
                t1_r = min(t1_right, key=lambda p: t1_l.distance(p))
                t1_line = LineString([(t1_l.x, t1_l.y), (t1_r.x, t1_r.y)])
                _ic_t1 = t1_line.intersection(arm_cl)
                if isinstance(_ic_t1, Point) and not _ic_t1.is_empty:
                    cw_geom_result = t1_line
                    cw_source_result = "ramp_pair"
                    n_ramp_pair += 1

            if cw_geom_result is not None:
                populated.at[st_row, f"crosswalk_{st_pos}_geometry"] = cw_geom_result  # type: ignore[index]
                populated.at[st_row, f"crosswalk_{st_pos}_source"] = cw_source_result  # type: ignore[index]
                _assigned.add((st_row, st_pos))
                continue

            # ── Tier 2: sidewalk endpoint pair (promote to curb ramps) ─────────
            if sw_tree is not None:
                t2_left: list[tuple[Point, int, str, bool]] = []
                t2_right: list[tuple[Point, int, str, bool]] = []
                for t2_sw_idx in sw_tree.query(slot_zone, predicate="intersects"):
                    t2_sw_row, t2_sw_side = all_sw_keys[t2_sw_idx]
                    t2_sw_col = f"sidewalk_{t2_sw_side}_geometry"
                    t2_sw_geom = populated.at[t2_sw_row, t2_sw_col]
                    if not isinstance(t2_sw_geom, BaseGeometry) or t2_sw_geom.is_empty:
                        continue
                    t2_coords = _flatten_coords(t2_sw_geom)
                    if len(t2_coords) < 2:
                        continue
                    for t2_is_start, t2_ep in ((True, t2_coords[0]), (False, t2_coords[-1])):
                        t2_ep_pt = Point(t2_ep)
                        if not slot_zone.covers(t2_ep_pt):
                            continue
                        which_t2 = _side_of_street(st_geom, t2_ep_pt)
                        t2_item = (t2_ep_pt, t2_sw_row, t2_sw_side, t2_is_start)
                        (t2_left if which_t2 == "left" else t2_right).append(t2_item)

                if t2_left and t2_right:
                    t2_l_item = t2_left[0]
                    t2_l_pt = t2_l_item[0]
                    t2_r_item = min(t2_right, key=lambda x: t2_l_pt.distance(x[0]))
                    t2_r_pt = t2_r_item[0]
                    t2_line = LineString([(t2_l_pt.x, t2_l_pt.y), (t2_r_pt.x, t2_r_pt.y)])
                    _ic_t2 = t2_line.intersection(arm_cl)
                    if isinstance(_ic_t2, Point) and not _ic_t2.is_empty:
                        cw_geom_result = t2_line
                        cw_source_result = "sw_endpoints"
                        n_sw_endpoints += 1
                        for t2_ep_pt_p, t2_sw_row_p, t2_sw_side_p, t2_is_start_p in [t2_l_item, t2_r_item]:
                            t2_ep_pos = "start" if t2_is_start_p else "end"
                            t2_ep_col = f"sidewalk_{t2_sw_side_p}_curbramp_{t2_ep_pos}_1_geometry"
                            if t2_ep_col in populated.columns and pd.isna(populated.at[t2_sw_row_p, t2_ep_col]):
                                populated.at[t2_sw_row_p, t2_ep_col] = t2_ep_pt_p  # type: ignore[index]
                                t2_syn_col = f"sidewalk_{t2_sw_side_p}_curbramp_{t2_ep_pos}_1_synthesized"
                                if t2_syn_col in populated.columns:
                                    populated.at[t2_sw_row_p, t2_syn_col] = True  # type: ignore[index]
                                n_new_ramps += 1
                        populated.at[st_row, f"crosswalk_{st_pos}_geometry"] = cw_geom_result  # type: ignore[index]
                        populated.at[st_row, f"crosswalk_{st_pos}_source"] = cw_source_result  # type: ignore[index]
                        _assigned.add((st_row, st_pos))

        # ── Per-node stub cleanup ─────────────────────────────────────────────
        for st_row_sc, _ in connections:
            for side_sc in ("left", "right"):
                _sw_sc = f"sidewalk_{side_sc}_geometry"
                if _sw_sc not in populated.columns:
                    continue
                sw_g_sc = populated.at[st_row_sc, _sw_sc]
                if not isinstance(sw_g_sc, BaseGeometry) or sw_g_sc.is_empty or sw_g_sc.length < 1e-3:
                    continue
                sw_frac_sc = sw_g_sc.intersection(hull).length / sw_g_sc.length
                if sw_frac_sc > _SW_STUB_HULL_FRAC:
                    populated.at[st_row_sc, _sw_sc] = pd.NA  # type: ignore[index]
                    for _rp_sc in ("start", "end"):
                        for _sl_sc in ("1", "2", "3"):
                            _rc_sc = f"sidewalk_{side_sc}_curbramp_{_rp_sc}_{_sl_sc}_geometry"
                            if _rc_sc in populated.columns:
                                populated.at[st_row_sc, _rc_sc] = pd.NA  # type: ignore[index]
                else:
                    for _rp_sc in ("start", "end"):
                        for _sl_sc in ("1", "2", "3"):
                            _rc_sc = f"sidewalk_{side_sc}_curbramp_{_rp_sc}_{_sl_sc}_geometry"
                            if _rc_sc not in populated.columns:
                                continue
                            rv_sc = populated.at[st_row_sc, _rc_sc]
                            if isinstance(rv_sc, Point) and hull.covers(rv_sc):
                                populated.at[st_row_sc, _rc_sc] = pd.NA  # type: ignore[index]

    # ── Null absorbed footway rows ────────────────────────────────────────────
    for _ar in _absorbed_footway_rows:
        for side_a in ("left", "right"):
            _sw_a = f"sidewalk_{side_a}_geometry"
            if _sw_a in populated.columns:
                populated.at[_ar, _sw_a] = pd.NA  # type: ignore[index]
            for _rp in ("start", "end"):
                for _sl in ("1", "2", "3"):
                    _rc_a = f"sidewalk_{side_a}_curbramp_{_rp}_{_sl}_geometry"
                    if _rc_a in populated.columns:
                        populated.at[_ar, _rc_a] = pd.NA  # type: ignore[index]

    # ── Validation pass ───────────────────────────────────────────────────────
    n_out_of_zone = 0
    n_suspect = 0
    for (val_row, val_pos), val_zone in _slot_zones.items():
        cw_val = populated.at[val_row, f"crosswalk_{val_pos}_geometry"]
        if not isinstance(cw_val, BaseGeometry) or cw_val.is_empty:
            continue
        qual = "ok"
        if not val_zone.covers(cw_val.centroid):
            qual = "out_of_zone"
            n_out_of_zone += 1
            st_name_v = populated.at[val_row, "name"] if "name" in populated.columns else str(val_row)
            print(f"  WARNING crosswalk out-of-zone: row {val_row} ({st_name_v}) pos={val_pos}")
        elif cw_val.length > 2.0 * hull_fallback_r:
            qual = "suspect_length"
            n_suspect += 1
        populated.at[val_row, f"crosswalk_{val_pos}_quality"] = qual  # type: ignore[index]

    print(
        f"Crosswalk geometries: {n_case_a} Case A (footway chains), "
        f"{n_case_b} Case B (sidewalk-street), {n_case_c} Case C (ramp pairs), "
        f"{n_ramp_pair} Tier-1 (ramp pair), {n_sw_endpoints} Tier-2 (sw endpoints), "
        f"{n_splits} sidewalk splits, {n_new_ramps} new curb ramps, "
        f"{n_out_of_zone} out-of-zone, {n_suspect} suspect-length"
    )
    return populated
"""

with open('c:/Dev/Proximity/Implementations/ProximityModel.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

start_line = None
end_line = None
for i, line in enumerate(lines):
    if start_line is None and '        # -- Find crossing candidates: footway rows in hull, short enough' in line:
        start_line = i
    if start_line is not None and line.strip() == 'return populated' and i > start_line:
        end_line = i
        break

assert start_line is not None and end_line is not None

new_lines = lines[:start_line] + [NEW_BODY] + lines[end_line + 1:]

with open('c:/Dev/Proximity/Implementations/ProximityModel.py', 'w', encoding='utf-8') as f:
    f.writelines(new_lines)

print(f"Done: replaced lines {start_line+1}-{end_line+1}")
