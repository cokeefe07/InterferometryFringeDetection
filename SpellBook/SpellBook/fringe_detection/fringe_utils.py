import cv2
import numpy as np
from skimage.morphology import skeletonize, remove_small_objects, remove_small_holes


def binarize(gray, method="Otsu", thresh=128, invert=False, blur=0):
    g = gray
    if blur > 0:
        k = int(2 * round(blur / 2) + 1)
        g = cv2.GaussianBlur(g, (k, k), 0)
    if method == "Otsu":
        _, bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif method == "Adaptive":
        bksz = int(thresh) | 1
        bksz = max(3, min(151, bksz))
        bw = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, bksz, 2)
    else:
        _, bw = cv2.threshold(g, int(thresh), 255, cv2.THRESH_BINARY)
    if invert:
        bw = 255 - bw
    return bw


def line_kernel(length, thickness=1, angle_deg=0):
    L = max(3, int(length)); T = max(1, int(thickness))
    size = int(np.ceil(np.sqrt(2) * L)) + 4
    canv = np.zeros((size, size), np.uint8); c = size // 2
    cv2.line(canv, (c - L // 2, c), (c + L // 2, c), 255, T)
    M = cv2.getRotationMatrix2D((c, c), angle_deg, 1.0)
    rot = cv2.warpAffine(canv, M, (size, size), flags=cv2.INTER_NEAREST, borderValue=0)
    rot = (rot > 0).astype(np.uint8)
    return rot


def oriented_opening(bw01, length, thickness, max_angle=8.0, step=2.0):
    angles = np.arange(-float(max_angle), float(max_angle) + 1e-6, float(step))
    out = np.zeros_like(bw01, np.uint8)
    for a in angles:
        k = line_kernel(length, thickness, a)
        er = cv2.erode(bw01, k, iterations=1)
        op = cv2.dilate(er, k, iterations=1)
        out = np.maximum(out, op)
    return out


def overlay_mask_on_gray(gray, mask01, line_alpha=0.85, bg_fade=0.0, bg_to='white', line_color=(255, 0, 0)):
    """Overlay a 0/1 mask onto a grayscale image.
    Accepts gray as uint8 (or other types, normalized to uint8) and mask01 as 0/1 (bool or uint8).
    Handles minor size mismatches by resizing mask.
    """
    # Normalize grayscale to uint8
    g = gray
    if g.ndim == 3:
        # Convert color to gray if needed
        g = cv2.cvtColor(g, cv2.COLOR_BGR2GRAY)
    if g.dtype != np.uint8:
        g = cv2.normalize(g, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)
    base = g.astype(np.float32)
    target = 255.0 if bg_to == 'white' else 0.0
    base = (1.0 - bg_fade) * base + bg_fade * target
    base = np.clip(base, 0, 255).astype(np.uint8)
    base_bgr = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    color = np.zeros_like(base_bgr)
    # line_color provided as RGB; convert to BGR for OpenCV
    color[..., 0] = line_color[2]
    color[..., 1] = line_color[1]
    color[..., 2] = line_color[0]
    m = mask01
    if m.dtype != np.uint8 and m.dtype != np.bool_:
        m = (m > 0).astype(np.uint8)
    if m.dtype != np.bool_:
        m = m.astype(bool)
    # Resize mask to base size if needed
    if m.shape != base.shape[:2]:
        m = cv2.resize(m.astype(np.uint8), (base.shape[1], base.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
    out = base_bgr.copy()
    out[m] = (line_alpha * color[m] + (1.0 - line_alpha) * base_bgr[m]).astype(np.uint8)
    return out


def remove_humps(skeleton, max_width=2):
    """
    Removes small vertical deviations (humps) in horizontal-ish lines.
    Detects 1-pixel up to max_width pixel wide vertical jumps and flattens them.
    skeleton: uint8 image (0/255 or 0/1)
    max_width: maximum width of hump to remove (default 2)
    Returns: uint8 image (0/255)
    """
    # Ensure binary 0-1
    skel = (skeleton > 0).astype(np.uint8)
    
    # We'll create a copy to modify
    out = skel.copy()
    
    # Iterate through widths from 1 to max_width
    for w in range(1, max_width + 1):
        # Update skel from previous pass
        skel = out.copy()
        
        # --- Up Hump (width w) ---
        # Pattern:
        #   Row y:   1 ... 1 (w times)
        #   Row y+1: 1 0 ... 0 1 (0s are w times)
        # Indices:
        #   C (Current row): skel[0:-1, 1 : -(w)] (if w=1, 1:-1. if w=2, 1:-2)
        #   BL (Below Left): skel[1:, 0 : -(w+1)]
        #   BR (Below Right): skel[1:, w+1 :]
        #   BC (Below Center): skel[1:, 1 : -(w)]
        
        # Construct slices
        # Note: python slice upper bound is exclusive.
        # If w=1: 1:-1. If w=2: 1:-2. Generally: 1 : -(w)
        # But if -(w) is 0 (e.g. w=0, impossible here), it's empty.
        # If w is large, we need to be careful.
        
        # Slice for "Center" (the hump top)
        # We need to check if ALL pixels in the width w are 1.
        # We can just check the slice. But for vectorized check, we need to check w columns.
        # Instead of complex convolution, let's just implement explicit checks for small w
        # since the user likely only cares about small widths (1-5).
        # Generalizing with loop over columns for the check:
        
        # Base slices
        # y range: 0 to H-2 (for top row of pattern)
        # x range: 1 to W-2-w+1 ?
        # Let's stick to the explicit slicing logic which is robust.
        
        # Define the valid range for x where a hump of width w can start.
        # It needs 1 pixel left context and 1 pixel right context.
        # So x starts at 1.
        # x ends such that x+w < W-1. => x < W-1-w.
        # So x goes up to W-2-w.
        # Slicing: [1 : -1-(w-1)] = [1 : -w]
        
        # Check if Top (C) is all 1s
        # We can check skel[0:-1, 1:-w] & skel[0:-1, 2:-w+1] ... & skel[0:-1, w:-1] ?
        # Let's build the mask for "Top is all 1s"
        top_mask = skel[0:-1, 1:-w] == 1
        for k in range(1, w):
            # Shifted slices
            # If k=1, we look at skel[0:-1, 2 : -w+1]
            # If w=2, k=1. 1:-2 vs 2:-1. Correct.
            # Slice end: -w+k. If -w+k is 0, it means end.
            end_idx = -w+k if (-w+k) < 0 else None
            top_mask = top_mask & (skel[0:-1, 1+k : end_idx] == 1)
            
        # Check if Bottom Center (BC) is all 0s
        bot_mask = skel[1:, 1:-w] == 0
        for k in range(1, w):
            end_idx = -w+k if (-w+k) < 0 else None
            bot_mask = bot_mask & (skel[1:, 1+k : end_idx] == 0)
            
        # Check Bottom Left (BL) and Bottom Right (BR)
        # BL: skel[1:, 0 : -(w+1)]
        # BR: skel[1:, w+1 :]
        # These are single pixels flanking the gap.
        BL = skel[1:, 0 : -(w+1)]
        BR = skel[1:, w+1 :]
        
        mask_up = top_mask & bot_mask & (BL == 1) & (BR == 1)
        
        # Apply changes for Up Hump
        # Set Top to 0
        for k in range(w):
            end_idx = -w+k if (-w+k) < 0 else None
            # We need to assign to the slice corresponding to mask_up
            # out[0:-1, 1+k : end_idx][mask_up] = 0
            # But we can't do boolean indexing on a slice if we want to assign to the original array's slice?
            # Yes we can: arr[slice][mask] = val
            
            # Re-derive the slice for the k-th pixel of the hump
            # The mask `mask_up` corresponds to the "start" position of the hump.
            # The k-th pixel is at offset k from the start.
            # So we apply the mask to the slice starting at 1+k.
            
            # Slice for the k-th pixel column in the top row
            s_top = out[0:-1, 1+k : (-w+k) if (-w+k) < 0 else None]
            s_top[mask_up] = 0
            
            # Set Bottom to 1
            s_bot = out[1:, 1+k : (-w+k) if (-w+k) < 0 else None]
            s_bot[mask_up] = 1

        # --- Down Hump (width w) ---
        # Pattern:
        #   Row y-1: 1 0 ... 0 1
        #   Row y:   1 ... 1
        # Indices:
        #   C (Current row): skel[1:, 1:-w]
        #   TL (Top Left): skel[0:-1, 0:-(w+1)]
        #   TR (Top Right): skel[0:-1, w+1:]
        #   TC (Top Center): skel[0:-1, 1:-w]
        
        # Check if Bottom (C) is all 1s
        bot_mask_d = skel[1:, 1:-w] == 1
        for k in range(1, w):
            end_idx = -w+k if (-w+k) < 0 else None
            bot_mask_d = bot_mask_d & (skel[1:, 1+k : end_idx] == 1)
            
        # Check if Top Center (TC) is all 0s
        top_mask_d = skel[0:-1, 1:-w] == 0
        for k in range(1, w):
            end_idx = -w+k if (-w+k) < 0 else None
            top_mask_d = top_mask_d & (skel[0:-1, 1+k : end_idx] == 0)
            
        TL = skel[0:-1, 0 : -(w+1)]
        TR = skel[0:-1, w+1 :]
        
        mask_down = bot_mask_d & top_mask_d & (TL == 1) & (TR == 1)
        
        # Apply changes for Down Hump
        for k in range(w):
            end_idx = -w+k if (-w+k) < 0 else None
            
            # Set Bottom to 0
            s_bot = out[1:, 1+k : end_idx]
            s_bot[mask_down] = 0
            
            # Set Top to 1
            s_top = out[0:-1, 1+k : end_idx]
            s_top[mask_down] = 1
            
    return (out * 255).astype(np.uint8)


def remove_branches(skeleton, max_length=10):
    """
    Removes branches (spurs) shorter than max_length from the skeleton.
    Only removes branches connected to a junction (pixels with >2 neighbors).
    Isolated lines are preserved.
    skeleton: uint8 image (0/255)
    max_length: maximum length of branch to remove
    """
    skel = (skeleton > 0).astype(np.uint8)
    h, w = skel.shape
    
    # Kernel for neighbor count
    kernel = np.array([[1,1,1],[1,0,1],[1,1,1]], dtype=np.uint8)
    
    # Find endpoints
    neighbors = cv2.filter2D(skel, -1, kernel, borderType=cv2.BORDER_CONSTANT)
    endpoints = np.argwhere((neighbors == 1) & (skel == 1))
    
    # Mask of pixels to remove
    to_remove = np.zeros_like(skel, dtype=bool)
    
    # Offsets for 8 neighbors
    offsets = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
    
    for r_start, c_start in endpoints:
        # Trace path
        path = [(r_start, c_start)]
        curr_r, curr_c = r_start, c_start
        
        is_spur = False
        
        # Walk
        for _ in range(max_length + 1):
            # Check if current pixel is a junction
            # We check the ORIGINAL connectivity
            # (If we used dynamic, we might eat a whole tree)
            # But we need to check if it connects to something "bigger".
            
            # Count neighbors of current pixel
            # (We can use the precomputed neighbors array, but we need to be careful 
            # if we are walking along the line, the neighbors count includes the path we came from)
            
            nb_count = neighbors[curr_r, curr_c]
            
            if nb_count > 2:
                # It's a junction!
                # The path *up to here* (excluding this junction pixel) is a spur.
                is_spur = True
                break
            
            # If not a junction, find the next pixel
            # We need to find a neighbor that is 1 and NOT in path (visited)
            found_next = False
            for dr, dc in offsets:
                nr, nc = curr_r + dr, curr_c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    if skel[nr, nc] == 1:
                        # Check if already in path (avoid backtracking)
                        # Since path is short, simple check is fine.
                        # Optimization: just check the last 2 pixels
                        if len(path) > 1 and (nr, nc) == path[-2]:
                            continue
                        if (nr, nc) == path[-1]: # Should not happen
                            continue
                            
                        # Found next pixel
                        curr_r, curr_c = nr, nc
                        path.append((nr, nc))
                        found_next = True
                        break
            
            if not found_next:
                # End of line (another endpoint)
                # It's an isolated line.
                break
        
        if is_spur:
            # Remove all pixels in path EXCEPT the last one (which is the junction)
            # The loop breaks when curr_r, curr_c is the junction.
            # But path includes the junction as the last element.
            # So remove path[:-1]
            for r, c in path[:-1]:
                to_remove[r, c] = True
                
    # Apply removal
    skel[to_remove] = 0
    return (skel * 255).astype(np.uint8)


def remove_steep_segments(skeleton, max_angle_deg):
    """
    Removes segments from the skeleton that have an angle steeper than max_angle_deg.
    Segments are paths between junctions or endpoints.
    """
    skel = (skeleton > 0).astype(np.uint8)
    h, w = skel.shape
    
    # 1. Find Nodes (Endpoints and Junctions)
    kernel = np.array([[1,1,1],[1,0,1],[1,1,1]], dtype=np.uint8)
    neighbors_count = cv2.filter2D(skel, -1, kernel, borderType=cv2.BORDER_CONSTANT)
    # Nodes are pixels with != 2 neighbors.
    
    node_mask = ((neighbors_count != 2) & (skel == 1))
    nodes = np.argwhere(node_mask)
    
    # To quickly check if a pixel is a node
    is_node = np.zeros_like(skel, dtype=bool)
    is_node[node_mask] = True
    
    # Mask to keep track of visited INTERNAL pixels of segments
    visited = np.zeros_like(skel, dtype=bool)
    
    # Set of processed direct edges (node-to-node) to avoid double counting
    processed_direct_edges = set()
    
    pixels_to_remove = []
    
    offsets = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
    
    for r_start, c_start in nodes:
        # Check all neighbors to find outgoing segments
        for dr, dc in offsets:
            nr, nc = r_start + dr, c_start + dc
            
            if 0 <= nr < h and 0 <= nc < w and skel[nr, nc] == 1:
                # Found a connected pixel
                
                if is_node[nr, nc]:
                    # Direct Node-to-Node connection
                    p1 = (r_start, c_start)
                    p2 = (nr, nc)
                    if p1 > p2: p1, p2 = p2, p1
                    
                    if (p1, p2) in processed_direct_edges:
                        continue
                    
                    processed_direct_edges.add((p1, p2))
                    
                    # Analyze this short segment
                    dy = p2[0] - p1[0]
                    dx = p2[1] - p1[1]
                    angle = np.degrees(np.arctan2(dy, dx))
                    
                    ang_check = angle % 180
                    if ang_check > 90: ang_check -= 180
                    
                    if abs(ang_check) > max_angle_deg:
                        # For direct connections, we can't easily remove "the edge" without removing a node.
                        # But usually these are part of a staircase.
                        # If we remove one of the pixels? No, that breaks nodes.
                        # Let's skip direct connections for now to be safe.
                        pass 
                
                else:
                    # Connection to a non-node pixel. This is a segment.
                    if visited[nr, nc]:
                        continue
                        
                    # Trace the segment
                    segment_pixels = [(nr, nc)]
                    curr_r, curr_c = nr, nc
                    visited[curr_r, curr_c] = True
                    
                    found_end = False
                    end_node = None
                    
                    while True:
                        # Find next neighbor
                        found_next = False
                        for ddr, ddc in offsets:
                            nnr, nnc = curr_r + ddr, curr_c + ddc
                            if 0 <= nnr < h and 0 <= nnc < w and skel[nnr, nnc] == 1:
                                # Don't go back to the pixel we just came from
                                if len(segment_pixels) >= 2:
                                    if (nnr, nnc) == segment_pixels[-2]:
                                        continue
                                elif (nnr, nnc) == (r_start, c_start):
                                    continue
                                    
                                if is_node[nnr, nnc]:
                                    # Hit a node! Segment ended.
                                    end_node = (nnr, nnc)
                                    found_end = True
                                    break
                                else:
                                    # Another internal pixel
                                    if not visited[nnr, nnc]:
                                        visited[nnr, nnc] = True
                                        segment_pixels.append((nnr, nnc))
                                        curr_r, curr_c = nnr, nnc
                                        found_next = True
                                        break
                        
                        if found_end:
                            break
                        if not found_next:
                            break
                    
                    if found_end:
                        # Analyze segment
                        p1 = (r_start, c_start)
                        p2 = end_node
                        
                        dy = p2[0] - p1[0]
                        dx = p2[1] - p1[1]
                        angle = np.degrees(np.arctan2(dy, dx))
                        
                        # Check deviation from horizontal
                        ang_check = angle % 180
                        if ang_check > 90: ang_check -= 180
                        
                        if abs(ang_check) > max_angle_deg:
                            # Remove segment pixels
                            pixels_to_remove.extend(segment_pixels)

    # Apply removal
    for r, c in pixels_to_remove:
        skel[r, c] = 0
        
    return (skel * 255).astype(np.uint8)


def fill_holes(bw_img, min_size=10):
    """
    Fills holes smaller than min_size in the binary image.
    bw_img: uint8 image (0/255 or 0/1)
    min_size: maximum area of hole to fill
    """
    bool_img = bw_img.astype(bool)
    filled = remove_small_holes(bool_img, area_threshold=min_size)
    return (filled.astype(np.uint8) * 255) if bw_img.max() > 1 else filled.astype(np.uint8)
