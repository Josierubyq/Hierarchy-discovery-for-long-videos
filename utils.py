import numpy as np
from sklearn.preprocessing import StandardScaler



def compute_window_similarity(window_a, window_b, method='cosine'):
    """Calculate the similarity between two windows"""
    if method == 'cosine':
        mean_a = np.mean(window_a, axis=0)
        mean_b = np.mean(window_b, axis=0)
        na, nb = np.linalg.norm(mean_a), np.linalg.norm(mean_b)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(mean_a, mean_b) / (na * nb))
    elif method == 'l2':
        mean_a = np.mean(window_a, axis=0)
        mean_b = np.mean(window_b, axis=0)
        return float(np.exp(-np.linalg.norm(mean_a - mean_b)))
    elif method == 'correlation':
        mean_a = np.mean(window_a, axis=0)
        mean_b = np.mean(window_b, axis=0)
        if np.std(mean_a) == 0 or np.std(mean_b) == 0:
            return 1.0 if np.allclose(mean_a, mean_b) else 0.0
        r = np.corrcoef(mean_a, mean_b)[0, 1]
        return float(r) if not np.isnan(r) else 0.0
    return 0.0


def sw_asymmetric(feats, window_size=10, step_reset=1, threshold=0.8,
                  similarity_method='cosine', min_segment_frames=10):
    """Asymmetric sliding window segmentation
    
    Right window fixed at window_size; the left window starts at window_size and increments by 1 each time,
    until it reaches the upper limit (3*window) or a split is triggered, at which point it resets; reset step size = 1.
    """
    T = feats.shape[0]
    right_size = int(window_size)
    left_init = int(window_size)
    grow = 1
    max_left = right_size * 3
    
    boundaries, similarities, positions = [], [], []
    pos, left_len = 0, left_init
    
    while pos + left_len + right_size <= T:
        left_start = pos
        left_end = pos + left_len
        right_end = left_end + right_size
        
        a = feats[left_start:left_end]
        b = feats[left_end:right_end]
        sim = compute_window_similarity(a, b, similarity_method)
        
        similarities.append(float(sim))
        positions.append(left_end)
        
        if sim < threshold:
            mid = left_end
            if (not boundaries) or (mid - boundaries[-1] >= min_segment_frames):
                boundaries.append(int(mid))
                pos = mid
                left_len = left_init
            else:
                left_len = min(left_len + grow, max_left)
                if (pos + left_len + right_size > T) or (left_len >= max_left):
                    pos += step_reset
                    left_len = left_init
        else:
            left_len = min(left_len + grow, max_left)
            if (pos + left_len + right_size > T) or (left_len >= max_left):
                pos += step_reset
                left_len = left_init
    
    return boundaries, similarities, positions


def traditional_diff_peaks(feats_scaled, fps):
    """Traditional differential peak detection (85th percentile), pre-filtering interval 0.3s"""
    div_boundaries = []
    try:
        from scipy.signal import savgol_filter
        diffs = feats_scaled[1:] - feats_scaled[:-1]
        l2 = np.linalg.norm(diffs, axis=1)
        if len(l2) >= 5:
            win = min(7, len(l2) // 4)
            win += 1 if win % 2 == 0 else 0
            if win >= 3:
                l2_s = savgol_filter(l2, win, 1, mode='mirror')
                thr = np.percentile(l2_s, 85)
                cand = np.where(l2_s > thr)[0]
                gap = max(1, int(0.3 * fps))  # 0.3s
                for c in cand:
                    if not div_boundaries or c - div_boundaries[-1] >= gap:
                        div_boundaries.append(int(c))
    except Exception:
        pass
    return div_boundaries


def refine_boundary(b, feats, fps, radius=None):
    """Refine boundaries: select the local maximum of the smoothed L2 difference curve within ±0.3s"""
    if radius is None:
        radius = max(2, int(0.3 * fps))  # 0.3s
    L = feats.shape[0]
    start = max(2, b - radius)
    end = min(L - 2, b + radius)
    diffs = feats[1:] - feats[:-1]
    l2 = np.linalg.norm(diffs, axis=1)
    l2_smooth = np.convolve(l2, np.ones(3)/3, mode='same') if len(l2) >= 3 else l2
    idx = np.argmax(l2_smooth[start:end+1])
    return int(start + idx)


def post_fuse_and_refine(boundaries_a, boundaries_b, feats_scaled, fps):
    """Fuse two sets of candidates, use 0.3s minimum interval for deduplication, then refine (same 0.3s minimum interval)"""
    MIN_DIST = max(1, int(0.3 * fps))  # 0.3s
    merged = sorted(set(boundaries_a + boundaries_b))
    final = []
    for b in merged:
        if not final or b - final[-1] >= MIN_DIST:
            final.append(int(b))
    # Refine
    refined = []
    for b in final:
        rb = refine_boundary(b, feats_scaled, fps)
        if not refined or rb - refined[-1] >= MIN_DIST:
            refined.append(rb)
    return refined


def evaluate_segmentation_quality(boundaries, total_frames, fps=10):
    """Evaluate segmentation quality"""
    if len(boundaries) == 0:
        return 0.7 if total_frames >= fps * 5 else 0.5
    
    seg_lens, prev = [], 0
    for b in boundaries + [total_frames]:
        seg_lens.append(b - prev)
        prev = b
    
    durs = [l / fps for l in seg_lens]
    mn, av, n, sd = min(durs), np.mean(durs), len(durs), np.std(durs)
    
    score = 1.0
    if mn < 1.0:
        score *= 0.2
    elif mn < 1.5:
        score *= 0.7
    
    if n > 6:
        score *= 0.3
    elif n > 4:
        score *= 0.8
    
    if av < 1.5:
        score *= 0.4
    elif av > 6.0:
        score *= 0.7
    
    if sd > av * 0.8:
        score *= 0.8
    
    return score


def adaptive_asymmetric_search(feats_scaled, fps):
    """Adaptive asymmetric sliding window parameter search"""
    # More dense grid: windows 10/12/15; thresholds 0.72/0.74/0.76/0.80
    window_sizes = [10, 12, 15]
    thresholds = [0.72, 0.74, 0.76, 0.80]
    methods = ['cosine']
    
    best_boundaries, best_score, best_params = [], 0.0, {}
    
    print(f"\n🔍 Adaptive search space:")
    print(f"   Window sizes: {window_sizes} frames = {[w/fps for w in window_sizes]} seconds")
    print(f"   Similarity thresholds: {thresholds}")
    print(f"   Similarity methods: {methods}")
    
    for ws in window_sizes:
        for thr in thresholds:
            for m in methods:
                try:
                    min_seg = max(fps, ws)
                    boundaries, sims, pos = sw_asymmetric(
                        feats_scaled, window_size=ws, step_reset=1,
                        threshold=thr, similarity_method=m, min_segment_frames=min_seg
                    )
                    score = evaluate_segmentation_quality(boundaries, feats_scaled.shape[0], fps)
                    
                    if score > best_score:
                        best_boundaries, best_score = boundaries, score
                        best_params = {
                            'window_size': ws,
                            'threshold': thr,
                            'method': m,
                            'num_boundaries': len(boundaries),
                            'score': score
                        }
                except Exception:
                    continue
    
    return best_boundaries, best_params


def enhanced_runner(feats, fps):
    """Complete Enhanced Segmentation Actuator (Asymmetric Window + Differential Peak + Fusion + Refinement)"""
    print("\n" + "-"*70)
    
    # Standardization Features
    scaler = StandardScaler()
    feats_scaled = scaler.fit_transform(feats)
    
    # 1) Asymmetric sliding window
    print("🔹 Step 1: Asymmetric Sliding Window Segmentation...")
    sw_boundaries, sw_params = adaptive_asymmetric_search(feats_scaled, fps)
    print(f"   ✓ Found {len(sw_boundaries)} boundaries")
    print(f"   ✓ Best parameters: ws={sw_params['window_size']}, thr={sw_params['threshold']:.2f}")
    print(f"   ✓ Quality score: {sw_params['score']:.3f}")
    
    # 2) Traditional differential peak detection (85th percentile), pre-deduplication interval 0.3s
    print("\n🔹 Step 2: Differential Peak Detection...")
    div_boundaries = traditional_diff_peaks(feats_scaled, fps)
    print(f"   ✓ Found {len(div_boundaries)} boundaries")
    
    # 3) Blending + Retouching
    print("\n🔹 Step 3: Fusion and Refinement...")
    final = post_fuse_and_refine(sw_boundaries, div_boundaries, feats_scaled, fps)
    print(f"   ✓ After fusion: {len(final)} boundaries")
    
    # 4) Quality Guarantee
    score = evaluate_segmentation_quality(final, feats.shape[0], fps)
    if score < 0.5:
        if len(sw_boundaries) <= len(div_boundaries):
            final = sw_boundaries
            method_used = 'asymmetric_only'
        else:
            final = div_boundaries
            method_used = 'divergence_only'
    else:
        method_used = 'asymmetric_hybrid_div'
    
    # Statistics
    num_segments = len(final) + 1
    avg_duration = feats.shape[0] / num_segments / fps
    
    print(f"\n📊 Final Statistics:")
    print(f"   Number of Boundaries: {len(final)}")
    print(f"   Number of Segments: {num_segments}")
    print(f"   Average Segment Length: {avg_duration:.1f} seconds")
    print(f"   Final Quality Score: {score:.3f}")
    print(f"   使用方法: {method_used}")
    print("-"*70)
    
    # Build a parameter dictionary
    params = {
        **sw_params,
        'method_used': method_used,
        'sw_boundaries': len(sw_boundaries),
        'div_boundaries': len(div_boundaries),
        'final_boundaries': len(final),
        'final_score': score,
        'num_segments': num_segments,
        'avg_segment_sec': avg_duration
    }
    
    return final, params

def create_segments(boundaries, T):
    """Create a fragment list from a boundary list"""
    segs = []
    prev = 0
    for b in sorted(boundaries):
        b = int(b)
        if b > prev:
            segs.append([int(prev), int(b)])
            prev = b
    if prev < T:
        segs.append([int(prev), int(T)])
    return segs

def extract_atomic_features(features, boundaries, fps):
    """Extract representative features for each atomic unit from CLIP features"""
    T = features.shape[0]
    segments = []
    
    # Build fragment list
    prev = 0
    for b in boundaries:
        if b > prev:
            segments.append((prev, b))
            prev = b
    if prev < T:
        segments.append((prev, T))
    
    atomic_features = []
    atomic_centers = []
    
    for start, end in segments:
        # Calculate the average features of the segment
        segment_feats = features[start:end]
        mean_feat = np.mean(segment_feats, axis=0)
        
        # L2 normalization
        mean_feat = mean_feat / (np.linalg.norm(mean_feat) + 1e-8)
        atomic_features.append(mean_feat)
        
        # Calculate time center (seconds)
        center_time = (start + end) / 2.0 / fps
        atomic_centers.append(center_time)
    
    return np.array(atomic_features), np.array(atomic_centers)

def build_semantic_consistency_map(atomic_features, atomic_centers, sigma):
    """Build the semantic consistency map (SCM)"""
    K = len(atomic_features)
    M = np.zeros((K, K))
    
    for i in range(K):
        for j in range(K):
            # 1. Semantic Core: Cosine Similarity
            K_sem = np.dot(atomic_features[i], atomic_features[j])
            
            # 2. Time-based kernel: Gaussian RBF
            time_diff = atomic_centers[i] - atomic_centers[j]
            K_temp = np.exp(- (time_diff ** 2) / (2 * sigma ** 2))
            
            # 3. Hybrid kernel
            M[i, j] = K_sem * K_temp
    
    return M

def similarity_to_distance(similarity_matrix):
    """Convert similarity matrix to distance matrix"""
    sim_clipped = np.clip(similarity_matrix, 0, 1)
    distance = np.sqrt(2 * (1 - sim_clipped))
    np.fill_diagonal(distance, 0)
    return distance

def time_to_frame(time_sec, fps):
    """Convert time (seconds) to frame index"""
    return int(time_sec * fps)

def calculate_split_error(actual_splits, target_splits, tolerance=15):
    """Calculate the error between actual and target split points (allowing for partial matching)"""
    if len(actual_splits) == 0 or len(target_splits) == 0:
        return float('inf')
    
    # There is a significant variation in the permitted quantity (relaxed to 5)
    if abs(len(actual_splits) - len(target_splits)) > 5:
        return float('inf')
    
    # Use nearest-neighbor matching
    total_error = 0
    matched_count = 0
    
    for target in target_splits:
        # Find the closest actual division point
        min_dist = float('inf')
        for actual in actual_splits:
            dist = abs(actual - target)
            if dist < min_dist:
                min_dist = dist
        
        if min_dist <= tolerance:  # within tolerance
            total_error += min_dist
            matched_count += 1
    
    if matched_count == 0:
        return float('inf')
    
    # Penalize unmatched split points
    penalty = abs(len(actual_splits) - len(target_splits)) * 10
    return (total_error / matched_count) + penalty

def get_split_times_from_segments(segments, fps):
    """Extract split times (in seconds) from segmentation results"""
    split_times = []
    for i in range(len(segments) - 1):
        split_frame = segments[i][1]  # End frame of the current segment
        split_time = split_frame / fps
        split_times.append(int(split_time))
    return split_times

def format_time(seconds):
    return f"{int(seconds//60):02d}:{int(seconds%60):02d}"

def labels_to_segments(labels, atomic_segs):
    segments = []
    K = len(labels)
    event_boundaries = [0]
    for i in range(1, K):
        if labels[i] != labels[i-1]:
            event_boundaries.append(i)
    event_boundaries.append(K)
    for i in range(len(event_boundaries) - 1):
        start_idx = event_boundaries[i]
        end_idx = event_boundaries[i+1] - 1
        start_frame = atomic_segs[start_idx][0]
        end_frame = atomic_segs[end_idx][1]
        segments.append((start_frame, end_frame))
    return segments

def calculate_split_error(actual_splits, target_splits, tolerance=15):
    """Calculate the error between actual and target split points"""
    if len(actual_splits) == 0 or len(target_splits) == 0:
        return float('inf')
    
    if abs(len(actual_splits) - len(target_splits)) > 5:
        return float('inf')
    
    total_error = 0
    matched_count = 0
    
    for target in target_splits:
        min_dist = float('inf')
        for actual in actual_splits:
            dist = abs(actual - target)
            if dist < min_dist:
                min_dist = dist
        
        if min_dist <= tolerance:
            total_error += min_dist
            matched_count += 1
    
    if matched_count == 0:
        return float('inf')
    
    penalty = abs(len(actual_splits) - len(target_splits)) * 10
    return (total_error / matched_count) + penalty
