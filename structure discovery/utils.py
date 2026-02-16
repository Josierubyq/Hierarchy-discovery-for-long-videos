def format_time(seconds):
    """Format time as mm:ss"""
    return f"{int(seconds//60):02d}:{int(seconds%60):02d}"

def labels_to_segments(labels, atomic_segs):
    """Convert cluster labels to event segments"""
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