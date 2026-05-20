import torch
import os
import warnings
from tqdm import tqdm
import torch.nn.functional as F
# with no cv2 or numpy ai slopspam..
try:
    from torchvision.models.optical_flow import raft_small, Raft_Small_Weights
    HAS_RAFT = True
except ImportError:
    HAS_RAFT = False
    warnings.warn("torchvision >= 0.12 required for RAFT optical flow.")

class torchflow:
    @classmethod
    def INPUT_TYPES(cls):
        max_cores = os.cpu_count()
        return {
            "required": {
                "frames": ("IMAGE",), 
                "input_fps": ("INT", {"default": 60, "min": 1, "max": 240}),
                "target_fps": ("INT", {"default": 30, "min": 1, "max": 240}),
                "method": ([
                    "Frame dropping",
                    "Frame blending",
                    "Optical flow",
                    "Motion‑compensated"
                ],),
                "cpu_threads": (
                    ["auto"] + [str(i) for i in range(1, max_cores + 1)],
                    {"default": "auto"},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE", "FLOAT")
    RETURN_NAMES = ("frames", "fps")
    FUNCTION = "downsample"
    CATEGORY = "Video"

    def downsample(self, frames, input_fps, target_fps, method, cpu_threads):

        if target_fps >= input_fps:
            print("Target FPS is equal or higher than input FPS. No downsampling applied.")
            return (frames, float(input_fps))

        pt_frames = frames.permute(0, 3, 1, 2)

        ratio = input_fps / target_fps
        frame_count = int(len(pt_frames) / ratio)
        indices_int = [int(i * ratio) for i in range(frame_count)]

        def generate_time_indices(num_frames, src_fps, dst_fps):
            duration = num_frames / float(src_fps)
            target_frame_count = int(round(duration * float(dst_fps)))
            if target_frame_count == 0:
                return []
            step = float(num_frames) / target_frame_count
            return [i * step for i in range(target_frame_count)]

        print(f"[DownsampleFPSNode] Method: {method}, "
              f"input_fps={input_fps}, target_fps={target_fps}, "
              f"in_frames={len(pt_frames)}")

        threads = os.cpu_count() if cpu_threads == "auto" else int(cpu_threads)
        if method == "Frame dropping":
            indices = indices_int
            selected = frame_drop(pt_frames, indices)

        elif method == "Frame blending":
            indices = indices_int
            selected = frame_blend(pt_frames, indices, threads)

        elif method == "Optical flow":
            indices = indices_int
            selected = optical_flow(pt_frames, indices, threads)

        elif method == "Motion‑compensated":
            indices = generate_time_indices(len(pt_frames), input_fps, target_fps)
            print(f"[DownsampleFPSNode] Motion‑compensated: "
                  f"in_frames={len(pt_frames)}, out_frames={len(indices)} (time‑accurate)")

            selected = motion_compensated(pt_frames, indices, threads)

        if isinstance(selected, list):
            selected = torch.stack(selected)

        out_tensor = selected.permute(0, 2, 3, 1)
        return (out_tensor, float(target_fps))

def _compute_raft_flow(model, transforms, img1, img2, device, max_size=512):
    orig_H, orig_W = img1.shape[2], img1.shape[3]
    scale_factor = 1.0
    
    if max(orig_H, orig_W) > max_size:
        scale_factor = max_size / float(max(orig_H, orig_W))
        new_H = int(orig_H * scale_factor)
        new_W = int(orig_W * scale_factor)
        img1_s = F.interpolate(img1, size=(new_H, new_W), mode='bicubic', align_corners=False)
        img2_s = F.interpolate(img2, size=(new_H, new_W), mode='bicubic', align_corners=False)
    else:
        img1_s, img2_s = img1, img2

    img1_t, img2_t = transforms(img1_s, img2_s)
    _, _, H_s, W_s = img1_t.shape
    pad_h = (8 - H_s % 8) % 8
    pad_w = (8 - W_s % 8) % 8
    
    if pad_h > 0 or pad_w > 0:
        img1_t = F.pad(img1_t, (0, pad_w, 0, pad_h))
        img2_t = F.pad(img2_t, (0, pad_w, 0, pad_h))
    
    if device.type == 'cuda':
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            list_of_flows = model(img1_t, img2_t)
    else:
        list_of_flows = model(img1_t, img2_t)
        
    flow = list_of_flows[-1].float()
    
    if pad_h > 0 or pad_w > 0:
        flow = flow[:, :, :H_s, :W_s]
        
    if scale_factor != 1.0:
        flow = F.interpolate(flow, size=(orig_H, orig_W), mode='bicubic', align_corners=False)
        flow = flow / scale_factor

    return flow

def optical_flow(pt_frames, indices, threads=0):
    if not HAS_RAFT:
        raise ImportError("torchvision.models.optical_flow is required for RAFT. Please upgrade torchvision.")
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cuts = detect_scene_cuts(pt_frames)
    
    print("[Optical Flow] Loading RAFT model...")
    weights = Raft_Small_Weights.DEFAULT
    transforms = weights.transforms()
    model = raft_small(weights=weights, progress=False).to(device).eval()
    
    flow_cache = {}
    
    print("[Optical Flow] Precomputing RAFT optical flow...")
    with torch.no_grad():
        for i in tqdm(range(len(pt_frames) - 1), desc="RAFT Flow Cache", colour="yellow", unit="pair"):
            if cuts[i]:
                continue
                
            img1 = pt_frames[i:i+1].to(device)
            img2 = pt_frames[i+1:i+2].to(device)
            
            flow = _compute_raft_flow(model, transforms, img1, img2, device)
            flow_cache[i] = flow.squeeze(0).cpu()
            
            del img1, img2, flow
            if i % 10 == 0:
                torch.cuda.empty_cache()
                
    del model
    torch.cuda.empty_cache()
            
    selected = []
    
    for idx in tqdm(indices, desc="Optical Flow (RAFT)", colour="blue", unit="frame"):
        i1 = int(idx)
        
        if i1 >= len(pt_frames) - 1 or cuts[i1]:
            selected.append(pt_frames[i1])
            continue
            
        flow = flow_cache.get(i1)
        if flow is None:
            selected.append(pt_frames[i1])
            continue
            
        warped = warp_halfway(pt_frames[i1].to(device), flow.to(device))
        selected.append(warped.cpu())
        
    return torch.stack(selected)

def _warp_frame(pt_frame, flow, t):
    C, H, W = pt_frame.shape
    device = pt_frame.device
    
    flow_scaled = flow * t
    
    y, x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
    
    x_norm = 2.0 * (x + flow_scaled[0]) / max(W - 1, 1) - 1.0
    y_norm = 2.0 * (y + flow_scaled[1]) / max(H - 1, 1) - 1.0
    
    grid = torch.stack((x_norm, y_norm), dim=-1).unsqueeze(0)
    frame_batch = pt_frame.unsqueeze(0)
    
    warped = F.grid_sample(frame_batch, grid, mode='bicubic', padding_mode='border', align_corners=True)
    return warped.squeeze(0)

def _interpolate_frame_cached(frame_a, frame_b, t, flow_fwd, flow_bwd):
    warp_a = _warp_frame(frame_a, flow_fwd, t)
    warp_b = _warp_frame(frame_b, flow_bwd, 1.0 - t)
    
    mag_fwd = torch.norm(flow_fwd, dim=0, keepdim=True)
    mag_bwd = torch.norm(flow_bwd, dim=0, keepdim=True)
    
    weight_a = torch.exp(-mag_fwd)
    weight_b = torch.exp(-mag_bwd)
    
    weights = weight_a + weight_b + 1e-6
    weight_a = weight_a / weights
    weight_b = weight_b / weights
    
    blended = warp_a * weight_a + warp_b * weight_b
    return torch.clamp(blended, 0.0, 1.0)

def motion_compensated(pt_frames, indices, threads=0):
    if not HAS_RAFT:
        raise ImportError("torchvision.models.optical_flow is required for RAFT. Please upgrade torchvision.")
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_frames = len(pt_frames)
    
    if num_frames < 2:
        return pt_frames

    cuts = detect_scene_cuts(pt_frames)

    print("[Motion‑compensated] Loading RAFT model...")
    weights = Raft_Small_Weights.DEFAULT
    transforms = weights.transforms()
    model = raft_small(weights=weights, progress=False).to(device).eval()

    flow_fwd_cache = {}
    flow_bwd_cache = {}

    print("[Motion‑compensated] Precomputing RAFT optical flow (forward and backward)...")
    with torch.no_grad():
        for i in tqdm(range(num_frames - 1), desc="RAFT Flow Cache", colour="yellow", unit="pair"):
            if cuts[i]:
                continue

            img1 = pt_frames[i:i+1].to(device)
            img2 = pt_frames[i+1:i+2].to(device)

            flow_fwd = _compute_raft_flow(model, transforms, img1, img2, device)
            flow_bwd = _compute_raft_flow(model, transforms, img2, img1, device)

            flow_fwd_cache[i] = flow_fwd.squeeze(0).cpu()
            flow_bwd_cache[i] = flow_bwd.squeeze(0).cpu()

            del img1, img2, flow_fwd, flow_bwd
            if i % 5 == 0:
                torch.cuda.empty_cache()

    del model
    torch.cuda.empty_cache()

    selected = []
    
    for idx in tqdm(indices, desc="Motion‑compensated (RAFT, cached)", colour="blue", unit="frame"):
        if idx <= 0.0:
            selected.append(pt_frames[0])
            continue
        if idx >= num_frames - 1:
            selected.append(pt_frames[-1])
            continue

        base = int(torch.floor(torch.tensor(idx)).item())
        t = float(idx - base)
        is_cut = cuts[base] if 0 <= base < len(cuts) else False

        frame_a = pt_frames[base]
        frame_b = pt_frames[base + 1]

        if is_cut:
            selected.append(frame_a if t < 0.5 else frame_b)
            continue

        if t <= 1e-6:
            selected.append(frame_a)
            continue
        if t >= 1.0 - 1e-6:
            selected.append(frame_b)
            continue

        flow_fwd = flow_fwd_cache.get(base)
        flow_bwd = flow_bwd_cache.get(base)

        if flow_fwd is None or flow_bwd is None:
            selected.append(frame_a if t < 0.5 else frame_b)
            continue

        out = _interpolate_frame_cached(frame_a.to(device), frame_b.to(device), t, flow_fwd.to(device), flow_bwd.to(device))
        selected.append(out.cpu())

    return torch.stack(selected)
    
def frame_drop(pt_frames, indices, threads=0):
    
    idx = torch.tensor(indices, dtype=torch.long, device=pt_frames.device)
    idx = torch.clamp(idx, 0, len(pt_frames) - 1)
    selected = pt_frames[idx]

    for _ in tqdm(range(len(idx)), desc="Dropping (fast tensor indexing)", unit="frame", colour="blue"):
        pass

    return selected

def detect_scene_cuts(pt_frames, threshold=0.05):
    n = len(pt_frames)
    if n < 2:
        return torch.zeros(0, dtype=torch.bool, device=torch.device('cpu'))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    small_frames = []
    chunk_size = 32
    
    for i in range(0, n, chunk_size):
        chunk = pt_frames[i:i+chunk_size].to(device)
        small_chunk = F.interpolate(chunk, size=(64, 64), mode='bicubic', align_corners=False)
        small_frames.append(small_chunk.cpu())
        
    small_frames = torch.cat(small_frames, dim=0)
    diff = small_frames[1:] - small_frames[:-1]
    mse = (diff ** 2).mean(dim=[1, 2, 3])
    
    cuts = mse > threshold
    return cuts

def estimate_motion(frame_a, frame_b):
    diff = torch.abs(frame_a - frame_b)
    return diff.mean().item()

def frame_blend(pt_frames, indices, threads=0, base_radius=3):
    
    n = len(pt_frames)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    cuts = detect_scene_cuts(pt_frames, threshold=0.05)
    
    def is_cut_between(a, b):
        if a == b:
            return False
        lo, hi = min(a, b), max(a, b)
        return cuts[lo:hi].any().item()

    selected = []

    for idx in tqdm(indices, desc=f"Blending (adaptive radius={base_radius})", colour="blue", unit="frame"):
        center = int(idx)

        if center < n - 1:
            motion = estimate_motion(pt_frames[center], pt_frames[center + 1])
        else:
            motion = 0.0

        adaptive_radius = int(base_radius * (1.0 - motion))
        adaptive_radius = max(1, min(adaptive_radius, base_radius))

        start = max(0, center - adaptive_radius)
        end = min(n - 1, center + adaptive_radius)

        frames_to_blend = []
        weights_rgb = []

        for i in range(start, end + 1):
            if is_cut_between(center, i):
                continue

            dist = abs(i - center)

            sigma = max((adaptive_radius / 2.0), 1e-6)
            w = torch.exp(torch.tensor(-(dist ** 2) / (2 * sigma ** 2), device=device, dtype=torch.float32))

            wr = w * 0.95
            wg = w * 1.0
            wb = w * 0.9

            frames_to_blend.append(pt_frames[i].to(device))
            weights_rgb.append(torch.stack([wr, wg, wb]).view(3, 1, 1).to(device))
            
        if not frames_to_blend:
            selected.append(pt_frames[center])
            continue

        stack_f = torch.stack(frames_to_blend)
        stack_w = torch.stack(weights_rgb)
        
        stack_w = stack_w / stack_w.sum(dim=0, keepdim=True)
        lin = stack_f ** 2.2
        acc = (lin * stack_w).sum(dim=0)
        out = acc ** (1.0 / 2.2)
        out = torch.clamp(out, 0.0, 1.0)
        selected.append(out.cpu())

    return torch.stack(selected)

def warp_halfway(pt_frame, flow):
    C, H, W = pt_frame.shape
    device = pt_frame.device
    
    flow_half = flow * 0.5
    
    y, x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
    
    x_norm = 2.0 * (x + flow_half[0]) / max(W - 1, 1) - 1.0
    y_norm = 2.0 * (y + flow_half[1]) / max(H - 1, 1) - 1.0
    
    grid = torch.stack((x_norm, y_norm), dim=-1).unsqueeze(0)
    frame_batch = pt_frame.unsqueeze(0)
    
    warped = F.grid_sample(frame_batch, grid, mode='bicubic', padding_mode='border', align_corners=True)
    return warped.squeeze(0)

