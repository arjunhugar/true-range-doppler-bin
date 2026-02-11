# plot_true_rd_plus.py
# True Range–Doppler (2D FFT) from BIN + CFAR overlay + slices + range profiles + 3D + animation

import os
import numpy as np
import matplotlib.pyplot as plt

# Optional: 3D plotting
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# Optional: animation
from matplotlib.animation import FuncAnimation

# =========================
# INPUT FILES (NO LAPTOP PATHS)
# =========================
# Put your data in the repo under: data/
# Example:
#   data/example.bin
#   data/example.meta.txt
bin_path  = os.path.join("data", "example.bin")
meta_path = os.path.join("data", "example.meta.txt")

# =========================
# WHAT TO GENERATE
# =========================
DO_MAIN_RD_PLOT      = True
DO_CFAR_OVERLAY      = True
DO_VELOCITY_SLICE    = True
DO_RANGE_PROFILES    = True
DO_3D_SURFACE        = False     # set True if you want 3D plot (can be heavy)
DO_ANIMATION         = False     # set True to generate an animation over frames

# =========================
# FRAME SELECTION
# =========================
# Plot one specific frame:
#   frame_index = 332
# Plot last frame:
#   frame_index = None
frame_index = None

# For animation (if enabled)
anim_start = 0
anim_stop  = None           # None = until end
anim_step  = 1
anim_fps   = 8

# =========================
# FFT / DISPLAY TUNING
# =========================
dyn_range_db = 55.0          # show top X dB (relative)
sum_over_rx  = True
rx_to_plot   = 0

# Crop for thesis-look
max_range_m  = 40.0
max_abs_vel  = 15.0

# Doppler zero-padding
doppler_pad_factor = 4       # 1 = no pad, 2/4 recommended

# Windows
use_blackman_harris_range = True
use_hann_doppler          = True

# =========================
# CFAR SETTINGS (2D CA-CFAR)
# =========================
cfar_guard_r = 2
cfar_guard_v = 2
cfar_train_r = 8
cfar_train_v = 8
cfar_threshold_db = 12.0
cfar_min_rel_db = 40.0

# =========================
# Helpers
# =========================
def read_meta(meta_file: str) -> dict:
    meta = {}
    with open(meta_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or "=" not in line:
                continue
            k, v = line.split("=", 1)
            meta[k.strip()] = v.strip()

    out = {}
    out["Ns"]  = int(meta["Ns"])
    out["Nc"]  = int(meta["Nc"])
    out["Nrx"] = int(meta["Nrx"])
    out["Ntx"] = int(meta.get("Ntx", "4"))
    out["fs"]  = float(meta["fs"])
    out["Tc"]  = float(meta["Tc"])
    out["dF"]  = float(meta["dF"])
    out["f0"]  = float(meta["f0"])
    out["c"]   = 299792458.0
    return out

def blackman_harris(N: int) -> np.ndarray:
    n = np.arange(N, dtype=np.float64)
    a0, a1, a2, a3 = 0.35875, 0.48829, 0.14128, 0.01168
    w = (a0
         - a1*np.cos(2*np.pi*n/(N-1))
         + a2*np.cos(4*np.pi*n/(N-1))
         - a3*np.cos(6*np.pi*n/(N-1)))
    return w.astype(np.float32)

def hann(N: int) -> np.ndarray:
    return np.hanning(N).astype(np.float32)

def build_axes(P, Ns, Nc_fft):
    fs, Tc, dF, f0, c = P["fs"], P["Tc"], P["dF"], P["f0"], P["c"]
    S = dF / Tc  # chirp slope (Hz/s)

    Nr = Ns // 2 + 1
    fb = np.arange(Nr) * (fs / Ns)
    rng_axis = c * fb / (2.0 * S)

    k = np.arange(Nc_fft) - (Nc_fft // 2)
    fd = k / (Nc_fft * Tc)
    vel_axis = c * fd / (2.0 * f0)
    return rng_axis, vel_axis

def compute_rd(frame, P, doppler_pad_factor=1, sum_over_rx=True, rx_to_plot=0,
               use_bh_range=True, use_hann_doppler=True):
    Ns, Nc, Nrx = frame.shape

    # 1) Remove per-chirp DC
    frame = frame - np.mean(frame, axis=0, keepdims=True)

    # 2) Windowing
    w_r = blackman_harris(Ns) if use_bh_range else hann(Ns)
    w_d = hann(Nc) if use_hann_doppler else np.ones(Nc, dtype=np.float32)
    frame_win = frame * w_r[:, None, None] * w_d[None, :, None]

    # 3) Range FFT
    RFFT = np.fft.rfft(frame_win, axis=0)

    # 4) Doppler FFT (zero-pad)
    Nc_fft = int(Nc * doppler_pad_factor)
    RD = np.fft.fftshift(np.fft.fft(RFFT, n=Nc_fft, axis=1), axes=1)

    # 5) Power
    pow_lin = np.abs(RD) ** 2
    if sum_over_rx:
        pow_lin = np.sum(pow_lin, axis=2)
    else:
        pow_lin = pow_lin[:, :, rx_to_plot]
    pow_db = 10.0 * np.log10(pow_lin + 1e-20)

    rng_axis, vel_axis = build_axes(P, Ns, Nc_fft)
    return pow_db, rng_axis, vel_axis

def crop_rd(pow_db, rng_axis, vel_axis, max_range_m, max_abs_vel):
    r_mask = rng_axis <= max_range_m
    v_mask = np.abs(vel_axis) <= max_abs_vel
    return pow_db[r_mask, :][:, v_mask], rng_axis[r_mask], vel_axis[v_mask]

def ca_cfar_2d(pow_db, guard_r, guard_v, train_r, train_v, thr_db, min_rel_db=None):
    R, V = pow_db.shape
    det = np.zeros((R, V), dtype=bool)

    if min_rel_db is not None:
        frame_max = np.nanmax(pow_db)
        test_mask = pow_db >= (frame_max - float(min_rel_db))
    else:
        test_mask = np.ones_like(det, dtype=bool)

    gr, gv = guard_r, guard_v
    tr, tv = train_r, train_v
    r0 = tr + gr
    v0 = tv + gv

    for r in range(r0, R - r0):
        for v in range(v0, V - v0):
            if not test_mask[r, v]:
                continue

            r1, r2 = r - (tr + gr), r + (tr + gr)
            v1, v2 = v - (tv + gv), v + (tv + gv)
            rg1, rg2 = r - gr, r + gr
            vg1, vg2 = v - gv, v + gv

            win = pow_db[r1:r2+1, v1:v2+1]
            mask = np.ones_like(win, dtype=bool)
            mask[(rg1 - r1):(rg2 - r1)+1, (vg1 - v1):(vg2 - v1)+1] = False

            noise_cells = win[mask]
            if noise_cells.size < 10:
                continue

            noise_est = np.median(noise_cells)
            if pow_db[r, v] > (noise_est + thr_db):
                det[r, v] = True

    return det

def ensure_out_dir():
    out_dir = os.path.join("out")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir

# =========================
# Load meta + bin
# =========================
if __name__ == "__main__":
    if not os.path.exists(bin_path):
        raise FileNotFoundError(
            f"BIN not found: {bin_path}\n"
            "Put your file in ./data/ and name it example.bin (or edit bin_path)."
        )
    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"META not found: {meta_path}\n"
            "Put your file in ./data/ and name it example.meta.txt (or edit meta_path)."
        )

    P = read_meta(meta_path)
    Ns, Nc, Nrx = P["Ns"], P["Nc"], P["Nrx"]
    frame_elems = Ns * Nc * Nrx

    x = np.fromfile(bin_path, dtype=np.float32)
    if x.size < frame_elems:
        raise ValueError(f"BIN too small. Need >= {frame_elems} floats, got {x.size}")

    n_frames = x.size // frame_elems
    x = x[: n_frames * frame_elems]

    if frame_index is None:
        frame_index = n_frames - 1

    if not (0 <= frame_index < n_frames):
        raise ValueError(f"frame_index out of range: {frame_index}. BIN has {n_frames} frames.")

    print(f"[OK] BIN frames available: {n_frames}, using frame_index={frame_index}")

    out_dir = ensure_out_dir()

    def get_frame(idx: int) -> np.ndarray:
        f = x[idx * frame_elems : (idx + 1) * frame_elems]
        return f.reshape(Ns, Nc, Nrx)

    frame = get_frame(frame_index)

    pow_db, rng_axis, vel_axis = compute_rd(
        frame, P,
        doppler_pad_factor=doppler_pad_factor,
        sum_over_rx=sum_over_rx,
        rx_to_plot=rx_to_plot,
        use_bh_range=use_blackman_harris_range,
        use_hann_doppler=use_hann_doppler
    )

    pow_db_crop, rng_crop, vel_crop = crop_rd(
        pow_db, rng_axis, vel_axis,
        max_range_m=max_range_m,
        max_abs_vel=max_abs_vel
    )

    vmax = np.nanmax(pow_db_crop)
    vmin = vmax - dyn_range_db

    # --- Main RD plot ---
    if DO_MAIN_RD_PLOT:
        plt.figure(figsize=(10, 5))
        extent = [vel_crop[0], vel_crop[-1], rng_crop[-1], rng_crop[0]]
        plt.imshow(pow_db_crop, extent=extent, aspect="auto", origin="upper",
                   vmin=vmin, vmax=vmax, interpolation="nearest")
        plt.xlabel("Velocity (m/s)")
        plt.ylabel("Range (m)")
        rx_txt = "Σ over Rx" if sum_over_rx else f"Rx{rx_to_plot}"
        plt.title(f"True Range–Doppler (2D FFT) from BIN — frame {frame_index} ({rx_txt})")
        plt.colorbar(label="Power (dB, relative to max)")
        plt.tight_layout()

        out_png = os.path.join(out_dir, f"RD_true_frame_{frame_index:06d}.png")
        plt.savefig(out_png, dpi=300)
        print(f"[OK] Saved RD -> {out_png}")
        plt.show()

    # --- CFAR overlay ---
    if DO_CFAR_OVERLAY:
        det_mask = ca_cfar_2d(
            pow_db_crop,
            guard_r=cfar_guard_r, guard_v=cfar_guard_v,
            train_r=cfar_train_r, train_v=cfar_train_v,
            thr_db=cfar_threshold_db,
            min_rel_db=cfar_min_rel_db
        )

        rr, vv = np.where(det_mask)
        det_ranges = rng_crop[rr]
        det_vels   = vel_crop[vv]

        plt.figure(figsize=(10, 5))
        extent = [vel_crop[0], vel_crop[-1], rng_crop[-1], rng_crop[0]]
        plt.imshow(pow_db_crop, extent=extent, aspect="auto", origin="upper",
                   vmin=vmin, vmax=vmax, interpolation="nearest")
        plt.scatter(det_vels, det_ranges, s=12)
        plt.xlabel("Velocity (m/s)")
        plt.ylabel("Range (m)")
        plt.title(f"RD + 2D CA-CFAR overlay — frame {frame_index} (thr={cfar_threshold_db:.1f} dB)")
        plt.colorbar(label="Power (dB, relative to max)")
        plt.tight_layout()

        out_png = os.path.join(out_dir, f"RD_CAFAR_overlay_frame_{frame_index:06d}.png")
        plt.savefig(out_png, dpi=300)
        print(f"[OK] Saved CFAR overlay -> {out_png}")
        plt.show()

    # --- Velocity slice ---
    if DO_VELOCITY_SLICE:
        r_peak = np.argmax(np.max(pow_db_crop, axis=1))
        r_val  = rng_crop[r_peak]
        vel_slice = pow_db_crop[r_peak, :]

        plt.figure(figsize=(9, 4))
        plt.plot(vel_crop, vel_slice)
        plt.grid(True)
        plt.xlabel("Velocity (m/s)")
        plt.ylabel("Power (dB, rel)")
        plt.title(f"Velocity slice at range ≈ {r_val:.2f} m — frame {frame_index}")
        plt.tight_layout()

        out_png = os.path.join(out_dir, f"velocity_slice_frame_{frame_index:06d}.png")
        plt.savefig(out_png, dpi=300)
        print(f"[OK] Saved velocity slice -> {out_png}")
        plt.show()

    # --- Range profiles ---
    if DO_RANGE_PROFILES:
        v0 = pow_db_crop.shape[1] // 2
        rp_zero_doppler = pow_db_crop[:, v0]
        rp_max = np.max(pow_db_crop, axis=1)

        plt.figure(figsize=(9, 4))
        plt.plot(rng_crop, rp_zero_doppler, label="Range profile @ 0 Doppler")
        plt.plot(rng_crop, rp_max, label="Range profile (max over Doppler)")
        plt.grid(True)
        plt.xlabel("Range (m)")
        plt.ylabel("Power (dB, rel)")
        plt.title(f"Range profiles — frame {frame_index}")
        plt.legend()
        plt.tight_layout()

        out_png = os.path.join(out_dir, f"range_profiles_frame_{frame_index:06d}.png")
        plt.savefig(out_png, dpi=300)
        print(f"[OK] Saved range profiles -> {out_png}")
        plt.show()
