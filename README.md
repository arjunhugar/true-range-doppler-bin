
# True Range–Doppler from BIN (2D FFT)

This repository contains a Python script that computes a Range–Doppler heatmap from a real-valued IF data cube stored in a `.bin` file + a `.meta.txt` file.

## Features
- 2D FFT (Range FFT + Doppler FFT with zero-padding)
- Per-chirp DC removal
- Blackman–Harris (range) + Hann (Doppler) windows
- Optional 2D CA-CFAR overlay
- Velocity slice and range profile plots
- Outputs saved to `./out/`

## How to run
```bash
pip install -r requirements.txt
python src/plot_true_rd_plus.py
