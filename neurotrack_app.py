"""
NeuroTrack App (Streamlit + CLI-friendly)

This single-file script can run as:
  - A Streamlit app (if `streamlit` is installed):
      streamlit run neurotrack_app.py
  - A CLI/test mode (if Streamlit is NOT installed):
      python neurotrack_app.py

In environments where `streamlit` is missing (ModuleNotFoundError), the script will automatically fall back to a self-contained test that generates synthetic keypoint data, runs the tremor & gait pipeline, prints the summary, saves a CSV, and writes a couple of PNG plots. This makes debugging easy without requiring additional packages.

Functions included here are the same analysis building blocks used earlier in the notebook:
 - extract_keypoints_df(results)
 - detect_tremor(df_kp, fps=...)
 - analyze_gait(df_kp, fps=...)
 - build_summary(tremor_df, gait_df, symmetry_df)

The Streamlit UI (if available) still uses the same functions so your app logic remains intact.
"""

# Standard imports
import os
import sys
import math
import tempfile
import warnings

# Optional UI/runtime imports
try:
    import streamlit as st
    HAS_STREAMLIT = True
except Exception:
    HAS_STREAMLIT = False

try:
    from ultralytics import YOLO
    HAS_ULTRALYTICS = True
except Exception:
    HAS_ULTRALYTICS = False

# Analysis imports (assume these are available in most Python envs)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq
from scipy.signal import find_peaks

# ---------------------- Core helper functions ----------------------

def save_uploaded_file(uploaded_file, filename="input_video.mp4"):
    """Save a file-like (Streamlit UploadedFile) to disk."""
    with open(filename, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return filename


def extract_keypoints_df(results):
    """
    Convert an Ultralytics YOLOv8 results list into a pandas DataFrame with columns:
      ['frame', 'person', 'keypoint_id', 'x', 'y', 'conf']

    The function is defensive: if `results` isn't the exact expected structure, it
    will try alternative access patterns and return an empty DataFrame if nothing found.
    """
    rows = []
    # Results is expected to be iterable of frame results
    for frame_idx, r in enumerate(results):
        kpts = getattr(r, 'keypoints', None)
        if kpts is None:
            # try alternative attribute names (robustness)
            kpts = getattr(r, 'masks', None)
        if kpts is None:
            continue
        # Try common ultralytics attributes
        try:
            xy = kpts.xy.cpu().numpy()    # (num_people, num_kp, 2)
            conf = kpts.conf.cpu().numpy() if hasattr(kpts, 'conf') else np.ones(xy.shape[:2])
            for person_id in range(xy.shape[0]):
                for kp_id in range(xy.shape[1]):
                    x, y = xy[person_id, kp_id]
                    c = float(conf[person_id, kp_id])
                    rows.append({'frame': int(frame_idx), 'person': int(person_id), 'keypoint_id': int(kp_id), 'x': float(x), 'y': float(y), 'conf': c})
        except Exception:
            # Fallback: if kpts is list-like with .xy attributes per person
            try:
                for person_id, p in enumerate(kpts):
                    arr = np.array(getattr(p, 'xy', p))
                    for kp_id, (x, y) in enumerate(arr):
                        rows.append({'frame': int(frame_idx), 'person': int(person_id), 'keypoint_id': int(kp_id), 'x': float(x), 'y': float(y), 'conf': 1.0})
            except Exception:
                continue

    if len(rows) == 0:
        return pd.DataFrame(columns=['frame', 'person', 'keypoint_id', 'x', 'y', 'conf'])
    return pd.DataFrame(rows)


def detect_tremor(df_kp, fps=30):
    """
    Compute tremor frequency per person/wrist using FFT on displacement magnitude.
    Returns a DataFrame: ['person', 'wrist_id', 'tremor_freq_hz']
    """
    if df_kp is None or df_kp.empty:
        return pd.DataFrame(columns=['person', 'wrist_id', 'tremor_freq_hz'])

    results = []
    wrist_ids = [9, 10]
    persons = sorted(df_kp['person'].unique())

    for person in persons:
        for wrist in wrist_ids:
            kp = df_kp[(df_kp['person'] == person) & (df_kp['keypoint_id'] == wrist)].sort_values('frame')
            if len(kp) < 4:
                results.append({'person': int(person), 'wrist_id': int(wrist), 'tremor_freq_hz': 0.0})
                continue
            dx = kp['x'].diff().fillna(0).values
            dy = kp['y'].diff().fillna(0).values
            signal = np.sqrt(dx ** 2 + dy ** 2)
            N = len(signal)
            if N < 2:
                results.append({'person': int(person), 'wrist_id': int(wrist), 'tremor_freq_hz': 0.0})
                continue
            yf = rfft(signal)
            xf = rfftfreq(N, 1.0 / fps)
            # Ignore DC component; pick dominant frequency
            if len(yf) <= 1:
                freq = 0.0
            else:
                dominant_idx = np.argmax(np.abs(yf[1:])) + 1
                freq = float(xf[dominant_idx])
            results.append({'person': int(person), 'wrist_id': int(wrist), 'tremor_freq_hz': freq})

    return pd.DataFrame(results)


def analyze_gait(df_kp, fps=30):
    """
    Analyze ankle vertical motion to find step frequency and count.
    Returns (gait_df, symmetry_df)
    gait_df columns: ['person', 'ankle', 'step_frequency_hz', 'num_steps']
    symmetry_df columns: ['person', 'gait_symmetry_%']
    """
    if df_kp is None or df_kp.empty:
        return pd.DataFrame(columns=['person', 'ankle', 'step_frequency_hz', 'num_steps']), pd.DataFrame(columns=['person', 'gait_symmetry_%'])

    ankle_ids = {15: 'Left', 16: 'Right'}
    rows = []
    persons = sorted(df_kp['person'].unique())

    for person in persons:
        per = df_kp[df_kp['person'] == person]
        for aid, name in ankle_ids.items():
            kp = per[per['keypoint_id'] == aid].sort_values('frame')
            if len(kp) < 4:
                rows.append({'person': int(person), 'ankle': name, 'step_frequency_hz': 0.0, 'num_steps': 0})
                continue
            y_vals = kp['y'].values
            frames = kp['frame'].values
            inv_y = -y_vals
            peaks, _ = find_peaks(inv_y, distance=5)
            if len(peaks) > 1:
                step_intervals_frames = np.diff(frames[peaks])
                step_times = step_intervals_frames / float(fps)
                avg_step_time = np.mean(step_times)
                step_freq = 1.0 / avg_step_time if avg_step_time > 0 else 0.0
            else:
                step_freq = 0.0
            rows.append({'person': int(person), 'ankle': name, 'step_frequency_hz': float(step_freq), 'num_steps': int(len(peaks))})

    gait_df = pd.DataFrame(rows)

    # symmetry calculation
    sym_rows = []
    for person in gait_df['person'].unique():
        left_f = gait_df[(gait_df['person'] == person) & (gait_df['ankle'] == 'Left')]['step_frequency_hz'].mean()
        right_f = gait_df[(gait_df['person'] == person) & (gait_df['ankle'] == 'Right')]['step_frequency_hz'].mean()
        left_f = float(left_f) if not np.isnan(left_f) else 0.0
        right_f = float(right_f) if not np.isnan(right_f) else 0.0
        if max(left_f, right_f) > 0:
            gsi = 100.0 * (1.0 - abs(left_f - right_f) / max(left_f, right_f))
        else:
            gsi = 0.0
        sym_rows.append({'person': int(person), 'gait_symmetry_%': round(gsi, 2)})

    symmetry_df = pd.DataFrame(sym_rows)
    return gait_df, symmetry_df


def build_summary(tremor_df, gait_df, symmetry_df):
    """Merge tremor, gait, symmetry into summary DataFrame."""
    if tremor_df is None or tremor_df.empty:
        tremor_df = pd.DataFrame(columns=['person', 'tremor_freq_hz'])
    # average step freq and total steps
    if gait_df is None or gait_df.empty:
        avg_step = pd.DataFrame(columns=['person', 'avg_step_freq_hz'])
        total_steps = pd.DataFrame(columns=['person', 'total_steps'])
    else:
        avg_step = gait_df.groupby('person')['step_frequency_hz'].mean().reset_index().rename(columns={'step_frequency_hz': 'avg_step_freq_hz'})
        total_steps = gait_df.groupby('person')['num_steps'].sum().reset_index().rename(columns={'num_steps': 'total_steps'})

    # prepare tremor average per person (average of both wrists)
    if 'tremor_freq_hz' in tremor_df.columns:
        tremor_avg = tremor_df.groupby('person')['tremor_freq_hz'].mean().reset_index()
    else:
        tremor_avg = pd.DataFrame(columns=['person', 'tremor_freq_hz'])

    # merge safely
    dfs = [tremor_avg, symmetry_df, avg_step, total_steps]
    report = dfs[0]
    for d in dfs[1:]:
        if d is None or d.empty:
            continue
        report = report.merge(d, on='person', how='outer')
    report = report.fillna(0)
    return report


# ---------------------- Synthetic test data generator ----------------------

def generate_synthetic_kp_df(num_persons=3, frames=300, fps=30, random_seed=42):
    """
    Create a synthetic keypoints DataFrame useful for local tests.
    It includes wrists (9,10) and ankles (15,16) for each person.
    Person behaviours:
      - person 0: normal walking + no tremor
      - person 1: faster walking
      - person 2: normal walking + small wrist tremor at 5 Hz
    """
    np.random.seed(random_seed)
    rows = []
    t = np.arange(frames)

    # parameters for persons
    params = [
        {'step_hz': 1.5, 'tremor_hz': 0.0, 'amp_step': 30.0},
        {'step_hz': 2.0, 'tremor_hz': 0.0, 'amp_step': 28.0},
        {'step_hz': 1.2, 'tremor_hz': 5.0, 'amp_step': 25.0},
    ]

    for person in range(num_persons):
        p = params[person % len(params)]
        step_hz = p['step_hz']
        tremor_hz = p['tremor_hz']
        amp = p['amp_step']

        # ankle bases
        base_y = 300 + 10 * person
        base_x = 100 + 40 * person

        # phase offsets so left/right are out of phase
        for frame_idx in t:
            # ankles
            for aid, x_off in zip([15, 16], [-10, 10]):
                phase = 0.0 if aid == 15 else math.pi
                y = base_y + amp * np.sin(2 * math.pi * step_hz * frame_idx / fps + phase) + np.random.randn() * 2.0
                x = base_x + x_off + (frame_idx * 0.5)  # slight forward motion
                rows.append({'frame': int(frame_idx), 'person': int(person), 'keypoint_id': int(aid), 'x': float(x), 'y': float(y), 'conf': 0.9})

            # wrists (swing + optional tremor)
            for wid, x_off in zip([9, 10], [-30, 30]):
                swing_amp = 20.0
                swing = swing_amp * np.sin(2 * math.pi * step_hz * frame_idx / fps)
                tremor = 0.0
                if tremor_hz > 0:
                    tremor = 4.0 * np.sin(2 * math.pi * tremor_hz * frame_idx / fps)
                y_w = base_y - 80 + swing + tremor + np.random.randn() * 1.5
                x_w = base_x + x_off + tremor * 0.2
                rows.append({'frame': int(frame_idx), 'person': int(person), 'keypoint_id': int(wid), 'x': float(x_w), 'y': float(y_w), 'conf': 0.9})

    return pd.DataFrame(rows)


# ---------------------- Runner utilities ----------------------

def run_pipeline_from_kp_df(kp_df, fps=30, save_prefix='neurotrack_run'):
    """Run tremor/gait analysis on a keypoint DataFrame and save results + plots."""
    tremor_df = detect_tremor(kp_df, fps=fps)
    gait_df, symmetry_df = analyze_gait(kp_df, fps=fps)
    summary = build_summary(tremor_df, gait_df, symmetry_df)

    # Add risk flag
    summary['risk_flag'] = ((summary.get('gait_symmetry_%', 0) < 90) |
                            (summary.get('avg_step_freq_hz', 0) < 0.5) |
                            (summary.get('tremor_freq_hz', 0) > 4))

    csv_path = f"{save_prefix}_results.csv"
    summary.to_csv(csv_path, index=False)
    print(f"Saved summary CSV -> {csv_path}")
    print(summary)

    # Simple plots
    try:
        plt.figure(figsize=(8, 4))
        colors = ['red' if f else 'skyblue' for f in summary['risk_flag']]
        plt.bar(summary['person'], summary['gait_symmetry_%'], color=colors)
        plt.axhline(90, color='red', linestyle='--')
        plt.xlabel('Person')
        plt.ylabel('Gait Symmetry (%)')
        plt.title('Gait Symmetry per Person')
        p1 = f"{save_prefix}_gait_symmetry.png"
        plt.savefig(p1, bbox_inches='tight')
        plt.close()
        print(f"Saved plot -> {p1}")

        plt.figure(figsize=(8, 4))
        colors2 = ['red' if f else 'lightgreen' for f in summary['risk_flag']]
        plt.bar(summary['person'], summary['avg_step_freq_hz'], color=colors2)
        plt.axhline(0.5, color='red', linestyle='--')
        plt.xlabel('Person')
        plt.ylabel('Avg Step Frequency (Hz)')
        plt.title('Average Step Frequency per Person')
        p2 = f"{save_prefix}_avg_step_freq.png"
        plt.savefig(p2, bbox_inches='tight')
        plt.close()
        print(f"Saved plot -> {p2}")

    except Exception as e:
        warnings.warn(f"Plotting failed: {e}")

    return summary


def run_pipeline_on_video(video_path, fps=30, save_prefix='neurotrack_video'):
    """
    Run YOLOv8 inference on video_path to get keypoints and analyze.
    Requires `ultralytics` to be installed. If not installed, raises RuntimeError.
    """
    if not HAS_ULTRALYTICS:
        raise RuntimeError("Ultralytics YOLO (ultralytics) is not installed in this environment. Install it or run in test mode.")

    model = YOLO('yolov8n-pose.pt')
    print(f"Running YOLOv8 on {video_path} (this may take a while)...")
    results = model(video_path)
    kp_df = extract_keypoints_df(results)
    print(f"Extracted keypoints: {kp_df.shape}")
    return run_pipeline_from_kp_df(kp_df, fps=fps, save_prefix=save_prefix)


# ---------------------- Entry point ----------------------

if __name__ == '__main__':
    if HAS_STREAMLIT:
        # Run the Streamlit app UI (same analysis functions used)
        st.set_page_config(page_title='NeuroTrack', layout='wide')
        st.title('🧠 NeuroTrack — Gait & Tremor Analysis (YOLOv8-Pose)')

        uploaded = st.file_uploader('Upload a subject video (mp4/mov/avi)', type=['mp4', 'mov', 'avi'])
        if uploaded is not None:
            tmpfile = save_uploaded_file(uploaded, 'input_video.mp4')
            st.video(tmpfile)

            if not HAS_ULTRALYTICS:
                st.error('Ultralytics is not installed in this environment. You can still run the analysis in test mode by running `python neurotrack_app.py`.')
            else:
                with st.spinner('Running YOLOv8-Pose (this may take a while)...'):
                    model = YOLO('yolov8n-pose.pt')
                    results = model(tmpfile)

                st.success('Pose estimation finished')
                kp_df = extract_keypoints_df(results)
                if kp_df.empty:
                    st.warning('No pose keypoints were detected. Try a clearer video (full body, good lighting).')
                else:
                    tremor_df = detect_tremor(kp_df, fps=30)
                    gait_df, symmetry_df = analyze_gait(kp_df, fps=30)
                    summary = build_summary(tremor_df, gait_df, symmetry_df)
                    summary['risk_flag'] = ((summary.get('gait_symmetry_%', 0) < 90) |
                                            (summary.get('avg_step_freq_hz', 0) < 0.5) |
                                            (summary.get('tremor_freq_hz', 0) > 4))
                    st.write('NeuroTrack Summary:')
                    st.dataframe(summary)
                    csv = summary.to_csv(index=False).encode('utf-8')
                    st.download_button('Download results CSV', data=csv, file_name='neurotrack_results.csv', mime='text/csv')

                    # simple plots
                    fig1, ax1 = plt.subplots()
                    colors = ['red' if f else 'skyblue' for f in summary['risk_flag']]
                    ax1.bar(summary['person'], summary['gait_symmetry_%'], color=colors)
                    ax1.axhline(90, color='red', linestyle='--')
                    ax1.set_xlabel('Person')
                    ax1.set_ylabel('Gait Symmetry (%)')
                    st.pyplot(fig1)

        else:
            st.info('Upload a video to run NeuroTrack. If you do not have streamlit installed you can run this script with `python neurotrack_app.py` to run a built-in test.')

    else:
        # Non-Streamlit fallback: run a self-test using synthetic data so the script remains runnable
        print('Streamlit not available in this environment. Running CLI self-test using synthetic keypoints...')
        print('If you want the full web UI, install Streamlit: pip install streamlit')

        # If user passed a video path argument, and ultralytics is available, run on that video
        if len(sys.argv) > 1 and HAS_ULTRALYTICS:
            video_path = sys.argv[1]
            print(f'Running pipeline on provided video: {video_path}')
            try:
                summary = run_pipeline_on_video(video_path, fps=30, save_prefix='neurotrack_video')
                print('Done. Summary:')
                print(summary)
            except Exception as e:
                print('Error running video pipeline:', e)
                print('Falling back to synthetic test...')
                kp_df = generate_synthetic_kp_df(num_persons=3, frames=300, fps=30)
                summary = run_pipeline_from_kp_df(kp_df, fps=30, save_prefix='neurotrack_synthetic')
        else:
            kp_df = generate_synthetic_kp_df(num_persons=3, frames=300, fps=30)
            print('Generated synthetic keypoints DataFrame with shape:', kp_df.shape)
            summary = run_pipeline_from_kp_df(kp_df, fps=30, save_prefix='neurotrack_synthetic')

        print('\nSelf-test complete. Check CSV and PNG files created in the current directory.')
        print('If you want the Streamlit app, install it with: pip install streamlit')
