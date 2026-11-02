from collections import deque

import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
import tensorflow as tf


# --- 1. MODEL DESERIALIZATION FIX ---
class CustomDense(tf.keras.layers.Dense):
    """Bypasses 'quantization_config' serialization errors across Keras versions."""
    def __init__(self, *args, quantization_config=None, **kwargs):
        super().__init__(*args, **kwargs)


# NEW: two models instead of one
root_model = tf.keras.models.load_model(
    "chord_root_model.keras",
    custom_objects={'Dense': CustomDense},
    compile=False
)
quality_model = tf.keras.models.load_model(
    "chord_quality_model.keras",
    custom_objects={'Dense': CustomDense},
    compile=False
)

quality_classes = ['Major', 'Minor']          # sigmoid output: < 0.5 = Major, >= 0.5 = Minor
ROOT_CLASSES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# --- Audio / FFT configuration (must match training exactly) ---
RATE = 48000
N = 24576                 # FFT window -> bin spacing 1.953125 Hz
N_BINS = 2048             # bins 1..2048 = 1.95 Hz .. 4000 Hz
BLOCK = 4096              # samples read per step (~85 ms); N / BLOCK = 6 blocks
RMS_THRESHOLD = 0.02
SMOOTHING_WINDOW = 3      # windows overlap heavily, so little smoothing is needed

window = np.hanning(N).astype(np.float32)
bin_freqs = np.arange(1, N_BINS + 1) * RATE / N

# NEW: bin -> pitch-class map (identical to training)
PC_FMIN = 100.0
semitones_from_c4 = 12 * np.log2(bin_freqs / 261.6256)
pc_of_bin = np.round(semitones_from_c4).astype(int) % 12
valid = bin_freqs >= PC_FMIN
P = np.zeros((N_BINS, 12), dtype=np.float32)
P[np.arange(N_BINS)[valid], pc_of_bin[valid]] = 1.0


def fft_features(seg):
    """Identical to the training feature function."""
    spec = np.abs(np.fft.rfft(seg * window))[1:N_BINS + 1]
    spec = spec / max(spec.max(), 1e-8)
    spec = np.log1p(100.0 * spec) / np.log1p(100.0)
    return spec.astype(np.float32)


# NEW: identical to training
def pc_features(seg):
    """12 values: spectral energy at C, C#, ... B (all octaves folded together)."""
    mag = np.abs(np.fft.rfft(seg * window))[1:N_BINS + 1]
    pc = mag @ P
    pc = pc / max(pc.max(), 1e-8)
    pc = np.log1p(100.0 * pc) / np.log1p(100.0)
    return pc.astype(np.float32)


# NEW: identical to training
def root_relative(pc, root):
    """Rotate pitch classes so the root is at index 0 (3 = minor third, 4 = major third, 7 = fifth)."""
    idx = (np.asarray(root)[:, None] + np.arange(12)[None, :]) % 12
    return np.take_along_axis(pc, idx, axis=1)


# NEW: smoothing keeps the root probabilities AND the pitch-class vectors,
# so the quality is always computed with the smoothed root, not a stale one.
root_history = deque(maxlen=SMOOTHING_WINDOW)
pc_history = deque(maxlen=SMOOTHING_WINDOW)

# Rolling buffer holding the most recent N samples
buffer = np.zeros(N, dtype=np.float32)
filled = 0                # samples of continuous non-silent audio collected so far

# Warm up both models
root_model(np.zeros((1, N_BINS), dtype=np.float32), training=False)
quality_model(np.zeros((1, 12), dtype=np.float32), training=False)


def predict_chord(audio_window):
    # Stage 1: root from the FFT
    root_probs = root_model(fft_features(audio_window)[None, :], training=False).numpy()[0]
    root_history.append(root_probs)
    pc_history.append(pc_features(audio_window))

    root_idx = int(np.argmax(np.mean(root_history, axis=0)))

    # Stage 2: quality from the root-relative pitch classes
    avg_pc = np.mean(pc_history, axis=0)[None, :]
    rel = root_relative(avg_pc, np.array([root_idx]))
    q_prob = float(quality_model(rel, training=False).numpy()[0][0])

    q_idx = int(q_prob > 0.5)            # single sigmoid unit -> threshold, NOT argmax
    return f"{ROOT_CLASSES[root_idx]} {quality_classes[q_idx]}"


stream = sd.InputStream(
    samplerate=RATE,
    channels=1,
    dtype='float32',
    blocksize=BLOCK
)
stream.start()

# --- Window with only the chord display ---
fig, ax = plt.subplots(figsize=(5, 5))
custom_font = 'Cambria'
ax.set_title("Predicted Chord (Cascade)", fontsize=14, fontname=custom_font, fontweight='bold')
ax.set_xticks([])
ax.set_yticks([])
ax.grid(False)

chord_text_display = ax.text(
    0.5, 0.5, '...',
    fontname=custom_font, size=40,
    ha='center', va='center', weight='bold', color='#888888'
)

plt.tight_layout()
plt.ion()
fig.show()

is_running = True


def on_close(event):
    global is_running
    is_running = False


fig.canvas.mpl_connect('close_event', on_close)

print("\n--- Running Real-Time Cascade Chord Predictor ---")

last_chord = None

try:
    while is_running and plt.fignum_exists(fig.number):
        data, overflowed = stream.read(BLOCK)
        block = data[:, 0]

        rms_energy = np.sqrt(np.mean(block ** 2))

        if rms_energy < RMS_THRESHOLD:
            # Silence: wipe everything so the old chord can't bleed into the next one
            chord = "..."
            buffer[:] = 0.0
            filled = 0
            root_history.clear()
            pc_history.clear()
        else:
            # Slide the window forward by one block
            buffer = np.concatenate([buffer[BLOCK:], block])
            filled += BLOCK

            if filled < N:
                # Window not yet full of real audio. A zero-padded partial window
                # has poor frequency resolution, so wait (~0.5 s after onset).
                chord = "..."
            else:
                chord = predict_chord(buffer)

        if chord != last_chord:
            last_chord = chord
            chord_text_display.set_text(chord)
            chord_text_display.set_color('#888888' if chord == "..." else '#d62728')
            fig.canvas.draw_idle()

        plt.pause(0.001)

except KeyboardInterrupt:
    print("\nStopping visualizer...")

finally:
    stream.stop()
    stream.close()
    plt.close('all')
    print("Resources cleanly released.")
