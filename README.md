# ChordFinder

Real-time major/minor chord recognition from audio, using a two-stage neural network on FFT features.

Play a chord into your microphone and the model shows its name (e.g. `F# Minor`) after about half a second.

## How it works

Each prediction uses a 24,576-sample window (0.512 s at 48 kHz), which gives an FFT bin spacing of exactly 1.953125 Hz. Bins 1–2048 therefore cover 1.95–4000 Hz with no interpolation, fine enough to separate neighbouring semitones, including the major/minor third.

The prediction runs in two stages:

1. **Root network.** (C, C#, D...) The 2048 FFT magnitude bins go into a dense network that predicts the root note (12 classes, C to B).
2. **Quality network.** (Major/Minor) The FFT energy is summed into 12 pitch classes and rotated so the predicted root sits at index 0. Index 3 is then the minor third and index 4 the major third, for every root. A small network (`12 → 32 → 16 → 1`) predicts major vs. minor from that rotated vector.

Because the quality network always sees the same root-relative layout, it learns "minor shape" once instead of 12 separate times. This is what fixed the quality prediction, which was the weak point of the earlier single-network versions.

## Results

On held-out recordings (split by file, never by chunk):

| Metric | Accuracy |
|---|---|
| Root | 98.8% |
| Quality, given the true root | 100% |
| Quality, given the predicted root | 98.8% |
| Whole chord (root and quality) | 98.8% |

The split keeps all windows from a recording on one side, so there is no leakage between neighbouring chunks. Other recordings of the same instrument can still appear in training, so accuracy on a completely new instrument may be lower. Set `GROUP_BY = 'instrument'` in the training script to test that.

## Files

| File | Purpose |
|---|---|
| `chord_cascade_pipeline.py` | Builds the dataset, trains both models, and prints the evaluation |
| `Final_Cascade.py` | Live microphone predictor with a chord display window |

## Dataset
I couldn't get hold of the perfect dataset for training so I created my own dataset:
Training uses [Wisteria717/Chords_Dataset](https://github.com/Wisteria717/Chords_Dataset) (Git LFS), which contains major and minor triads in root position and first and second inversion, played on several instruments (piano, organ, accordion, acoustic and metal guitar, synth bass). Each file holds 24 chords at 120 BPM: three beats played, then one beat of rest.

## Setup

```bash
pip install numpy librosa tensorflow scikit-learn sounddevice matplotlib
```

## Usage

**Train** (written for Google Colab, so adjust `dataset_dir` for local use):

```run
ChordFinder.ipynb
```

This saves `chord_root_model.keras` and `chord_quality_model.keras`.

**Run live:** put both model files next to the script (same folder), then:

```bash (or first activate your venv if a new one was created)
python Final_Cascade.py
```

Close the window or press Ctrl+C to stop.

## Limitations

- Major and minor triads only (no 7ths, sus, diminished, etc.).
- Assumes 48 kHz audio and A = 440 Hz tuning.
- About 0.5 s latency, a consequence of the long window needed for 1.95 Hz resolution.
- Root detection ignores energy below 100 Hz, so very low notes are identified through their harmonics.
- `RMS_THRESHOLD` in `Final_Cascade.py` may need adjusting for your microphone level.

## Training-script options

- `GROUP_BY`: `'file'` (default) or `'instrument'` for a stricter held-out test.
- `PC_FMIN`: lowest frequency (Hz) used for the pitch-class map.
