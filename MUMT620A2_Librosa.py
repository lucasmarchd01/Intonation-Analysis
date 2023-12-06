import librosa
import matplotlib.pyplot as plt
import numpy as np

y, sr = librosa.load("Sean_brahms.wav")
o_env = librosa.onset.onset_strength(y=y, sr=sr)
times = librosa.times_like(o_env, sr=sr)
onset_frames = librosa.onset.onset_detect(onset_envelope=o_env, sr=sr)
D = np.abs(librosa.stft(y))

fig, ax = plt.subplots(nrows=2, sharex=True)
librosa.display.specshow(
    librosa.amplitude_to_db(D, ref=np.max), x_axis="time", y_axis="log", ax=ax[0]
)
ax[0].set(title="Power spectrogram")
ax[0].label_outer()
ax[1].plot(times, o_env, label="Onset strength")
ax[1].vlines(
    times[onset_frames],
    0,
    o_env.max(),
    color="r",
    alpha=0.9,
    linestyle="--",
    label="Onsets",
)
ax[1].legend()

f0, voiced_flag, voiced_probs = librosa.pyin(
    y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7")
)
times = librosa.times_like(f0)
D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)

fig, ax = plt.subplots()
img = librosa.display.specshow(D, x_axis="time", y_axis="log", ax=ax)
ax.set(title="pYIN fundamental frequency estimation")
fig.colorbar(img, ax=ax, format="%+2.f dB")
ax.plot(times, f0, label="f0", color="cyan", linewidth=3)
ax.legend(loc="upper right")

S = np.abs(librosa.stft(y))
pitches, magnitudes = librosa.piptrack(S=S, sr=sr)
pitch = []
for i in range(magnitudes.shape[1]):
    index = magnitudes[:, i].argmax()
    pitch.append(pitches[index, i])

# print("pitch tuning offset:", librosa.pitch_tuning(pitches))
# print("pitchmean:", np.mean(pitch))
# print("pitchstd:", np.std(pitch))
# print("pitchmax:", np.max(pitch))
# print("pitchmin:", np.min(pitch))


f0, voiced_flag, voiced_probs = librosa.pyin(
    y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7")
)
times = librosa.times_like(f0)


tuning_deviations = []
# Apply pitch_tuning at each time step
for pitch in f0:
    deviation = librosa.pitch_tuning(pitch)
    tuning_deviations.append(deviation)

tuning_deviations_cents = np.array(tuning_deviations) * 100

# Calculate statistics
print("mean_deviation", np.mean(tuning_deviations_cents))
print("std_deviation", np.std(tuning_deviations_cents))
print("max_deviation", np.max(tuning_deviations_cents))
print("min_deviation", np.min(tuning_deviations_cents))

interval_duration = 5.0

num_bins = int(np.ceil(times[-1] / interval_duration))

tuning_deviations_per_interval = []

# Split the tuning deviations into intervals
for i in range(num_bins):
    start_time = i * interval_duration
    end_time = (i + 1) * interval_duration
    interval_indices = np.where((times >= start_time) & (times < end_time))
    deviations_in_interval = tuning_deviations_cents[interval_indices]
    tuning_deviations_per_interval.append(deviations_in_interval)

# Create a figure to display both the box plot and the line plot
plt.figure(figsize=(12, 8))

# Plot the box plot
plt.subplot(1, 2, 1)
plt.boxplot(tuning_deviations_per_interval, labels=range(1, num_bins + 1))
plt.xlabel("Time Interval")
plt.ylabel("Tuning Deviation (cents)")
plt.title("Pitch Deviation in Time Intervals (Box Plot)")


interval_duration = 1.0

num_bins = int(np.ceil(times[-1] / interval_duration))

tuning_deviations_per_interval = []

# Split the tuning deviations into intervals
for i in range(num_bins):
    start_time = i * interval_duration
    end_time = (i + 1) * interval_duration
    interval_indices = np.where((times >= start_time) & (times < end_time))
    deviations_in_interval = tuning_deviations_cents[interval_indices]
    tuning_deviations_per_interval.append(deviations_in_interval)

# Plot the line plot
plt.subplot(1, 2, 2)
bin_centers = np.arange(num_bins) * interval_duration + interval_duration / 2.0
average_deviations = [
    np.mean(deviations) for deviations in tuning_deviations_per_interval
]
plt.plot(bin_centers, average_deviations, marker="o", linestyle="-")
plt.xlabel("Time (s)")
plt.ylabel("Average Tuning Deviation (cents)")
plt.title("Average Pitch Deviation Over Time Intervals (Line Plot)")

plt.tight_layout()
plt.show()

# Ensure times and tuning_deviations have the same length
times = times[: len(tuning_deviations_cents)]

# Plot tuning deviations over time
plt.figure(figsize=(12, 8))
plt.plot(times, tuning_deviations_cents, label="Tuning Deviation (cents)")
plt.xlabel("Time (s)")
plt.ylabel("Tuning Deviation (cents)")
plt.title("Pitch Deviation Over Time")
plt.legend()
plt.grid()
plt.show()
