import pandas as pd
import matplotlib.pyplot as plt

gpu_data = pd.read_csv('./image_signal_processing/fft_gpu_manual.csv')
cufft_data = pd.read_csv('./image_signal_processing/fft_cufft.csv')

N = len(gpu_data)
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

ax1.semilogy(gpu_data['Frequency'][:N//2], gpu_data['Magnitude'][:N//2])
ax1.set_xlabel('Frequency (Hz)')
ax1.set_ylabel('Magnitude')
ax1.set_title('GPU Manual - Magnitude Spectrum')
ax1.grid(True)

ax2.semilogy(cufft_data['Frequency'][:N//2], cufft_data['Magnitude'][:N//2])
ax2.set_xlabel('Frequency (Hz)')
ax2.set_ylabel('Magnitude')
ax2.set_title('cuFFT - Magnitude Spectrum')
ax2.grid(True)

ax3.plot(gpu_data['Frequency'][:N//2], gpu_data['Phase'][:N//2])
ax3.set_xlabel('Frequency (Hz)')
ax3.set_ylabel('Phase (radians)')
ax3.set_title('GPU Manual - Phase Spectrum')
ax3.grid(True)

ax4.plot(cufft_data['Frequency'][:N//2], cufft_data['Phase'][:N//2])
ax4.set_xlabel('Frequency (Hz)')
ax4.set_ylabel('Phase (radians)')
ax4.set_title('cuFFT - Phase Spectrum')
ax4.grid(True)

plt.tight_layout()
plt.savefig('./image_signal_processing/fft_comparison.jpg', dpi=300, bbox_inches='tight')
plt.close()