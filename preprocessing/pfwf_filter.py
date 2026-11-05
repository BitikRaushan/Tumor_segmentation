"""
Probabilistic Fused Wiener Filter (PFWF / PHWF)
Reconstructed from BSPC 2025 Paper (Table 2 & Section 3.1)
"""
import numpy as np
from scipy.ndimage import uniform_filter

class ProbabilisticWienerFilter:
    def __init__(self, noise_threshold: float = 20.0):
        self.noise_threshold = noise_threshold

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        Args:
            image: 2D numpy array with values in [0, 255] or [0, 1]
        Returns:
            Denoised 2D numpy array
        """
        orig_dtype = image.dtype
        img = image.astype(np.float32)
        
        # Scale to [0, 255] if normalized
        is_normalized = img.max() <= 1.0
        if is_normalized:
            img = img * 255.0

        # Step 2: Evaluate local noise standard deviation via high-pass Laplacian kernel
        laplacian_mask = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.float32)
        from scipy.signal import convolve2d
        lap = convolve2d(img, laplacian_mask, mode='same', boundary='symm')
        sigma_gm = np.mean(np.abs(lap))

        # Step 3: Select kernel window size based on noise threshold
        w_size = 3 if sigma_gm < self.noise_threshold else 5

        # Step 4: Local mean and variance estimation
        local_mean = uniform_filter(img, size=w_size)
        local_sqr_mean = uniform_filter(img ** 2, size=w_size)
        local_var = np.maximum(local_sqr_mean - local_mean ** 2, 0)

        # Estimate global noise variance (average of lowest 10% local variances)
        noise_var = np.mean(np.sort(local_var.flatten())[:int(local_var.size * 0.1)])
        noise_var = max(noise_var, 1e-4)

        # Standard Wiener filter formulation
        wiener_out = local_mean + (np.maximum(local_var - noise_var, 0) / (local_var + 1e-8)) * (img - local_mean)

        # Step 5 & 6: Sliding window difference and uniform probabilistic blending
        diff = np.abs(img - local_mean)
        mu_diff = np.mean(diff)
        
        # Probabilistic mask based on uniform variance condition
        prob_weight = np.where(diff < mu_diff, 0.7, 0.3)
        denoised = prob_weight * wiener_out + (1.0 - prob_weight) * local_mean

        if is_normalized:
            denoised = denoised / 255.0

        return denoised.astype(orig_dtype)