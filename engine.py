import numpy as np


class PaulStretchEngine:
    def __init__(
        self,
        samplerate,
        stretch=8.0,
        window_size=0.25,
        randomness=1.0,
        normalize=True,
    ):
        self.samplerate = samplerate
        self.stretch = max(1.0, float(stretch))
        self.window_size = max(0.01, float(window_size))
        self.randomness = max(
            0.0,
            min(1.0, float(randomness))
        )
        self.normalize = normalize
        self.cancelled = False

    def cancel(self):
        self.cancelled = True

    def _next_power_of_two(self, n):
        return 1 << (n - 1).bit_length()

    def _process_mono(
        self,
        samples,
        progress_callback=None
    ):
        samples = np.asarray(
            samples,
            dtype=np.float32
        )

        if samples.size == 0:
            raise ValueError(
                "Áudio vazio."
            )

        requested_window = int(
            self.window_size *
            self.samplerate
        )

        fft_size = self._next_power_of_two(
            max(1024, requested_window)
        )

        if fft_size > 131072:
            fft_size = 131072

        if samples.size < fft_size:
            raise ValueError(
                "Áudio menor que a janela."
            )

        half_window = fft_size // 2

        analysis_hop = max(
            1,
            int(
                half_window /
                self.stretch
            )
        )

        synthesis_hop = half_window

        window = np.hanning(
            fft_size
        ).astype(np.float32)

        estimated_length = int(
            samples.size *
            self.stretch
        ) + fft_size

        estimated_mb = (
            estimated_length * 4
        ) / (1024 * 1024)

        if estimated_mb > 512:
            raise MemoryError(
                f"Renderização exigiria "
                f"aprox. {estimated_mb:.0f} MB."
            )

        output = np.zeros(
            estimated_length,
            dtype=np.float32
        )

        input_pos = 0
        output_pos = 0

        total = samples.size

        while (
            input_pos + fft_size
            < total
        ):
            if self.cancelled:
                return None

            chunk = (
                samples[
                    input_pos:
                    input_pos + fft_size
                ]
                * window
            )

            spectrum = np.fft.rfft(
                chunk
            )

            magnitude = np.abs(
                spectrum
            )

            original_phase = np.angle(
                spectrum
            )

            random_phase = np.random.uniform(
                0,
                2 * np.pi,
                len(magnitude)
            )

            phase = (
                original_phase
                * (1.0 - self.randomness)
                +
                random_phase
                * self.randomness
            )

            stretched_spectrum = (
                magnitude
                * np.exp(1j * phase)
            )

            stretched_chunk = (
                np.fft.irfft(
                    stretched_spectrum
                )
            )

            stretched_chunk *= window

            end_pos = (
                output_pos +
                fft_size
            )

            if end_pos > output.size:
                break

            output[
                output_pos:end_pos
            ] += stretched_chunk

            input_pos += analysis_hop
            output_pos += synthesis_hop

            if progress_callback:
                progress_callback(
                    min(
                        1.0,
                        input_pos / total
                    )
                )

        output = output[
            :output_pos + fft_size
        ]

        if self.normalize:
            peak = np.max(
                np.abs(output)
            )

            if peak > 0:
                output /= peak

        return output.astype(
            np.float32
        )

    def process(
        self,
        samples,
        progress_callback=None
    ):
        samples = np.asarray(
            samples,
            dtype=np.float32
        )

        if samples.ndim == 1:
            return self._process_mono(
                samples,
                progress_callback
            )

        if samples.ndim == 2:

            left = self._process_mono(
                samples[:, 0],
                progress_callback
            )

            if left is None:
                return None

            right = self._process_mono(
                samples[:, 1]
            )

            if right is None:
                return None

            min_len = min(
                len(left),
                len(right)
            )

            return np.column_stack(
                (
                    left[:min_len],
                    right[:min_len]
                )
            )

        raise ValueError(
            "Formato de áudio inválido."
        )
