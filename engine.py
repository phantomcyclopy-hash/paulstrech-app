import numpy as np


class PaulStretchEngine:
    def __init__(
        self,
        samplerate,
        stretch=8.0,
        window_size=0.25
    ):
        self.samplerate = samplerate
        self.stretch = stretch
        self.window_size = window_size
        self.cancelled = False

    def cancel(self):
        self.cancelled = True

    def process(
        self,
        samples,
        progress_callback=None
    ):
        samples = np.asarray(
            samples,
            dtype=np.float32
        )

        if len(samples) == 0:
            raise ValueError(
                "Áudio vazio."
            )

        window_size_samples = int(
            self.window_size * self.samplerate
        )

        if window_size_samples < 16:
            window_size_samples = 16

        if window_size_samples % 2:
            window_size_samples += 1

        if len(samples) < window_size_samples:
            raise ValueError(
                "Áudio menor que a janela selecionada."
            )

        half_window = (
            window_size_samples // 2
        )

        analysis_hop = max(
            1,
            int(
                half_window /
                self.stretch
            )
        )

        synthesis_hop = (
            half_window
        )

        window = np.hanning(
            window_size_samples
        )

        estimated_length = int(
            len(samples) *
            self.stretch
        ) + window_size_samples

        output = np.zeros(
            estimated_length,
            dtype=np.float32
        )

        input_pos = 0
        output_pos = 0

        total = len(samples)

        while (
            input_pos +
            window_size_samples
            < total
        ):

            if self.cancelled:
                return None

            chunk = (
                samples[
                    input_pos:
                    input_pos +
                    window_size_samples
                ]
                * window
            )

            spectrum = np.fft.rfft(
                chunk
            )

            magnitude = np.abs(
                spectrum
            )

            random_phase = np.random.uniform(
                0,
                2 * np.pi,
                len(magnitude)
            )

            spectrum_stretched = (
                magnitude *
                np.exp(
                    1j *
                    random_phase
                )
            )

            stretched_chunk = (
                np.fft.irfft(
                    spectrum_stretched
                )
            )

            stretched_chunk *= window

            end_pos = (
                output_pos +
                window_size_samples
            )

            if end_pos > len(output):
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
                        input_pos /
                        total
                    )
                )

        output = output[
            :output_pos +
            window_size_samples
        ]

        peak = np.max(
            np.abs(output)
        )

        if peak > 0:
            output /= peak

        return output.astype(
            np.float32
        )
