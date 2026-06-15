import numpy as np

class PaulStretchEngine:
    def __init__(
        self,
        samplerate,
        stretch=8.0,
        window_size=0.25,
        normalize=True,
    ):
        self.samplerate = samplerate
        self.stretch = max(1.0, float(stretch))
        self.window_size = max(0.01, float(window_size))
        self.normalize = normalize
        self.cancelled = False

    def cancel(self):
        self.cancelled = True

    def _next_power_of_two(self, n):
        return 1 << (n - 1).bit_length()

    def process_stream(self, samples):
        """
        Gerador que processa o áudio em blocos (Streaming).
        Retorna (yield) cada pedaço pronto para a pilha de processamento,
        evitando alocações massivas de uma só vez na RAM.
        """
        samples = np.asarray(samples, dtype=np.float32)

        if samples.size == 0:
            raise ValueError("Áudio vazio.")

        requested_window = int(self.window_size * self.samplerate)
        
        # Limitação da FFT para 65536 (Otimização Android / Termux)
        fft_size = self._next_power_of_two(max(1024, requested_window))
        if fft_size > 65536:
            fft_size = 65536

        if samples.size < fft_size:
            raise ValueError("Áudio menor que a janela de análise.")

        half_window = fft_size // 2
        analysis_hop = max(1, int(half_window / self.stretch))
        synthesis_hop = half_window

        window = np.hanning(fft_size).astype(np.float32)

        input_pos = 0
        total = samples.size

        # Buffer local para o overlap-add do bloco atual
        buffer_out = np.zeros(fft_size, dtype=np.float32)

        while input_pos + fft_size < total:
            if self.cancelled:
                yield None
                return

            chunk = samples[input_pos:input_pos + fft_size] * window
            spectrum = np.fft.rfft(chunk)
            magnitude = np.abs(spectrum)

            # Fase 100% aleatória para fidelidade ao PaulStretch clássico
            random_phase = np.random.uniform(0, 2 * np.pi, len(magnitude))
            stretched_spectrum = magnitude * np.exp(1j * random_phase)

            stretched_chunk = np.fft.irfft(stretched_spectrum)
            
            # Ajuste de ganho de 0.5 no overlap-add para estabilizar a amplitude
            stretched_chunk *= window * 0.5

            # Acumula no buffer de saída
            buffer_out += stretched_chunk

            # Separa a parte que já está concluída (que não receberá mais overlap)
            yield_chunk = buffer_out[:synthesis_hop].copy()

            # Desloca o buffer para a próxima iteração
            buffer_out = np.concatenate([buffer_out[synthesis_hop:], np.zeros(synthesis_hop, dtype=np.float32)])

            input_pos += analysis_hop
            progress = min(1.0, input_pos / total)
            
            yield yield_chunk, progress

        # Entrega o resíduo que sobrou no buffer após o término do loop
        yield buffer_out, 1.0
