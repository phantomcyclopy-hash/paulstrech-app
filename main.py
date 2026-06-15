import os
import threading
import numpy as np
import scipy.io.wavfile as wav
import flet as ft
from pydub import AudioSegment

# --- MOTOR MATEMÁTICO DO PAULSTRETCH ---
def paulstretch(samplerate, smp, stretch, windowsize_seconds=0.25):
    n = len(smp)
    n_window = int(windowsize_seconds * samplerate)
    if n_window % 2 == 1: n_window += 1
    half_n_window = n_window // 2
    
    hop = int(half_n_window / stretch)
    if hop == 0: hop = 1
    
    window = np.hanning(n_window)
    out_len = int(n * stretch) + n_window
    output = np.zeros(out_len)
    
    pos = 0
    out_pos = 0
    while pos + n_window < n:
        chunk = smp[pos:pos+n_window] * window
        freq = np.fft.rfft(chunk)
        phases = np.random.uniform(0, 2 * np.pi, len(freq))
        freq = np.abs(freq) * np.exp(1j * phases)
        
        chunk_out = np.fft.irfft(freq) * window
        output[out_pos:out_pos+n_window] += chunk_out
        
        pos += hop
        out_pos += half_n_window
        
    return output[:int(n*stretch)]

# --- INTERFACE DO APLICATIVO ---
def main(page: ft.Page):
    page.title = "PaulStretch Studio"
    page.theme_mode = ft.ThemeMode.DARK
    page.vertical_alignment = ft.MainAxisAlignment.CENTER
    page.horizontal_alignment = ft.CrossAxisAlignment.CENTER
    page.padding = 20

    selected_file_path = None

    # Componentes Visuais da Interface Moderna
    title = ft.Text("PaulStretch Studio", size=26, weight=ft.FontWeight.BOLD, color=ft.Colors.BLUE_400)
    subtitle = ft.Text("Texturas Espectrais Portáteis", size=14, color=ft.Colors.GREY_400)
    
    lbl_file = ft.Text("Nenhum arquivo selecionado", color=ft.Colors.GREY_500, italic=True, text_align=ft.TextAlign.CENTER)
    
    txt_stretch = ft.TextField(label="Fator de Esticamento", value="10.0", width=280, keyboard_type=ft.KeyboardType.NUMBER)
    txt_window = ft.TextField(label="Tamanho da Janela (segundos)", value="0.25", width=280, keyboard_type=ft.KeyboardType.NUMBER)
    txt_output_name = ft.TextField(label="Nome do arquivo de saída", value="resultado_espectral", width=280)
    
    btn_process = ft.ElevatedButton("Esticar Áudio", icon=ft.Icons.AUDIO_FILE, disabled=True, bgcolor=ft.Colors.BLUE_700, color=ft.Colors.WHITE, width=220)
    progress_ring = ft.ProgressRing(visible=False)
    lbl_status = ft.Text("", size=14, weight=ft.FontWeight.W_500, text_align=ft.TextAlign.CENTER)

    def process_audio_thread(file_path, stretch_factor, window_size, out_name):
        try:
            lbl_status.value = "A abrir e converter o áudio..."
            page.update()
            
            # Pydub lê os formatos suportados usando o decoder nativo do APK
            audio = AudioSegment.from_file(file_path)
            samplerate = audio.frame_rate
            
            # Converte os dados brutos para array do numpy
            samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
            
            # Normalização de ganho
            max_val = np.max(np.abs(samples)) if len(samples) > 0 else 1
            if max_val == 0: max_val = 1
            samples /= max_val
            
            # Processamento de canais (Estéreo vs Mono)
            if audio.channels > 1:
                lbl_status.value = "A processar canais Estéreo...\n(Isto pode levar alguns segundos)"
                page.update()
                samples = samples.reshape((-1, audio.channels))
                left = paulstretch(samplerate, samples[:, 0], stretch_factor, window_size)
                right = paulstretch(samplerate, samples[:, 1], stretch_factor, window_size)
                out_data = np.vstack((left, right)).T
            else:
                lbl_status.value = "A processar canal Mono..."
                page.update()
                out_data = paulstretch(samplerate, samples, stretch_factor, window_size)
            
            # Pasta de Downloads pública do Android para não dar erro de permissão restrita
            output_dir = "/storage/emulated/0/Download"
            if not os.path.exists(output_dir):
                output_dir = os.path.dirname(file_path)
                
            final_path = os.path.join(output_dir, f"{out_name}.wav")
            
            lbl_status.value = "A renderizar o ficheiro WAV final..."
            page.update()
            
            # Exportação estável convertendo para 16-bit PCM
            wav.write(final_path, samplerate, (out_data * 32767).astype(np.int16))
            
            lbl_status.value = f"Sucesso!\nGuardado em Downloads como:\n{out_name}.wav"
            lbl_status.color = ft.Colors.GREEN_400
        except Exception as e:
            lbl_status.value = f"Erro: {str(e)}"
            lbl_status.color = ft.Colors.RED_400
        finally:
            progress_ring.visible = False
            btn_process.disabled = False
            page.update()

    def on_process_click(e):
        btn_process.disabled = True
        progress_ring.visible = True
        lbl_status.value = "A iniciar motor espectral..."
        lbl_status.color = ft.Colors.WHITE
        page.update()
        
        stretch_val = float(txt_stretch.value)
        window_val = float(txt_window.value)
        output_val = txt_output_name.value.strip()
        
        threading.Thread(target=process_audio_thread, args=(selected_file_path, stretch_val, window_val, output_val), daemon=True).start()

    def on_file_picker_result(e: ft.FilePickerResultEvent):
        nonlocal selected_file_path
        if e.files:
            selected_file_path = e.files[0].path
            lbl_file.value = f"Selecionado: {e.files[0].name}"
            lbl_file.color = ft.Colors.BLUE_200
            btn_process.disabled = False
            
            base_name = os.path.splitext(e.files[0].name)[0]
            txt_output_name.value = f"{base_name}_spectral"
        else:
            lbl_file.value = "Seleção cancelada"
            btn_process.disabled = True
        page.update()

    file_picker = ft.FilePicker(on_result=on_file_picker_result)
    page.overlay.append(file_picker)

    btn_select = ft.OutlinedButton(
        "Selecionar Ficheiro de Áudio",
        icon=ft.Icons.FOLDER_OPEN,
        on_click=lambda _: file_picker.pick_files(allow_multiple=False)
    )
    
    btn_process.on_click = on_process_click

    page.add(
        ft.Card(
            content=ft.Container(
                content=ft.Column(
                    [
                        title,
                        subtitle,
                        ft.Divider(height=15, color=ft.Colors.TRANSPARENT),
                        btn_select,
                        lbl_file,
                        ft.Divider(height=10, color=ft.Colors.TRANSPARENT),
                        txt_stretch,
                        txt_window,
                        txt_output_name,
                        ft.Divider(height=15, color=ft.Colors.TRANSPARENT),
                        btn_process,
                        progress_ring,
                        lbl_status
                    ],
                    horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                    spacing=12
                ),
                padding=20
            )
        )
    )

ft.app(target=main)
