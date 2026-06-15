import os
import threading
import numpy as np
import scipy.io.wavfile as wav
import flet as ft

from pydub import AudioSegment
from engine import PaulStretchEngine


def format_time(seconds):
    seconds = int(seconds)

    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60

    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"

    return f"{m:02d}:{s:02d}"


def main(page: ft.Page):
    page.title = "PaulStretch Studio"
    page.theme_mode = ft.ThemeMode.DARK
    page.padding = 20
    page.window_width = 500

    selected_file = None
    current_engine = None

    # -------------------------
    # UI
    # -------------------------

    title = ft.Text(
        "PaulStretch Studio",
        size=28,
        weight=ft.FontWeight.BOLD
    )

    subtitle = ft.Text(
        "Extreme Spectral Time Stretching"
    )

    file_name = ft.Text(
        "Nenhum arquivo selecionado"
    )

    info_duration = ft.Text("-")
    info_samplerate = ft.Text("-")
    info_channels = ft.Text("-")

    stretch_field = ft.TextField(
        label="Stretch",
        value="10",
        keyboard_type=ft.KeyboardType.NUMBER
    )

    window_field = ft.TextField(
        label="Window Size (segundos)",
        value="0.25",
        keyboard_type=ft.KeyboardType.NUMBER
    )

    output_field = ft.TextField(
        label="Nome do arquivo"
    )

    estimated_duration = ft.Text(
        "Duração estimada: -"
    )

    progress_bar = ft.ProgressBar(
        value=0,
        visible=False
    )

    status_text = ft.Text()

    process_button = ft.ElevatedButton(
        "Processar",
        disabled=True
    )

    cancel_button = ft.OutlinedButton(
        "Cancelar",
        visible=False
    )

    # -------------------------
    # Atualiza duração prevista
    # -------------------------

    def update_estimation(e=None):
        try:
            if not selected_file:
                return

            audio = AudioSegment.from_file(
                selected_file
            )

            stretch = float(
                stretch_field.value
            )

            final_duration = (
                audio.duration_seconds
                * stretch
            )

            estimated_duration.value = (
                f"Duração estimada: "
                f"{format_time(final_duration)}"
            )

            page.update()

        except:
            pass

    stretch_field.on_change = update_estimation

    # -------------------------
    # File Picker
    # -------------------------

    def file_selected(e):
        nonlocal selected_file

        if not e.files:
            return

        selected_file = e.files[0].path

        audio = AudioSegment.from_file(
            selected_file
        )

        file_name.value = e.files[0].name

        info_duration.value = format_time(
            audio.duration_seconds
        )

        info_samplerate.value = (
            f"{audio.frame_rate} Hz"
        )

        info_channels.value = (
            str(audio.channels)
        )

        output_field.value = (
            os.path.splitext(
                e.files[0].name
            )[0]
            + "_stretched"
        )

        process_button.disabled = False

        update_estimation()

        page.update()

    picker = ft.FilePicker(
        on_result=file_selected
    )

    page.overlay.append(picker)

    select_button = ft.ElevatedButton(
        "Selecionar Áudio",
        icon=ft.Icons.FOLDER_OPEN,
        on_click=lambda _:
        picker.pick_files(
            allow_multiple=False
        )
    )

    # -------------------------
    # Processamento
    # -------------------------

    def update_progress(value):
        progress_bar.value = value
        page.update()

    def process_thread():
        nonlocal current_engine

        try:
            audio = AudioSegment.from_file(
                selected_file
            )

            samples = np.array(
                audio.get_array_of_samples(),
                dtype=np.float32
            )

            max_val = np.max(
                np.abs(samples)
            )

            if max_val > 0:
                samples /= max_val

            stretch = float(
                stretch_field.value
            )

            window_size = float(
                window_field.value
            )

            current_engine = (
                PaulStretchEngine(
                    samplerate=audio.frame_rate,
                    stretch=stretch,
                    window_size=window_size
                )
            )

            status_text.value = (
                "Processando..."
            )

            page.update()

            if audio.channels == 2:

                samples = samples.reshape(
                    (-1, 2)
                )

                left = current_engine.process(
                    samples[:, 0],
                    update_progress
                )

                right = current_engine.process(
                    samples[:, 1]
                )

                if left is None:
                    return

                output = np.column_stack(
                    (
                        left,
                        right
                    )
                )

            else:

                output = (
                    current_engine.process(
                        samples,
                        update_progress
                    )
                )

                if output is None:
                    return

            output_dir = (
                "/storage/emulated/0/Download"
            )

            if not os.path.exists(
                output_dir
            ):
                output_dir = os.path.dirname(
                    selected_file
                )

            out_path = os.path.join(
                output_dir,
                output_field.value
                + ".wav"
            )

            wav.write(
                out_path,
                audio.frame_rate,
                (
                    output * 32767
                ).astype(np.int16)
            )

            status_text.value = (
                f"Concluído!\n{out_path}"
            )

        except Exception as ex:

            status_text.value = (
                f"Erro:\n{str(ex)}"
            )

        finally:

            progress_bar.visible = False
            cancel_button.visible = False
            process_button.disabled = False

            page.update()

    def start_processing(e):

        if not selected_file:
            return

        progress_bar.visible = True
        progress_bar.value = 0

        process_button.disabled = True
        cancel_button.visible = True

        page.update()

        threading.Thread(
            target=process_thread,
            daemon=True
        ).start()

    process_button.on_click = (
        start_processing
    )

    # -------------------------
    # Cancelamento
    # -------------------------

    def cancel_processing(e):

        nonlocal current_engine

        if current_engine:
            current_engine.cancel()

        status_text.value = (
            "Cancelado."
        )

        page.update()

    cancel_button.on_click = (
        cancel_processing
    )

    # -------------------------
    # Layout
    # -------------------------

    page.add(
        ft.Card(
            content=ft.Container(
                padding=20,
                width=450,
                content=ft.Column(
                    controls=[
                        title,
                        subtitle,

                        ft.Divider(),

                        select_button,
                        file_name,

                        ft.Divider(),

                        ft.Text(
                            "Informações"
                        ),

                        ft.Row([
                            ft.Text(
                                "Duração:"
                            ),
                            info_duration
                        ]),

                        ft.Row([
                            ft.Text(
                                "Sample Rate:"
                            ),
                            info_samplerate
                        ]),

                        ft.Row([
                            ft.Text(
                                "Canais:"
                            ),
                            info_channels
                        ]),

                        stretch_field,
                        window_field,
                        output_field,

                        estimated_duration,

                        progress_bar,

                        process_button,
                        cancel_button,

                        status_text
                    ]
                )
            )
        )
    )


ft.app(target=main)
