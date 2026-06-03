from __future__ import annotations

import json
import queue
import threading
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SpeechConfig:
    model_path: str = "models/vosk-model-small-cn-0.22"
    sample_rate: int = 16000
    device: int | str | None = None


class VoskRecorder:
    """Background microphone recorder powered by Vosk."""

    def __init__(self, config: SpeechConfig) -> None:
        self.config = config
        self._text_queue: queue.Queue[str] = queue.Queue()
        self._partial_queue: queue.Queue[str] = queue.Queue()
        self._error_queue: queue.Queue[str] = queue.Queue()
        self._status_queue: queue.Queue[str] = queue.Queue()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        if self.is_running:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="classvoice-vosk", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=3)

    def drain_texts(self) -> list[str]:
        return _drain_queue(self._text_queue)

    def drain_partials(self) -> list[str]:
        return _drain_queue(self._partial_queue)

    def drain_errors(self) -> list[str]:
        return _drain_queue(self._error_queue)

    def drain_statuses(self) -> list[str]:
        return _drain_queue(self._status_queue)

    def _run(self) -> None:
        try:
            self._validate_dependencies()

            import sounddevice as sd
            from vosk import KaldiRecognizer

            model_dir = Path(self.config.model_path)
            if not model_dir.exists():
                raise FileNotFoundError(f"Vosk 模型目录不存在：{model_dir}")

            self._status_queue.put("正在加载 Vosk 模型...")
            model = load_vosk_model(str(model_dir))
            self._status_queue.put("Vosk 模型加载完成，开始监听麦克风。")
            recognizer = KaldiRecognizer(model, self.config.sample_rate)
            audio_queue: queue.Queue[bytes] = queue.Queue()

            def callback(indata, frames, time_info, status):  # noqa: ANN001
                if status:
                    self._error_queue.put(str(status))
                audio_queue.put(bytes(indata))

            with sd.RawInputStream(
                samplerate=self.config.sample_rate,
                blocksize=8000,
                dtype="int16",
                channels=1,
                callback=callback,
                device=self.config.device,
            ):
                while not self._stop_event.is_set():
                    try:
                        data = audio_queue.get(timeout=0.2)
                    except queue.Empty:
                        continue

                    if recognizer.AcceptWaveform(data):
                        self._emit_result(recognizer.Result(), final=True)
                    else:
                        self._emit_result(recognizer.PartialResult(), final=False)

                self._emit_result(recognizer.FinalResult(), final=True)
        except Exception as exc:
            self._error_queue.put(str(exc))

    def _emit_result(self, raw_result: str, final: bool) -> None:
        try:
            result = json.loads(raw_result)
        except json.JSONDecodeError:
            return

        key = "text" if final else "partial"
        text = result.get(key, "").strip()
        if not text:
            return
        if final:
            self._text_queue.put(text)
        else:
            self._partial_queue.put(text)

    @staticmethod
    def _validate_dependencies() -> None:
        try:
            import sounddevice  # noqa: F401
            import vosk  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("请先安装依赖：pip install vosk sounddevice") from exc


@lru_cache(maxsize=2)
def load_vosk_model(model_path: str):
    from vosk import Model

    return Model(model_path)


def list_input_devices() -> list[dict[str, Any]]:
    try:
        import sounddevice as sd
    except ImportError:
        return []

    hostapis = sd.query_hostapis()
    devices = sd.query_devices()
    input_devices: list[dict[str, Any]] = []
    for index, device in enumerate(devices):
        if device.get("max_input_channels", 0) <= 0:
            continue
        hostapi_index = int(device.get("hostapi", -1))
        hostapi_name = hostapis[hostapi_index]["name"] if hostapi_index >= 0 else "unknown"
        input_devices.append(
            {
                "index": index,
                "name": str(device.get("name", f"Device {index}")),
                "hostapi": hostapi_name,
                "channels": int(device.get("max_input_channels", 0)),
                "default_samplerate": int(device.get("default_samplerate", 0)),
            }
        )
    return input_devices


def get_default_input_device() -> int | None:
    try:
        import sounddevice as sd
    except ImportError:
        return None

    default_device = sd.default.device
    if hasattr(default_device, "__getitem__"):
        value = default_device[0]
    elif hasattr(default_device, "input"):
        value = default_device.input
    elif isinstance(default_device, (list, tuple)) and default_device:
        value = default_device[0]
    else:
        value = default_device
    return int(value) if value is not None and int(value) >= 0 else None


def measure_microphone_level(
    device: int | str | None,
    sample_rate: int,
    seconds: float = 2.0,
) -> dict[str, float]:
    try:
        import numpy as np
        import sounddevice as sd
    except ImportError as exc:
        raise RuntimeError("请先安装依赖：pip install sounddevice numpy") from exc

    frames = int(sample_rate * seconds)
    audio = sd.rec(
        frames,
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
        device=device,
    )
    sd.wait()

    rms = float(np.sqrt(np.mean(np.square(audio)))) if audio.size else 0.0
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    return {"rms": rms, "peak": peak}


def recognize_microphone_once(
    model_path: str,
    device: int | str | None,
    sample_rate: int,
    seconds: float = 5.0,
) -> dict[str, Any]:
    import json

    import sounddevice as sd
    from vosk import KaldiRecognizer

    model = load_vosk_model(model_path)
    recognizer = KaldiRecognizer(model, sample_rate)
    frames = int(sample_rate * seconds)
    audio = sd.rec(frames, samplerate=sample_rate, channels=1, dtype="int16", device=device)
    sd.wait()

    audio_bytes = audio.tobytes()
    accepted = recognizer.AcceptWaveform(audio_bytes)
    raw = recognizer.Result() if accepted else recognizer.FinalResult()
    result = json.loads(raw)
    return {
        "accepted": accepted,
        "text": result.get("text", "").strip(),
        "raw": result,
    }


def _drain_queue(source: queue.Queue[str]) -> list[str]:
    items: list[str] = []
    while not source.empty():
        items.append(source.get_nowait())
    return items
