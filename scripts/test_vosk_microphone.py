from __future__ import annotations

import argparse

from classvoice.speech import measure_microphone_level, recognize_microphone_once


def main() -> None:
    parser = argparse.ArgumentParser(description="Test Vosk microphone recognition.")
    parser.add_argument("--model-path", default="models/vosk-model-small-cn-0.22")
    parser.add_argument("--device", type=int, default=None)
    parser.add_argument("--sample-rate", type=int, default=44100)
    parser.add_argument("--seconds", type=float, default=5.0)
    args = parser.parse_args()

    print("请对着麦克风说一句中文，开始测试...")
    level = measure_microphone_level(args.device, args.sample_rate, seconds=2.0)
    print(f"microphone level: rms={level['rms']:.6f}, peak={level['peak']:.6f}")
    result = recognize_microphone_once(
        model_path=args.model_path,
        device=args.device,
        sample_rate=args.sample_rate,
        seconds=args.seconds,
    )
    print(f"accepted: {result['accepted']}")
    print(f"text: {result['text']}")
    print(f"raw: {result['raw']}")


if __name__ == "__main__":
    main()
