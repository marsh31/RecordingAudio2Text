#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import datetime

import numpy as np
import sounddevice as sd
import soundfile as sf
import whisper


def record_audio(filename: str, samplerate: int = 16000, channels: int = 1):
    """
    Enterキーが押されるまでマイクから音声を録音し、WAVファイルに保存します。
    :param filename: 出力する WAV ファイル名
    :param samplerate: サンプリングレート（Hz）
    :param channels: チャンネル数（モノラル=1, ステレオ=2）
    """
    print(
        f"録音開始: Enterキーを押すと停止します (サンプルレート {samplerate}Hz、チャンネル {channels})"
    )
    frames = []

    def callback(indata, frame_count, time_info, status):
        frames.append(indata.copy())

    with sd.InputStream(samplerate=samplerate, channels=channels, callback=callback):
        input()  # Enterキー待ち

    recording = np.concatenate(frames, axis=0)
    sf.write(filename, recording, samplerate)
    print(f"WAV ファイルを保存しました: {filename}")


def transcribe_local(audio_path: str, model_name: str = "base") -> str:
    """ローカルの Whisper モデルで文字起こし"""
    print(f"Whisper モデル '{model_name}' をロード中…")

    model = whisper.load_model(model_name)
    print(f"'{audio_path}' を文字起こし中…")

    result = model.transcribe(audio_path)
    print("文字起こし完了")

    return result["text"]


def save_text(text: str, filename: str):
    """文字起こし結果をテキストファイルに保存"""
    with open(filename, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"テキスト保存 → {filename}")


def main():
    parser = argparse.ArgumentParser(
        description="ローカル Whisper で録音→文字起こし→保存"
    )

    ts = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    audio_file = f"{ts}_audio.wav"
    text_file = f"{ts}_text.txt"

    parser.add_argument(
        "-a", "--audio", type=str, default=audio_file, help="出力 WAV ファイル名"
    )
    parser.add_argument(
        "-t", "--text", type=str, default=text_file, help="出力テキストファイル名"
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="base",
        choices=["tiny", "base", "small", "medium", "large"],
        help="使用する Whisper モデル",
    )
    args = parser.parse_args()

    record_audio(args.audio)
    transcript = transcribe_local(args.audio, args.model)
    save_text(transcript, args.text)


if __name__ == "__main__":
    main()
