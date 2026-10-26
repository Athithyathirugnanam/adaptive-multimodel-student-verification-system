"""Microphone & Whisper Diagnostics Script

This script checks:
1. Available Windows audio capture devices.
2. Direct microphone audio recording test (3 seconds).
3. Audio level analysis (RMS, Peak, Silence check).
4. Local Faster-Whisper model transcription test.
5. Actionable guidance for Windows microphone privacy and sound settings.
"""

import os
import shutil
import subprocess
import sys
import tempfile
import wave
import numpy as np

def run_diagnostic():
    print("=" * 60)
    print("RUNNING MICROPHONE & WHISPER SYSTEM DIAGNOSTICS")
    print("=" * 60)

    # 1. Check Python & Dependencies
    print("\n[1/5] Checking Python Environment & Tools...")
    print(f"  * Python: {sys.executable}")
    ffmpeg_exe = shutil.which("ffmpeg")
    print(f"  * FFmpeg: {ffmpeg_exe or 'NOT FOUND (Please install or add to PATH)'}")

    try:
        import faster_whisper
        print(f"  * faster-whisper: Installed (v{getattr(faster_whisper, '__version__', 'unknown')})")
    except ImportError:
        print("  * faster-whisper: NOT INSTALLED")

    # 2. Check DirectShow audio devices
    print("\n[2/5] Enumerating Windows DirectShow Audio Devices...")
    dshow_devices = []
    if ffmpeg_exe:
        cmd = [ffmpeg_exe, "-list_devices", "true", "-f", "dshow", "-i", "dummy"]
        proc = subprocess.run(cmd, capture_output=True, text=True, errors="ignore")
        for line in proc.stderr.splitlines():
            if "(audio)" in line:
                start = line.find('"')
                end = line.find('"', start + 1)
                if start != -1 and end != -1:
                    dev_name = line[start + 1:end]
                    dshow_devices.append(dev_name)
                    print(f"  [Mic] Found Device: {dev_name}")

    if not dshow_devices:
        print("  Warning: No DirectShow audio input devices detected via FFmpeg.")

    # 3. Test recording from available devices
    print("\n[3/5] Testing 3-Second Audio Capture from Microphone(s)...")
    working_dev = None
    for dev in dshow_devices:
        print(f"\n  Testing '{dev}'...")
        temp_wav = os.path.join(tempfile.gettempdir(), "test_mic_diag.wav")
        if os.path.exists(temp_wav):
            try:
                os.remove(temp_wav)
            except OSError:
                pass

        cmd = [
            ffmpeg_exe, "-y", "-loglevel", "error",
            "-f", "dshow", "-i", f"audio={dev}",
            "-t", "3", "-ac", "1", "-ar", "16000",
            temp_wav
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode == 0 and os.path.exists(temp_wav):
            try:
                with wave.open(temp_wav, "rb") as wf:
                    raw = wf.readframes(wf.getnframes())
                    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                    rms = float(np.sqrt(np.mean(samples ** 2))) if samples.size else 0.0
                    peak = float(np.max(np.abs(samples))) if samples.size else 0.0
                
                print(f"    * Duration: 3.0s | RMS Level: {rms:.6f} | Peak: {peak:.6f}")
                if rms < 0.0005:
                    print("    * Status: SILENCE / MUTED (Input level is near 0)")
                elif rms < 0.003:
                    print("    * Status: VERY QUIET (Speak closer or boost mic volume)")
                    working_dev = dev
                else:
                    print("    * Status: AUDIO DETECTED (Working correctly)")
                    working_dev = dev
            except Exception as e:
                print(f"    * Audio analysis error: {e}")
        else:
            print(f"    * Capture failed: {res.stderr[:200]}")

    # 4. Whisper model test
    print("\n[4/5] Testing Whisper STT Engine...")
    try:
        from verification.whisper_service import _transcribe_faster_whisper
        from verification.config import VerificationConfig

        v_cfg = VerificationConfig()
        model_size = v_cfg.whisper_cfg.get("model_size", "base")
        print(f"  * Initializing Whisper model '{model_size}'...")
        test_tone_wav = os.path.join(tempfile.gettempdir(), "test_tone.wav")
        rate = 16000
        t = np.linspace(0, 1.5, int(1.5 * rate), endpoint=False)
        audio = (np.sin(2 * np.pi * 440 * t) * 8000).astype(np.int16)
        with wave.open(test_tone_wav, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(rate)
            wf.writeframes(audio.tobytes())

        stt_res = _transcribe_faster_whisper(test_tone_wav, model_size)
        print("  * Whisper Engine: OPERATIONAL (Faster-Whisper is ready to transcribe speech)")
    except Exception as e:
        print(f"  * Whisper Error: {e}")

    # 5. Summary and Guidance
    print("\n[5/5] DIAGNOSTIC SUMMARY & WINDOWS SETTINGS CHECKLIST")
    print("-" * 60)
    if working_dev:
        print(f"Working audio input detected on: '{working_dev}'")
    else:
        print("NOTE: All recorded test samples had near-zero RMS (digital silence).")
        print("\nWINDOWS SETTINGS CHECKLIST TO UNMUTE / ENABLE MIC:")
        print("  1. Windows Mic Privacy:")
        print("     Settings -> Privacy & security -> Microphone -> Ensure 'Microphone access' and 'Let apps access your microphone' are ON.")
        print("  2. Windows Sound Settings / Input Volume:")
        print("     Settings -> System -> Sound -> Input -> Select your microphone and set Volume slider to 80-100%.")
        print("  3. Hardware / Laptop Mute Key:")
        print("     On HP / OMEN laptops, check the F8 or dedicated Mic Mute key (look for an orange/white LED on the key).")
        print("  4. Browser Device Selection:")
        print("     In the web app, select 'Microphone Array (Intel Smart Sound)' or your external headset from the 'Microphone Device' dropdown.")
    print("=" * 60)

if __name__ == "__main__":
    run_diagnostic()