"""
Windows FFmpeg DLL shim.

FFmpeg here is a Chocolatey `ffmpeg-shared` install; its DLLs aren't on the
default loader path, so any library that dlopen's them (torchaudio's FFmpeg
backend / torchcodec) fails unless the bin dir is registered first.

Import this module *before* importing torchaudio if you intend to use an
FFmpeg-routed decode path.  It's a no-op when FFmpeg isn't found, and the
current soundfile + torchaudio.transforms path doesn't need it.
"""

import glob
import os

_CANDIDATES = [
    r"C:\ProgramData\chocolatey\lib\ffmpeg-shared\tools\ffmpeg-*\bin",
    r"C:\ProgramData\chocolatey\lib\ffmpeg\tools\ffmpeg-*\bin",
]


def add_ffmpeg_to_dll_path() -> str | None:
    if not hasattr(os, "add_dll_directory"):  # non-Windows
        return None
    for pattern in _CANDIDATES:
        matches = glob.glob(pattern)
        if matches:
            os.add_dll_directory(matches[0])
            return matches[0]
    return None


# Apply on import.
_FFMPEG_BIN = add_ffmpeg_to_dll_path()
