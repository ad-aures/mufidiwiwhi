# Mufidiwiwhi - multi-file diarisation transcription with Whisper.
# (C) 2026 Ad Aures · Benjamin Bellamy <benjamin@podlibre.org>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

"""Mufidiwiwhi: multi-file diarisation transcription with Whisper."""

__version__ = "2.0.0"

from .core import (
    Cancelled,
    CorrectionConfig,
    RunConfig,
    SpeakerInput,
    run_pipeline,
)

__all__ = [
    "Cancelled",
    "CorrectionConfig",
    "RunConfig",
    "SpeakerInput",
    "__version__",
    "run_pipeline",
]
