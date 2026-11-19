"""재생 안정성 점검. 자세한 설명은 amp/devtools/playback_check.py 참고.

    python scripts/playback_check.py [--audio 곡.mp3] [--rounds 3] [--device-change 20]
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from amp.devtools.playback_check import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
