"""실제 음원 검증 도구. 자세한 설명은 amp/devtools/validate.py 참고.

    python scripts/validate_audio.py <오디오 파일> --type A|B|C [--ref 정답.csv] [--device auto|cpu|cuda]
    python scripts/validate_audio.py --self-check
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from amp.devtools.validate import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
