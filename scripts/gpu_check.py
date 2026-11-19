"""GPU 가사 인식 진단. 자세한 설명은 amp/devtools/gpu_check.py 참고.

    python scripts/gpu_check.py [--audio 보컬.flac] [--models large-v3,turbo] [--seconds 60]
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from amp.devtools.gpu_check import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
