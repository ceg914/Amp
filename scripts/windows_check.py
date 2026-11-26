"""v0.7.0 Windows 점검. 자세한 설명은 amp/devtools/windows_check.py 참고.

    python scripts/windows_check.py [--hash] [--no-probe] [--out 폴더]
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from amp.devtools.windows_check import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
