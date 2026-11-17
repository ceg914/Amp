import importlib.util
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("install_separation", os.path.join(ROOT, "scripts", "install_separation.py"))
inst = importlib.util.module_from_spec(spec)
spec.loader.exec_module(inst)

SMI = """+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 581.15                 Driver Version: 581.15         CUDA Version: 13.0     |
|-------------------------------+----------------------+---------------------------------+"""


def test_parse_and_choose_build():
    assert inst.parse_cuda_version(SMI) == (13, 0)
    assert inst.parse_cuda_version("no gpu") is None
    assert inst.choose_build((13, 2)) == "cu132"
    assert inst.choose_build((13, 0)) == "cu130"
    assert inst.choose_build((12, 8)) == "cu126"
    assert inst.choose_build((11, 8)) == "cpu"  # 드라이버가 오래됨
    assert inst.choose_build(None) == "cpu"
    assert inst.choose_build((13, 2), force_cpu=True) == "cpu"


def test_commands_per_platform():
    win_cpu = inst.commands("cpu", "py", "win32")
    assert win_cpu[0] == ["py", "-m", "pip", "install", "--upgrade", "torch"]  # Windows CPU판 = PyPI 기본
    win_gpu = inst.commands("cu132", "py", "win32")
    assert win_gpu[0][-2:] == ["--index-url", "https://download.pytorch.org/whl/cu132"]
    linux_cpu = inst.commands("cpu", "py", "linux")
    assert linux_cpu[0][-1].endswith("/whl/cpu")
    assert win_cpu[1][-1].endswith("requirements-separation.txt")
