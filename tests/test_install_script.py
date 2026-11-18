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


def test_find_nvidia_smi_checks_windows_locations():
    """사용자 Windows 실행 기록에서 GPU가 있는데 CPU판 PyTorch가 설치된 것을 발견했다 (nvidia-smi가 PATH 밖일 수 있음)."""
    nvsmi = os.path.join(os.environ.get("ProgramFiles", r"C:\Program Files"), "NVIDIA Corporation", "NVSMI", "nvidia-smi.exe")
    assert inst.find_nvidia_smi(which=lambda n: "/usr/bin/nvidia-smi") == "/usr/bin/nvidia-smi"
    assert inst.find_nvidia_smi(which=lambda n: None, isfile=lambda p: p == nvsmi, nt=True) == nvsmi
    assert inst.find_nvidia_smi(which=lambda n: None, isfile=lambda p: True, nt=False) is None


def test_cpu_torch_is_removed_before_gpu_build():
    cmds = inst.commands("cu132", "py", "win32", existing=("2.15.0+cpu", None))
    assert cmds[0] == ["py", "-m", "pip", "uninstall", "-y", "torch"]
    assert cmds[1][-2:] == ["--index-url", "https://download.pytorch.org/whl/cu132"]
    assert inst.commands("cu132", "py", "win32", existing=("2.15.0+cu132", "13.2"))[0][3] == "install"
    assert inst.commands("cpu", "py", "win32", existing=("2.15.0+cpu", None))[0][3] == "install"


class _R:
    def __init__(self, stdout="", stderr="", returncode=0):
        self.stdout, self.stderr, self.returncode = stdout, stderr, returncode


def test_cuda_version_from_query_output_and_diagnostics():
    """사용자 Windows: nvidia-smi는 찾았지만 CUDA 버전을 못 읽어 CPU판을 설치했다 → -q 출력도 보고, 못 읽으면 이유를 보여 준다."""
    q = "==============NVSMI LOG==============\nDriver Version                            : 581.15\nCUDA Version                              : 13.0\n"
    assert inst.parse_cuda_version(q) == (13, 0)
    calls = []

    def run(args, **kw):
        calls.append(args)
        return _R(stdout="weird header without version") if len(args) == 1 else _R(stdout=q)

    assert inst.detect_cuda("nvidia-smi.exe", run=run) == ((13, 0), "")
    assert calls == [["nvidia-smi.exe"], ["nvidia-smi.exe", "-q"]]

    def broken(args, **kw):
        return _R(stdout="NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver.", returncode=9)

    version, diag = inst.detect_cuda("nvidia-smi.exe", run=broken)
    assert version is None and "종료 코드 9" in diag and "couldn't communicate" in diag
    assert inst.detect_cuda(None, run=broken) == (None, "") or True  # PATH에 없으면 실행하지 않는다


def test_build_can_be_forced(capsys, monkeypatch):
    monkeypatch.setattr(inst, "find_nvidia_smi", lambda: "nvidia-smi.exe")
    monkeypatch.setattr(inst, "detect_cuda", lambda exe=None: (None, "기본 출력: 종료 코드 9, 출력 앞부분: failed"))
    monkeypatch.setattr(inst, "installed_torch", lambda python=None: ("2.15.0+cpu", None))
    assert inst.main(["--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "PyTorch 빌드: cpu" in out and "종료 코드 9" in out and "--build cu126" in out
    assert inst.main(["--dry-run", "--build", "cu126"]) == 0
    out = capsys.readouterr().out
    assert "uninstall -y torch" in out and "whl/cu126" in out


def test_batch_files_are_ascii_and_messages_exist():
    """cmd.exe가 UTF-8 배치 파일 안의 한글을 잘못 읽어 주석 일부를 명령으로 실행했다 (사용자 Windows, run.bat).
    배치 파일은 ASCII만 쓰고, 한글 안내는 scripts/messages/*.txt를 type으로 보여 준다."""
    import re

    for name in ("run.bat", "install_separation.bat"):
        data = open(os.path.join(ROOT, name), "rb").read()
        data.decode("ascii")  # 한글 등 여러 바이트 문자가 있으면 실패
        assert b"\r\n" in data and b"\n" not in data.replace(b"\r\n", b"")  # CRLF만
        assert not re.search(rb"(?im)^\s*rem\b.*[|&<>]", data)  # rem 줄에 파이프·리디렉션 기호 금지
        for msg in re.findall(rb'type "(?:%MSG%|scripts\\messages)\\([^"]+)"', data):
            path = os.path.join(ROOT, "scripts", "messages", msg.decode())
            raw = open(path, "rb").read()
            assert not raw.startswith(b"\xef\xbb\xbf")  # BOM이 있으면 type이 그대로 찍는다
            raw.decode("utf-8")
