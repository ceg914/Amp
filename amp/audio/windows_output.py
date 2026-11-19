"""Windows 기본 출력 장치를 Qt를 거치지 않고 Core Audio에 직접 묻는다.

Qt의 Windows 장치 목록은(개발 중 소스 기준) '기본 장치' 표시를 Windows 알림으로 고친다. 사용자 PC 점검에서 Qt 장치 변경 알림이 온 순간에도
Qt가 본 기본 장치가 목록의 첫 장치(Sound BlasterX G5)였다(STEP_3_5_VALIDATION_REPORT 7.4절). 사용자가 장치를 바꾼 직후였다면 Qt 표시가
Windows를 따라가지 못한 것이다(확인하지 못함). 그래서 재생기는 Windows에 직접 물은 기본 장치를 먼저 쓴다. Qt의 QAudioDevice.id()도 같은 끝점 id 문자열(UTF-8)이다.

Windows가 아니면 None을 돌려준다. 호출이 실패하면 OSError를 낸다(부르는 쪽이 Qt 기본 장치로 물러난다).
"""

from __future__ import annotations

import sys

# ERole: Qt는 멀티미디어 역할의 기본 장치를 쓴다
ROLES = {"multimedia": 1, "console": 0, "communications": 2}
_CLSID_ENUMERATOR = "BCDE0395-E52F-467C-8E3D-C4579291692E"
_IID_ENUMERATOR = "A95664D2-9614-4F35-A746-DE8DB63617E6"


def default_output_ids(roles: tuple[str, ...] = tuple(ROLES)) -> dict[str, str] | None:
    """역할별 기본 출력(eRender) 장치 id. 기본 장치가 없으면 빈 문자열. Windows가 아니면 None."""
    if sys.platform != "win32":
        return None
    import ctypes
    import uuid
    from ctypes import wintypes

    class GUID(ctypes.Structure):
        _fields_ = [("data", ctypes.c_ubyte * 16)]

    def guid(text: str) -> GUID:
        g = GUID()
        ctypes.memmove(g.data, uuid.UUID(text).bytes_le, 16)
        return g

    def method(obj: ctypes.c_void_p, index: int, *argtypes):
        vtbl = ctypes.cast(obj, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
        return ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, *argtypes)(vtbl[index])

    ole32 = ctypes.windll.ole32
    ole32.CoInitializeEx.restype = ctypes.c_long
    ole32.CoInitializeEx.argtypes = [ctypes.c_void_p, wintypes.DWORD]
    ole32.CoCreateInstance.restype = ctypes.c_long
    ole32.CoCreateInstance.argtypes = [ctypes.POINTER(GUID), ctypes.c_void_p, wintypes.DWORD,
                                       ctypes.POINTER(GUID), ctypes.POINTER(ctypes.c_void_p)]
    ole32.CoTaskMemFree.restype = None
    ole32.CoTaskMemFree.argtypes = [ctypes.c_void_p]
    # Qt가 먼저 COM을 초기화해 두므로 보통 S_FALSE나 RPC_E_CHANGED_MODE가 돌아온다. 어느 쪽이든 그대로 쓸 수 있다
    ole32.CoInitializeEx(None, 0x2)
    enum = ctypes.c_void_p()
    hr = ole32.CoCreateInstance(ctypes.byref(guid(_CLSID_ENUMERATOR)), None, 0x17,
                                ctypes.byref(guid(_IID_ENUMERATOR)), ctypes.byref(enum))
    if hr < 0 or not enum.value:
        raise OSError(f"MMDeviceEnumerator 생성 실패 (0x{hr & 0xFFFFFFFF:08X})")
    try:
        out: dict[str, str] = {}
        get_default = method(enum, 4, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_void_p))
        for name in roles:
            dev = ctypes.c_void_p()
            hr = get_default(enum, 0, ROLES[name], ctypes.byref(dev))  # GetDefaultAudioEndpoint(eRender, role)
            if hr < 0 or not dev.value:
                out[name] = ""  # 이 역할의 기본 출력 없음
                continue
            try:
                pid = ctypes.c_wchar_p()
                hr = method(dev, 5, ctypes.POINTER(ctypes.c_wchar_p))(dev, ctypes.byref(pid))  # IMMDevice::GetId
                if hr < 0:
                    raise OSError(f"IMMDevice::GetId 실패 (0x{hr & 0xFFFFFFFF:08X})")
                try:
                    out[name] = pid.value or ""
                finally:
                    ole32.CoTaskMemFree(pid)
            finally:
                method(dev, 2)(dev)  # Release
        return out
    finally:
        method(enum, 2)(enum)


def default_output_id() -> str | None:
    """멀티미디어 역할의 기본 출력 장치 id (Qt와 같은 역할). Windows가 아니면 None."""
    ids = default_output_ids(("multimedia",))
    return None if ids is None else ids["multimedia"]
