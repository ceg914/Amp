"""프로그램 전체에서 쓰는 예외 계층."""


class AmpError(Exception):
    """사용자에게 그대로 보여 줄 수 있는 메시지를 담는 기본 예외."""


class AudioLoadError(AmpError):
    """오디오 파일을 열거나 디코딩하지 못했을 때."""


class ProjectFormatError(AmpError):
    """프로젝트 파일이 손상됐거나 지원하지 않는 형식일 때."""


class EditError(AmpError):
    """편집 요청이 유효하지 않을 때 (예: 길이가 너무 짧은 음표)."""


class AnalysisError(AmpError):
    """분석 단계가 실패했을 때."""


class AnalysisCancelled(AmpError):
    """사용자가 분석을 취소했을 때."""


class StorageError(AmpError):
    """파일을 쓰거나 읽지 못했을 때 (디스크 부족, 권한, 잠김 등)."""
