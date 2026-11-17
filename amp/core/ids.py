"""객체 식별자 생성."""

import uuid


def new_id() -> str:
    """충돌 가능성이 사실상 없는 짧은 식별자를 만든다."""
    return uuid.uuid4().hex[:16]
