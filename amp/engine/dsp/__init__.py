"""무거운 신호 처리 함수. 분석 작업 프로세스(``amp.engine.worker``)가 호출한다.

librosa·torch 같은 라이브러리는 함수 안에서 import한다. GUI 프로세스가 이 패키지를 import해도
무거운 라이브러리가 따라 올라오지 않게 하기 위해서다.
"""
