"""오디오 분석 엔진 (Qt 비의존).

UI는 ``AnalysisPipeline`` 과 ``AnalysisResult`` 만 알면 된다.
각 분석 단계는 ``interfaces`` 의 추상 클래스를 구현한 백엔드로 교체할 수 있다.
"""
