"""가사·음절 정렬 (STEP 3).

데이터 책임
- ``analysis/lyrics.json`` : 가사 텍스트의 주인. 줄·단어·음절(고유 id), 언어, 자동 인식 원본, 편집 이력.
- ``analysis/alignment.json`` : 음절별 시간(시작·끝·신뢰도·출처·사용자 수정 여부)과 음절↔음표 연결.

음표(notes.json)는 가사를 모른다. 연결은 alignment.json 한 곳에만 있어 두 파일이 서로 어긋날 수 없다.
"""
