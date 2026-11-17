"""프로젝트의 분석 실행 기록(analysis/runs/)을 나열하거나 두 실행을 비교한다.

사용법
    python scripts/compare_runs.py <프로젝트 폴더> --list
    python scripts/compare_runs.py <프로젝트 폴더>                 가장 최근 두 실행 비교
    python scripts/compare_runs.py <프로젝트 폴더> <A> <B>         실행 ID(앞부분만 써도 됨) 지정
    옵션: --out <파일>  보고서를 프로젝트 밖에 쓴다 (기본: <프로젝트>/analysis/reports/)
          --stdout     보고서를 화면에 출력만 한다

기록 파일만 읽는다. 프로젝트의 음표·설정은 바꾸지 않는다.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from amp.project import runs  # noqa: E402
from amp.project.layout import ProjectPaths  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="compare_runs", description="분석 실행 기록 비교")
    ap.add_argument("project")
    ap.add_argument("ids", nargs="*", help="비교할 실행 ID 두 개 (없으면 최근 두 개)")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--out")
    ap.add_argument("--stdout", action="store_true")
    ns = ap.parse_args(argv)

    analysis = ProjectPaths(ns.project).analysis
    records = runs.list_runs(analysis)
    if ns.list:
        if not records:
            print("실행 기록이 없습니다.")
        for r in records:
            total = r.total_seconds
            print(f"{r.run_id}  {r.kind:9s} {r.created}  음표 {len(r.notes):4d}  "
                  f"민감도 {r.settings.get('sensitivity', '-'):8s}  "
                  f"{'-' if total is None else f'{total:.2f}s'}")
        return 0
    try:
        if ns.ids:
            if len(ns.ids) != 2:
                ap.error("실행 ID는 두 개를 주세요.")
            a, b = runs.find_run(analysis, ns.ids[0]), runs.find_run(analysis, ns.ids[1])
        else:
            if len(records) < 2:
                print(f"비교하려면 실행 기록이 2개 이상 필요합니다 (현재 {len(records)}개).", file=sys.stderr)
                return 1
            a, b = records[-2], records[-1]
    except KeyError as e:
        print(e.args[0], file=sys.stderr)
        return 1
    text = runs.markdown_report(runs.compare_runs(a, b))
    if ns.stdout:
        print(text)
        return 0
    if ns.out:
        with open(ns.out, "w", encoding="utf-8") as f:
            f.write(text)
        print(ns.out)
    else:
        print(runs.write_report(analysis, a, b))
    return 0


if __name__ == "__main__":
    sys.exit(main())
