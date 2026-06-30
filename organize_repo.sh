#!/bin/bash
# ──────────────────────────────────────────
# pitch_project 폴더 구조 정리 스크립트
# rapsodo-visual-analysis(=pitching-mechanics-analysis-using-a-phone) 폴더 안에서 실행하세요
#
# 사용법:
#   cd ~/rapsodo-visual-analysis
#   (이 파일을 폴더 안에 저장한 뒤)
#   bash organize_repo.sh
# ──────────────────────────────────────────

echo "폴더 구조 생성 중..."
mkdir -p docs
mkdir -p src
mkdir -p sample_data
mkdir -p results
mkdir -p figures
mkdir -p videos

echo "코드 파일 이동 중..."
[ -f pitch_summary_v2.py ] && mv pitch_summary_v2.py src/

echo "결과 데이터(csv) 이동 중..."
[ -f pitch_metrics.csv ] && mv pitch_metrics.csv results/
[ -f side_metrics.csv ] && mv side_metrics.csv results/

echo "그래프/이미지(figures) 이동 중..."
[ -f phase_timeline.png ] && mv phase_timeline.png figures/
[ -f hss_check_frame.png ] && mv hss_check_frame.png figures/
[ -f stride_check_frame.png ] && mv stride_check_frame.png figures/
[ -f extension_check_frame.png ] && mv extension_check_frame.png figures/
[ -f hss_graph.png ] && mv hss_graph.png figures/
[ -f stride_release_graph.png ] && mv stride_release_graph.png figures/
[ -f peak_frame.png ] && mv peak_frame.png figures/

echo "샘플 영상만 videos/ 폴더로 복사 (원본은 그대로 두고 .gitignore로 제외)..."
[ -f IMG_9225_2.mov ] && cp IMG_9225_2.mov videos/sample_pitch_side_view.mov

echo ""
echo "완료! 현재 구조:"
echo ""
find . -maxdepth 2 -not -path '*/venv/*' -not -path '*/.git/*' | sort
