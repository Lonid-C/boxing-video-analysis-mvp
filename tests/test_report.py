from pathlib import Path

from boxing_mvp.report import write_report


def test_report_separates_formal_events_from_cv_candidates(tmp_path):
    path = Path(write_report(str(tmp_path), "video.mp4", {
        "events": [
            {"source": "cv", "side": "red", "start_time_sec": 1, "end_time_sec": 1.4},
            {"source": "vlm_scan", "fighter": "blue", "hit_or_miss": "hit",
             "start_time_sec": 2, "end_time_sec": 2.4, "reason": "清晰接触"},
        ],
        "vlm_events": [{"source": "vlm_scan", "fighter": "blue", "hit_or_miss": "hit",
                        "start_time_sec": 2, "end_time_sec": 2.4, "reason": "清晰接触"}],
    }))
    page = path.read_text(encoding="utf-8")
    visible_page = page.split("<script>", 1)[0]
    assert "正式事件时间轴与回放" in page and "CV 候选时间轴" in page
    assert visible_page.count("清晰接触") == 1
    assert visible_page.count("仅表示疑似出拳动作，不表示命中") == 1
    assert "playEvent(2.000,2.400)" in page
