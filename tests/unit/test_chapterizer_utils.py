# Unit tests for chapterizer pure utility functions

from __future__ import annotations

import pytest

from src.agents.chapterizer import (
    ChapterInfo,
    _normalize_title,
    _collapse_ws,
    _fast_check_pass,
    _is_toc_region,
    _find_skip_toc,
    _split_by_meta,
    _VALID_CN_CHAPTER,
    _NOISE_PATTERNS,
)


# ========================================================================
# ChapterInfo
# ========================================================================

class TestChapterInfo:
    def test_default_values(self):
        ch = ChapterInfo(title="第一章")
        assert ch.title == "第一章"
        assert ch.level == 1
        assert ch.start_marker == ""
        assert ch.text == ""
        assert ch.text_length == 0

    def test_to_dict_with_text(self):
        ch = ChapterInfo(title="第一章", level=1, text="深度学习是..." * 50)
        d = ch.to_dict()
        assert d["title"] == "第一章"
        assert d["level"] == 1
        assert len(d["text_preview"]) <= 200

    def test_to_dict_empty_text(self):
        ch = ChapterInfo(title="前言")
        d = ch.to_dict()
        assert d["text_preview"] == ""
        assert d["text_length"] == 0


# ========================================================================
# _normalize_title
# ========================================================================

class TestNormalizeTitle:
    def test_strips_markdown_headers(self):
        assert _normalize_title("## 第一章 概述") == "第一章：概述"
        assert _normalize_title("# 深度学习") == "深度学习"

    def test_corrects_ocr_errors(self):
        """'童' should be corrected to '章'."""
        result = _normalize_title("第一童 静电场")
        assert "童" not in result
        assert "章" in result

    def test_normalizes_chinese_chapter_prefix(self):
        """Should normalize Chinese chapter prefix with colon."""
        result = _normalize_title("第一章 静电场的基本规律")
        assert result == "第一章：静电场的基本规律"

    def test_normalizes_english_chapter_prefix(self):
        result = _normalize_title("Chapter 1 Introduction")
        assert result == "Chapter 1: Introduction"

    def test_strips_trailing_punctuation(self):
        """Trailing dots and punctuation should be removed."""
        result = _normalize_title("第一章 概述...")
        assert not result.endswith("...")
        assert "概述" in result

    def test_collapses_whitespace(self):
        result = _normalize_title("第一章   静电场   ")
        assert result == "第一章：静电场"

    def test_handles_empty(self):
        assert _normalize_title("") == ""
        assert _normalize_title("   ") == ""

    def test_handles_non_chapter_text(self):
        """Non-chapter text should pass through with cleaning."""
        result = _normalize_title("  前言  ")
        assert result == "前言"


# ========================================================================
# _collapse_ws
# ========================================================================

class TestCollapseWs:
    def test_collapses_spaces(self):
        assert _collapse_ws("第一章   静电场  ") == "第一章 静电场"

    def test_normalizes_colons(self):
        """Both Chinese and English colons should become spaces."""
        result = _collapse_ws("第一章：静电场")
        assert result == "第一章 静电场"

        result2 = _collapse_ws("Chapter 1: Intro")
        assert result2 == "Chapter 1 Intro"


# ========================================================================
# _fast_check_pass
# ========================================================================

class TestFastCheckPass:
    def _make_chapter(self, title: str) -> ChapterInfo:
        return ChapterInfo(title=title)

    def test_less_than_two_chapters_returns_false(self):
        """Need at least 2 chapters to pass."""
        chapters = [self._make_chapter("第一章 概述")]
        assert _fast_check_pass(chapters) is False

    def test_empty_title_returns_false(self):
        chapters = [self._make_chapter(""), self._make_chapter("第一章")]
        assert _fast_check_pass(chapters) is False

    def test_all_valid_chapters_passes(self):
        chapters = [
            self._make_chapter("第一章 静电场"),
            self._make_chapter("第二章 导体"),
            self._make_chapter("第三章 介质"),
        ]
        assert _fast_check_pass(chapters) is True

    def test_mixed_chapters_below_threshold_fails(self):
        """Only 2/5 valid → 40% < 80% threshold."""
        chapters = [
            self._make_chapter("第一章 概述"),
            self._make_chapter("第二章 方法"),
            self._make_chapter("前言"),
            self._make_chapter("附录"),
            self._make_chapter("参考文献"),
        ]
        assert _fast_check_pass(chapters) is False

    def test_noise_title_returns_false(self):
        """Formula/figure-like titles should trigger rejection."""
        chapters = [
            self._make_chapter("第一章 概述"),
            self._make_chapter("图 1-1 电场线分布"),
        ]
        assert _fast_check_pass(chapters) is False

    def test_english_chapters_pass(self):
        chapters = [
            self._make_chapter("Chapter 1 Introduction"),
            self._make_chapter("Chapter 2 Methods"),
        ]
        assert _fast_check_pass(chapters) is True

    def test_numbered_list_format_passes(self):
        chapters = [
            self._make_chapter("1. Introduction"),
            self._make_chapter("2. Background"),
        ]
        assert _fast_check_pass(chapters) is True


# ========================================================================
# _is_toc_region
# ========================================================================

class TestIsTocRegion:
    def test_dense_chapter_markers_is_toc(self):
        """Text with many chapter markers in a small window → TOC."""
        text = "第一章 电场\n第二章 磁场\n第三章 电磁感应\n第四章 麦克斯韦方程"
        assert _is_toc_region(text, pos=len(text) // 2) is True

    def test_sparse_markers_is_not_toc(self):
        """Single chapter marker in isolation → not TOC."""
        text = "这是第一章的正文内容，讨论了电场的基本性质。" + "X" * 1000
        assert _is_toc_region(text, pos=20) is False

    def test_english_chapter_markers(self):
        text = "Chapter 1\nChapter 2\nChapter 3\nChapter 4"
        assert _is_toc_region(text, pos=15) is True


# ========================================================================
# _find_skip_toc
# ========================================================================

class TestFindSkipToc:
    def test_finds_pattern_outside_toc(self):
        """Should skip TOC region and find pattern in body text."""
        # Dense chapter markers → TOC-like region, then body text
        toc_region = "Chapter 1\nChapter 2\nChapter 3\nChapter 4\n"
        body_region = "The quick brown fox " * 20 + "TARGET_HERE" + " extra text " * 20
        text = toc_region + body_region
        pos = _find_skip_toc(text, "TARGET_HERE")
        # Should find it (outside TOC region)
        assert pos >= 0

    def test_returns_negative_when_not_found(self):
        text = "nothing here"
        pos = _find_skip_toc(text, "不存在")
        assert pos == -1


# ========================================================================
# _split_by_meta
# ========================================================================

class TestSplitByMeta:
    def test_splits_by_exact_markers(self):
        """Exact marker match should locate chapters in text."""
        content = "contents " * 30  # enough text for minimum
        text = f"Ch1 Start\n{content}\nCh2 Start\n{content}"
        meta = [
            {"title": "Ch1", "start_marker": "Ch1 Start"},
            {"title": "Ch2", "start_marker": "Ch2 Start"},
        ]
        chapters = _split_by_meta(text, meta)
        assert len(chapters) >= 1

    def test_empty_meta_returns_empty(self):
        chapters = _split_by_meta("some text", [])
        assert chapters == []

    def test_marker_not_found_still_includes_title_only(self):
        """When a chapter marker can't be located, include it as title-only."""
        content = "content text " * 50
        text = f"Chapter One Start\n{content}"
        meta = [
            {"title": "Chapter One", "start_marker": "Chapter One Start"},
            {"title": "Chapter Two Missing", "start_marker": "Chapter Two Missing"},
        ]
        chapters = _split_by_meta(text, meta)
        assert len(chapters) >= 1

    def test_deduplicates_positions(self):
        """Duplicate positions should be removed."""
        content = "content text " * 30
        text = f"Start Here\n{content}"
        meta = [
            {"title": "First", "start_marker": "Start Here"},
            {"title": "Second", "start_marker": "Start Here"},  # same marker
        ]
        chapters = _split_by_meta(text, meta)
        # Duplicate positions should not create duplicate chapters
        assert len(chapters) <= 1


# ========================================================================
# Regex pattern validation
# ========================================================================

class TestRegexPatterns:
    def test_valid_cn_chapter_matches_standard_format(self):
        assert _VALID_CN_CHAPTER.match("第一章 静电场")
        assert _VALID_CN_CHAPTER.match("第十二章 总结")
        assert _VALID_CN_CHAPTER.match("第一课 入门")
        assert _VALID_CN_CHAPTER.match("第三讲 进阶")

    def test_valid_cn_does_not_match_noise(self):
        assert _VALID_CN_CHAPTER.match("前言") is None
        assert _VALID_CN_CHAPTER.match("附录A") is None

    def test_noise_patterns_catches_figures(self):
        assert _NOISE_PATTERNS.search("图 1-1 电场线")
        assert _NOISE_PATTERNS.search("表 3 数据")

    def test_noise_does_not_match_chapter(self):
        assert _NOISE_PATTERNS.search("第一章 概述") is None
