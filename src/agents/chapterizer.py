# 分档专家 Agent — 双角色 Reflection（检测 → 审核，最多 2 轮）
# 从文档前 20 页中识别目录结构，提取章节名称和文本范围

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass
from typing import Callable, Awaitable

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from ..config import Configuration
from ..documents.parser import Document
from ..prompts.chapterizer import CHAPTER_DETECTOR_PROMPT, CHAPTER_REVIEWER_PROMPT
from .base import BaseAgent, AgentInput, AgentOutput

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[str, str, int, int], Awaitable[None]] | None

MAX_REFLECTION_ROUNDS = 2


@dataclass
class ChapterInfo:
    """一级章节信息."""
    title: str
    level: int = 1
    start_marker: str = ""
    text: str = ""
    text_length: int = 0

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "level": self.level,
            "text_preview": self.text[:200] if self.text else "",
            "text_length": len(self.text) if self.text else 0,
        }


# ===========================================================================
# ChapterizerAgent
# ===========================================================================

class ChapterizerAgent(BaseAgent):
    """分档专家：LLM 检测目录 → 审核，最多 2 轮 Reflection."""

    def __init__(self, config: Configuration):
        super().__init__(config)
        self._detect_llm: ChatOpenAI | None = None
        self._review_llm: ChatOpenAI | None = None

    def _get_detect_llm(self) -> ChatOpenAI:
        """LLM for chapter detection — fast and cheap."""
        if self._detect_llm is None:
            model = self._config.llm_chapter_detect_model or self._config.llm_model_id
            self._detect_llm = self._make_llm(
                model=model,
                temperature=0.0,
                timeout=self._config.chapter_detect_timeout,
                max_retries=1,
            )
        return self._detect_llm

    def _get_review_llm(self) -> ChatOpenAI:
        """LLM for chapter review — more capable for validation."""
        if self._review_llm is None:
            model = self._config.llm_chapter_review_model or self._config.llm_model_id
            self._review_llm = self._make_llm(
                model=model,
                temperature=0.0,
                timeout=30,
                max_retries=1,
                max_tokens=2000,
            )
        return self._review_llm

    def _get_llm(self) -> ChatOpenAI:
        """Backward-compat: use detect LLM."""
        return self._get_detect_llm()

    # ── BaseAgent interface ────────────────────────────────────────────

    async def run(self, input: AgentInput) -> AgentOutput:
        """Generic entry point. Delegates to detect_all()."""
        doc = input.metadata.get("document")
        if doc is None:
            return AgentOutput(success=False, error="No document provided in AgentInput.metadata['document']")
        try:
            chapters = await self.detect_all(doc)
            return AgentOutput(success=True, metadata={"chapters": chapters})
        except Exception as e:
            return AgentOutput(success=False, error=str(e))
    # ==================================================================
    # Public API
    # ==================================================================

    async def detect_all(
        self, doc: Document,
        on_progress: ProgressCallback = None,
    ) -> list[ChapterInfo]:
        """检测文档一级章节，含双角色 Reflection。

        从前 20 页中定位目录 → 提取章节名 → 审核，最多 2 轮，失败则报错。

        若 Document.metadata 含 vlm_chapters（VLM 预检测的章节列表），
        跳过 LLM 检测阶段，直接进入审核和 split。
        """
        import time as _time
        _total_t0 = _time.time()
        fname = doc.filename
        logger.info("[CHAPTERIZER] ====== start detect_all for %s (%d chars) ======",
                     fname, len(doc.content))

        # ---- Fast path: VLM pre-detected chapters ----
        vlm_chapters: list[str] | None = doc.metadata.get("vlm_chapters")
        if vlm_chapters:
            logger.info("[CHAPTERIZER] using %d VLM pre-detected chapters, "
                         "skipping LLM detection", len(vlm_chapters))
            meta = [
                {"title": t, "start_marker": t, "level": 1}
                for t in vlm_chapters
            ]
            chapters = _split_by_meta(doc.content, meta)
            if chapters:
                await self._report(on_progress, fname,
                                   f"审核中 (VLM预检测:{len(chapters)}章)...", 1, 2)
                # Run review for quality check (async, not to_thread)
                try:
                    review = await self._llm_review(doc, chapters)
                except Exception:
                    logger.warning("[CHAPTERIZER] VLM chapters review failed, "
                                   "using as-is", exc_info=True)
                    review = {"verdict": "all_valid", "issues": [], "chapters": []}

                reviewed = review.get("chapters", [])
                if reviewed:
                    valid_chapters = []
                    for i, ch in enumerate(chapters):
                        if i < len(reviewed) and not reviewed[i].get("valid", True):
                            logger.info("[CHAPTERIZER] VLM review removed: '%s'", ch.title)
                        else:
                            valid_chapters.append(ch)
                    chapters = valid_chapters

                if chapters:
                    await self._report(on_progress, fname,
                                       f"审核通过(VLM):{len(chapters)}章", 2, 2)
                    logger.info("[CHAPTERIZER] ====== DONE (VLM) in %.1fs, %d chapters ======",
                                 _time.time() - _total_t0, len(chapters))
                    return chapters

            logger.warning("[CHAPTERIZER] VLM chapters failed review/split, "
                           "falling back to LLM detection")
            # Fall through to normal Reflection loop

        # ---- Reflection 循环：LLM 检测 ↔ 审核 ----
        feedback = ""
        for round_num in range(1, MAX_REFLECTION_ROUNDS + 1):
            logger.info("[CHAPTERIZER] -------- reflection round %d/%d --------",
                         round_num, MAX_REFLECTION_ROUNDS)
            await self._report(on_progress, fname,
                               f"检测目录中 (第{round_num}轮)...", 0, 2)

            # ---- 检测专家：定位目录 → 提取章节 ----
            _t_detect = _time.time()
            try:
                chapters = await self._llm_detect(doc, feedback)
            except Exception as e:
                logger.warning("LLM detect round %d failed: %s", round_num, e)
                if round_num < MAX_REFLECTION_ROUNDS:
                    feedback = f"上一轮检测失败({e})，请重新检测"
                    continue
                raise RuntimeError("检测章节失败") from e
            logger.info("[CHAPTERIZER] round %d detect: %.1fs -> %d chapters",
                         round_num, _time.time() - _t_detect, len(chapters))

            if not chapters:
                if round_num < MAX_REFLECTION_ROUNDS:
                    feedback = "未检测到目录结构，请仔细检查文档前 20 页的目录部分"
                    continue
                raise RuntimeError("检测章节失败")

            # ---- 启发式快判：格式足够规范则跳过 LLM 审核，省一次调用 ----
            try:
                fast_ok = _fast_check_pass(chapters)
            except Exception:
                logger.warning("[CHAPTERIZER] fast-check error, falling back to LLM review",
                               exc_info=True)
                fast_ok = False
            if fast_ok:
                logger.info("[CHAPTERIZER] round %d fast-check PASS (all %d titles look valid), "
                             "skipping LLM review", round_num, len(chapters))
                await self._report(on_progress, fname,
                                   f"审核通过(快判)：{len(chapters)} 个章节", 2, 2)
                logger.info("[CHAPTERIZER] ====== DONE in %.1fs (fast), %d chapters ======",
                             _time.time() - _total_t0, len(chapters))
                return chapters

            # ---- 审核专家：逐项校验 ----
            await self._report(on_progress, fname,
                               f"审核中 (第{round_num}轮)...", 1, 2)
            _t_review = _time.time()
            try:
                review = await self._llm_review(doc, chapters)
            except Exception as e:
                logger.warning("LLM review round %d failed: %s", round_num, e)
                if round_num < MAX_REFLECTION_ROUNDS:
                    feedback = f"审核出错({e})，请重新检测"
                    continue
                raise RuntimeError("检测章节失败") from e
            logger.info("[CHAPTERIZER] round %d review: %.1fs -> verdict=%s",
                         round_num, _time.time() - _t_review, review.get("verdict"))

            # ---- 应用审核结果：逐项过滤无效章节 ----
            reviewed = review.get("chapters", [])
            if reviewed:
                valid_chapters = []
                removed = []
                for i, ch in enumerate(chapters):
                    if i < len(reviewed) and not reviewed[i].get("valid", True):
                        reason = reviewed[i].get("reason", "未说明")
                        removed.append(f"'{ch.title}' ({reason})")
                        logger.info("[CHAPTERIZER] round %d removed: '%s' — %s",
                                     round_num, ch.title, reason)
                    else:
                        valid_chapters.append(ch)
                chapters = valid_chapters
                if removed:
                    logger.info("[CHAPTERIZER] round %d filtered: removed %d/%d — %s",
                                 round_num, len(removed), len(removed) + len(chapters),
                                 "; ".join(removed[:5]))

            if not chapters:
                if round_num < MAX_REFLECTION_ROUNDS:
                    feedback = "所有章节均被审核剔除，请重新检测真正的章节标题"
                    continue
                raise RuntimeError("检测章节失败")

            if review.get("verdict") == "all_valid":
                await self._report(on_progress, fname,
                                   f"审核通过：{len(chapters)} 个章节", 2, 2)
                logger.info("[CHAPTERIZER] ====== DONE in %.1fs, %d chapters ======",
                             _time.time() - _total_t0, len(chapters))
                return chapters

            # 审核不通过 → 收集反馈，下一轮
            issues = review.get("issues", [])
            feedback = "；".join(issues) if issues else "部分章节不符合要求，请重新检测"
            logger.info("[CHAPTERIZER] round %d feedback: %s", round_num, feedback[:200])

        # 超出最大轮次 — 返回已有的章节（若有），不因少数缺失而全盘否定
        if chapters:
            logger.warning("[CHAPTERIZER] ====== PARTIAL after %.1fs, %d chapters (review issues unresolved) ======",
                          _time.time() - _total_t0, len(chapters))
            return chapters
        logger.error("[CHAPTERIZER] ====== FAILED after %.1fs (max rounds exceeded) ======",
                      _time.time() - _total_t0)
        raise RuntimeError("检测章节失败")

    async def _llm_detect(self, doc: Document, feedback: str = "") -> list[ChapterInfo]:
        """Async: 调用 LLM 从文档前部定位目录，提取章节名和起始标记."""
        import time as _t
        llm = self._get_detect_llm()
        text = doc.content

        # OCR 文本可能很长（15K-30K 字），全文直传会降低 LLM 检测质量。
        # 智能截断：保留足够空间容纳完整目录 + 若干章节正文。
        MAX_CHARS = 15000
        truncated = False
        if len(text) > MAX_CHARS:
            text = text[:MAX_CHARS]
            truncated = True
            logger.info(
                "[CHAPTERIZER] text truncated: %d → %d chars for LLM detect",
                len(doc.content), MAX_CHARS,
            )

        prompt = CHAPTER_DETECTOR_PROMPT
        truncation_note = (
            f"（注：原文共 {len(doc.content)} 字，已截断至前 {MAX_CHARS} 字。"
            f"如目录完整，请据此提取所有章节；如章节列表被截断，仅返回已确认的部分。）"
        ) if truncated else ""
        user_msg = f"文档：{doc.filename}（前 20 页，共 {len(doc.content)} 字）{truncation_note}\n\n{text}"

        if feedback:
            prompt += f"\n\n**上一轮审核反馈：**\n{feedback}\n请根据反馈重新检测，排除误检项。"

        _t0 = _t.time()
        resp = await llm.ainvoke([
            SystemMessage(content=prompt),
            HumanMessage(content=user_msg),
        ])
        logger.info("Chapterizer LLM detect: %.1fs for %s", _t.time() - _t0, doc.filename)

        return self._parse_chapter_response(resp, doc, text)

    def _parse_chapter_response(self, resp, doc: Document, text: str) -> list[ChapterInfo]:
        """解析 LLM 响应，提取章节列表并用 start_marker 切分文本."""
        resp_text = resp.content if hasattr(resp, "content") else str(resp)
        m = re.search(r'\{.*\}', resp_text, re.DOTALL)
        if not m:
            return []
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            return []

        meta = data.get("chapters", [])
        if not meta:
            return []

        for ch in meta:
            ch["level"] = 1

        return _split_by_meta(text, meta)

    # ==================================================================
    # LLM: 审核专家
    # ==================================================================

    async def _llm_review(self, doc: Document, chapters: list[ChapterInfo]) -> dict:
        """Async: 调用 LLM 审核章节列表."""
        import time as _t
        _t0 = _t.time()
        llm = self._get_review_llm()

        chapter_summary = []
        for i, ch in enumerate(chapters):
            preview = ch.text[:150].replace('\n', ' ').strip()
            chapter_summary.append(
                f"[{i}] 标题: {ch.title} | 长度: {ch.text_length}字\n    内容预览: {preview}..."
            )

        review_input = (
            f"文档：{doc.filename}（{len(doc.content)}字）\n\n"
            f"检测到的 {len(chapters)} 个章节：\n\n" +
            "\n\n".join(chapter_summary)
        )

        resp = await llm.ainvoke([
            SystemMessage(content=CHAPTER_REVIEWER_PROMPT),
            HumanMessage(content=review_input),
        ])
        resp_text = resp.content if hasattr(resp, "content") else str(resp)

        m = re.search(r'\{.*\}', resp_text, re.DOTALL)
        if not m:
            return {"verdict": "all_valid", "issues": [], "chapters": []}
        try:
            result = json.loads(m.group(0))
        except json.JSONDecodeError:
            result = {"verdict": "all_valid", "issues": [], "chapters": []}

        logger.info("Chapterizer LLM review: %.1fs for %s (%d chapters, verdict=%s)",
                     _t.time() - _t0, doc.filename, len(chapters),
                     result.get("verdict", "?"))
        return result

    # ==================================================================
    # Helpers
    # ==================================================================

    async def _report(self, cb, fname, stage, done, total):
        if cb:
            await cb(fname, stage, done, total)


# ===========================================================================
# Utilities
# ===========================================================================

def _normalize_title(raw: str) -> str:
    """Clean chapter title with OCR correction."""
    title = raw.strip()
    if not title:
        return title
    title = re.sub(r"^#{1,3}\s+", "", title)
    title = re.sub(r"\s+", " ", title)
    title = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", title)
    # OCR corrections for common PDF extraction errors
    title = re.sub(r"([第十])童", r"\1章", title)   # X童 -> X章
    title = re.sub(r"([第十])量", r"\1章", title)   # X量 -> X章
    prefix_cn = re.match(r"^(第\s*[零〇一二三四五六七八九十百千\d]+\s*[章童篇部单元])", title)
    if prefix_cn:
        rest = title[prefix_cn.end():].strip()
        clean_prefix = prefix_cn.group(1).replace("童", "章").replace("篇", "章").replace("部", "章")
        title = clean_prefix + "：" + rest if rest else clean_prefix
    prefix_en = re.match(r"^(Chapter\s+\d+)", title, re.IGNORECASE)
    if prefix_en:
        rest = title[prefix_en.end():].strip()
        title = prefix_en.group(1) + ": " + rest if rest else prefix_en.group(1)
    title = re.sub(r"[\s.。…\-—：:，、]+$", "", title)
    return title.strip()

def _clean(title):
    return _normalize_title(title)

def _collapse_ws(s: str) -> str:
    """Collapse whitespace + normalize colons for fuzzy matching.

    Chinese colons (：) inserted by _normalize_title don't exist in
    raw/OCR text (which uses spaces or nothing). Normalizing both to
    spaces ensures they match regardless of formatting.
    """
    s = re.sub(r'[：:]+', ' ', s)
    return re.sub(r'\s+', ' ', s).strip()


# ---- 章节 title 合法性正则 ----
_VALID_CN_CHAPTER = re.compile(
    r'^(第\s*[零〇一二三四五六七八九十百千\d]+\s*[章节篇部])'  # 第一章 / 第十二章
    r'|^(第\s*[零〇一二三四五六七八九十百千\d]+\s*[课讲])'    # 第一课 / 第三讲
    r'|^([零〇一二三四五六七八九十]+[、，,]\s*)'              # 一、 / 二、
    r'|^(Chapter\s+\d+)'                                      # Chapter 1
    r'|^(Part\s+\d+)'                                         # Part 1
    r'|^(Section\s+\d+)'                                      # Section 1
    r'|^(\d+[\.\、]\s+\w)'                                     # 1. xxx / 1、xxx
)
# 明确是噪音的 pattern：公式编号、图表编号、纯数字编号
_NOISE_PATTERNS = re.compile(
    r'^[\d\s\.\-\+\(\)\[\]f\(t,z\)图 Table Figure 表]+$'  # 全是数字/符号
    r'|图\s*\d+|表\s*\d+|Fig|Table'                         # 图表编号
    r'|^\d+\.\d+\.\d+'                                      # 1.1.1 二级标题
    r'|^[\(（][\d一二三四五六七八九十]+[\)）]'                     # (一) (1) 可能是一级但更可能是小节
)


def _fast_check_pass(chapters: list) -> bool:
    """Heuristic check: if all detected chapter titles look like proper
    first-level chapters, skip the LLM review call.

    Returns True when we're confident enough to bypass review.
    """
    if len(chapters) < 2:
        return False  # too few, better let LLM review

    valid_count = 0
    for ch in chapters:
        title = ch.title.strip()
        if not title:
            return False  # empty title → suspicious, needs review
        # Noise check
        if _NOISE_PATTERNS.search(title):
            return False  # smells like a formula/figure ref, needs review
        # Valid chapter pattern check
        if _VALID_CN_CHAPTER.match(title):
            valid_count += 1

    # Pass only when ≥ 80% of titles match standard chapter patterns
    ratio = valid_count / len(chapters)
    return ratio >= 0.8


def _is_toc_region(text: str, pos: int, window: int = 400) -> bool:
    """Check if a position is likely inside a TOC (dense cluster of chapter markers)."""
    ctx_start = max(0, pos - window // 2)
    ctx_end = min(len(text), pos + window // 2)
    ctx = text[ctx_start:ctx_end]
    chapter_markers = re.findall(
        r'第[零〇一二三四五六七八九十百千\d]+[章童篇]|Chapter\s+\d+',
        ctx,
    )
    # ≥3 chapter-like patterns in a small window → likely TOC
    return len(chapter_markers) >= 3


def _find_skip_toc(text: str, pattern: str, start: int = 0) -> int:
    """Find pattern in text, skipping matches that fall in TOC-dense regions."""
    pos = text.find(pattern, start)
    while pos >= 0:
        if not _is_toc_region(text, pos):
            return pos
        pos = text.find(pattern, pos + 1)
    return -1


def _split_by_meta(text: str, meta: list[dict]) -> list[ChapterInfo]:
    """用 LLM 检测的标记拆分文档（支持空格归一化 + OCR 纠错匹配）.

    核心改进：fallback 定位时利用 norm_text 找到的近似位置做窗口搜索，
    避免用短前缀全局查找时命中目录页而非正文。
    """
    norm_text = _collapse_ws(text)
    positions = []
    for ch in meta:
        marker = ch.get("start_marker", "").strip()
        title = ch.get("title", "").strip()
        pos = -1

        # 1) Exact match — 但跳过 TOC 区域
        for cand in [marker, title]:
            if cand:
                pos = _find_skip_toc(text, cand)
                if pos >= 0:
                    logger.debug("_split_by_meta: exact match '%s' at pos %d", cand[:40], pos)
                    break

        # 2) Collapsed-whitespace match — 利用 norm_pos 做窗口搜索
        if pos < 0:
            for cand in [marker, title]:
                norm_cand = _collapse_ws(cand)
                if norm_cand and len(norm_cand) >= 6:
                    norm_pos = norm_text.find(norm_cand)
                    if norm_pos >= 0:
                        # 用 norm_pos 估算原文位置，窗口搜索原始 cand
                        win_start = max(0, norm_pos - 300)
                        win_end = min(len(text), norm_pos + len(cand) + 300)
                        pos = text.find(cand, win_start, win_end)
                        if pos >= 0:
                            logger.debug(
                                "_split_by_meta: norm-window match '%s' at pos %d "
                                "(norm_pos=%d, window=[%d:%d])",
                                cand[:40], pos, norm_pos, win_start, win_end,
                            )
                            break
                        # 窗口搜 cand 失败，退而用 prefix 在窗口搜
                        prefix = norm_cand.split()[0] if norm_cand.split() else norm_cand[:4]
                        if len(prefix) >= 4:
                            pos = text.find(prefix, win_start, win_end)
                            if pos >= 0:
                                logger.debug(
                                    "_split_by_meta: norm-window prefix match '%s' at pos %d",
                                    prefix, pos,
                                )
                                break

        # 3) Chapter prefix fallback with OCR correction — 跳过 TOC 区域
        if pos < 0:
            prefix_match = re.match(
                r'^(第[零〇一二三四五六七八九十百千\d]+)([章童篇])|^(Chapter\s+\d+)',
                title,
            )
            if prefix_match:
                prefix = prefix_match.group(1) or prefix_match.group(3) or ''
                marker_char = prefix_match.group(2) or ''
                if marker_char:
                    prefix = prefix + '章'  # normalize 童/篇 → 章
                if len(prefix) >= 4:
                    pos = _find_skip_toc(text, prefix)
                    if pos >= 0:
                        logger.debug(
                            "_split_by_meta: prefix fallback '%s' at pos %d (skipped TOC)",
                            prefix, pos,
                        )

        if pos < 0:
            logger.warning(
                "_split_by_meta: could not locate chapter '%s' (marker='%s') in text",
                title[:60], marker[:60],
            )

        positions.append((pos if pos >= 0 else 999999, title, marker))

    # Sort and deduplicate
    positions.sort(key=lambda x: x[0])
    seen, filtered, skipped_titles = set(), [], []
    for pos, title, marker in positions:
        if pos < 999999 and pos not in seen:
            seen.add(pos)
            filtered.append((pos, title, marker))
        elif pos >= 999999:
            skipped_titles.append(title)

    chapters = []
    for i, (pos, title, marker) in enumerate(filtered):
        next_positions = [p for p, _, _ in filtered[i + 1:] if p > pos]
        end = min(next_positions) if next_positions else len(text)
        ch_text = text[pos:end].strip()
        if not ch_text or len(ch_text) < 20:
            logger.debug("_split_by_meta: chapter '%s' too short (%d chars), skipping",
                         title, len(ch_text) if ch_text else 0)
            continue
        chapters.append(ChapterInfo(
            title=_normalize_title(title) or f"章节{i + 1}",
            level=1,
            start_marker=marker,
            text=ch_text,
            text_length=len(ch_text),
        ))

    # Append title-only chapters for any that could not be located in text.
    # Common when: (a) TOC is in first 20 pages but chapter body headings
    # are beyond page 20, or (b) OCR text has errors preventing exact match.
    if skipped_titles:
        logger.info(
            "_split_by_meta: %d/%d chapters located, adding %d title-only",
            len(chapters), len(positions), len(skipped_titles),
        )
        for title in skipped_titles:
            clean_title = _normalize_title(title)
            if not clean_title:
                continue
            chapters.append(ChapterInfo(
                title=clean_title,
                level=1,
                start_marker=title,
                text=f"{clean_title}（目录检测，正文在20页之后）",
                text_length=0,
            ))

    return [c for c in chapters if c.text] if any(c.text_length > 0 for c in chapters) else chapters


# Backward-compat
def detect_chapters(doc: Document, config: Configuration) -> list[ChapterInfo]:
    agent = ChapterizerAgent(config)
    return asyncio.run(agent.detect_all(doc))
