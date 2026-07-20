"""
Migrate existing chunk_ids to new chapter-aware format.

Before: 数据库系统概论（第5版） .pdf_0, ...pdf_69
After:  数据库系统概论（第5版） .pdf_Ch1_0, ...pdf_Ch1_69

Updates: ChromaDB, FTS5, KG concepts, merged_55.json.

Usage:
  python scripts/migrate_chunk_ids.py          # migrate + verify
  python scripts/migrate_chunk_ids.py --dry-run # check what would change
"""
from __future__ import annotations
import json, sys, re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _extract_ch_num(chapter_title: str) -> str:
    """第1章 绪论 -> 1"""
    cn = {'零': 0, '〇': 0, '一': 1, '二': 2, '三': 3, '四': 4,
          '五': 5, '六': 6, '七': 7, '八': 8, '九': 9, '十': 10}
    m = re.match(r'第\s*([零〇一二三四五六七八九十百千\d]+)\s*[章童篇]', chapter_title)
    if not m:
        m = re.match(r'Chapter\s+(\d+)', chapter_title, re.IGNORECASE)
    if m:
        s = m.group(1)
        if s.isdigit():
            return s
        # Parse Chinese numeral
        result = 0
        if '百' in s:
            parts = s.split('百')
            result += cn.get(parts[0], 0) * 100
            s = parts[1] if len(parts) > 1 else ''
        if '十' in s:
            parts = s.split('十')
            result += (cn.get(parts[0], 1) if parts[0] else 10) * 10
            s = parts[1] if len(parts) > 1 else ''
        if s:
            result += cn.get(s, 0)
        return str(result)
    return ''


def main(dry_run: bool = False):
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
    from src.context import init_context, get_context
    from src.config import Configuration
    init_context(Configuration())
    ctx = get_context()
    vs = ctx.vector_store
    kg = ctx.knowledge_graph

    # ── 1. Scan all chunks that need migration ──
    data = vs._collection.get(include=['embeddings', 'documents', 'metadatas'])
    total = len(data['ids'])

    migrate_list = []  # (old_id, new_id, embeddings, document, metadata)
    skip_count = 0
    for cid, emb, doc_text, meta in zip(data['ids'], data['embeddings'], data['documents'], data['metadatas']):
        chapter_title = meta.get('chapter_title', '')
        doc_filename = meta.get('doc_filename', '')

        # Only migrate chunks that have chapter info AND use old format
        if not chapter_title or '_Ch' in cid:
            skip_count += 1
            continue

        # Skip BEIR datasets
        if doc_filename.startswith('beir:'):
            skip_count += 1
            continue

        ch_num = _extract_ch_num(chapter_title)
        if not ch_num:
            skip_count += 1
            continue

        # Build new ID: {doc_filename}_Ch{N}_{chunk_index}
        chunk_idx = meta.get('chunk_index', 0)
        # Handle old format: filename_N or filename_N_vM
        new_id = f"{doc_filename}_Ch{ch_num}_{chunk_idx}"

        # Preserve _v suffix if present
        old_suffix_match = re.search(r'_v\d+$', cid.split(f'_{chunk_idx}')[-1] if f'_{chunk_idx}' in cid else '')
        if old_suffix_match:
            new_id += old_suffix_match.group(0)

        if cid != new_id:
            migrate_list.append((cid, new_id, emb, doc_text, meta, doc_filename, chapter_title))
        else:
            skip_count += 1

    print(f"Total chunks in VS: {total}")
    print(f"To migrate: {len(migrate_list)}")
    print(f"Already migrated / skip: {skip_count}")
    print()

    if not migrate_list:
        print("Nothing to migrate.")
        return

    # Show sample
    for old, new, _, _, _, _, _ in migrate_list[:5]:
        print(f"  {old}  ->  {new}")
    if len(migrate_list) > 5:
        print(f"  ... and {len(migrate_list) - 5} more")
    print()

    if dry_run:
        print("DRY RUN — no changes made.")
        return

    # ── 2. Migrate ChromaDB ──
    # ChromaDB doesn't support renaming IDs, so delete + re-add
    old_ids = [m[0] for m in migrate_list]
    try:
        vs._collection.delete(ids=old_ids)
        print(f"Deleted {len(old_ids)} old chunks from ChromaDB")
    except Exception as e:
        print(f"ERROR deleting from ChromaDB: {e}")
        return

    new_ids = []
    new_embeddings = []
    new_documents = []
    new_metadatas = []
    for old_id, new_id, emb, doc_text, meta, _, _ in migrate_list:
        new_ids.append(new_id)
        new_embeddings.append(emb)
        new_documents.append(doc_text)
        new_metadatas.append(meta)

    try:
        vs._collection.add(
            ids=new_ids,
            embeddings=new_embeddings,
            documents=new_documents,
            metadatas=new_metadatas,
        )
        print(f"Re-added {len(new_ids)} chunks with new IDs")
    except Exception as e:
        print(f"ERROR re-adding to ChromaDB: {e}")
        return

    # ── 3. Migrate FTS5 ──
    for old_id, new_id, _, _, _, _, _ in migrate_list:
        vs._fts_conn.execute(
            "UPDATE chunk_fts SET chunk_id = ? WHERE chunk_id = ?",
            (new_id, old_id),
        )
    vs._fts_conn.commit()
    print(f"Updated {len(migrate_list)} FTS5 rows")

    # ── 4. Migrate in-memory lists ──
    # _all_texts and _all_metas don't store chunk_id, they're index-based.
    # But the order changed because we deleted + re-added. Rebuild from ChromaDB.
    reloaded = vs._collection.get(include=['documents', 'metadatas'])
    vs._all_texts = list(reloaded['documents'])
    vs._all_metas = list(reloaded['metadatas'])
    vs._content_hashes = {vs._text_hash(t) for t in vs._all_texts}
    print("Rebuilt in-memory state from ChromaDB")

    # ── 5. Migrate KG concepts ──
    import sqlite3
    kg_conn = sqlite3.connect(kg._db_path)
    kg_updated = 0
    for old_id, new_id, _, _, _, doc_filename, _ in migrate_list:
        cursor = kg_conn.execute(
            "UPDATE concepts SET source_chunk_id = ? WHERE source_chunk_id = ?",
            (new_id, old_id),
        )
        kg_updated += cursor.rowcount
    kg_conn.commit()
    kg_conn.close()
    print(f"Updated {kg_updated} KG concept source_chunk_ids")

    # ── 6. Migrate merged_55.json ──
    eval_path = PROJECT_ROOT / "data/eval/merged_55.json"
    if eval_path.exists():
        eval_data = json.loads(eval_path.read_text(encoding="utf-8"))
        updated = 0
        for item in eval_data:
            new_chunks = []
            for rc in item.get('relevant_chunks', []):
                # Try to match against old IDs
                found = False
                for old_id, new_id, _, _, _, _, _ in migrate_list:
                    if rc == old_id:
                        new_chunks.append(new_id)
                        found = True
                        updated += 1
                        break
                if not found:
                    new_chunks.append(rc)  # Keep as-is if no match
            item['relevant_chunks'] = new_chunks
        eval_path.write_text(json.dumps(eval_data, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Updated {updated} relevant_chunks in merged_55.json")

    # ── 7. Verify ──
    print("\n=== Verification ===")
    data2 = vs._collection.get(include=['metadatas'])
    ch1_ids = [cid for cid, m in zip(data2['ids'], data2['metadatas'])
               if '数据库' in m.get('doc_filename', '') and '第1章' in m.get('chapter_title', '')]
    ch1_ids.sort()
    print(f"Chapter 1 chunks after migration: {len(ch1_ids)}")
    print(f"  First: {ch1_ids[0]}")
    print(f"  Last:  {ch1_ids[-1]}")
    all_have_ch = all('_Ch' in cid for cid in ch1_ids)
    print(f"  All have _Ch prefix: {all_have_ch}")
    print("\nMigration complete!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    main(dry_run=args.dry_run)
