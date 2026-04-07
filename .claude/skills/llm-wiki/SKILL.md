---
name: llm-wiki
description: >-
  Maintain a persistent personal knowledge base (wiki) using LLMs. Incrementally
  builds and maintains a structured, interlinked collection of markdown files
  from raw sources. Handles ingesting new documents, querying accumulated
  knowledge, and linting the wiki for consistency. Use when adding sources to a
  knowledge base, building a research wiki, maintaining a personal wiki,
  organizing documents, or when the user mentions "wiki", "knowledge base",
  "ingest", or "personal wiki".
---

# LLM Wiki

Maintain a persistent, compounding knowledge base using LLMs.

## Architecture

Three layers:

1. **Raw sources** (`raw/`) - Immutable source documents (articles, papers, notes)
2. **The wiki** (`wiki/`) - LLM-generated markdown files with summaries, entity pages, concept pages
3. **The schema** (this skill) - Conventions for structure and workflows

Two special files:
- **index.md** - Content catalog with links and summaries, organized by category
- **log.md** - Chronological append-only record of all operations

## Directory Structure

```
wiki-project/
├── raw/                  # Source documents (immutable - COPY originals here)
│   ├── articles/
│   ├── papers/
│   └── assets/           # Downloaded images
├── wiki/                 # LLM-maintained knowledge base
│   ├── index.md          # Catalog of all pages
│   ├── entities/         # People, organizations, tools
│   ├── concepts/         # Ideas, frameworks, methods
│   ├── sources/          # Summaries of raw documents
│   └── synthesis/        # Higher-level analysis
└── log.md                # Chronological operation log
```

## Operations

### Ingest Workflow

Process a new source into the wiki. Follow steps in order:

1. **Copy raw source**
   - Copy the original file to appropriate `raw/` subdirectory
   - Preserve original filename for traceability
   - Example: `cp ~/Downloads/paper.pdf raw/papers/`

2. **Read source**
   - Read the document (use offset/limit for large PDFs >100K chars)
   - For PDFs exceeding limits: read first portion, then use Grep to find sections
   - Discuss key takeaways with user
   - Identify entities, concepts, and connections

3. **Create/update wiki pages**
   - Write/update source summary in `wiki/sources/`
   - Update/create entity pages in `wiki/entities/`
   - Update/create concept pages in `wiki/concepts/`
   - Add cross-references (wikilink format: `[[Page Name]]`)

4. **Update index.md**
   - Add new entries with links and one-line summaries
   - Maintain organization by category

5. **Log the operation**
   - Append to `log.md`: `## [YYYY-MM-DD] ingest | Source Title`
   - Include list of files created/modified

6. **Git sync (if applicable)**
   - Stage new files: `git add raw/ wiki/ log.md`
   - Commit with descriptive message including source name

### Query Workflow

Answer questions using the wiki:

1. **Read index.md** to locate relevant pages
2. **Read relevant wiki pages** for detailed information
3. **Synthesize answer** with citations to wiki pages
4. **File valuable answers** back into wiki as new pages if useful

### Lint Workflow

Health-check the wiki periodically:

1. **Check for contradictions** between pages
2. **Find stale claims** superseded by newer sources
3. **Identify orphan pages** with no inbound links
4. **Find missing concept pages** for important mentioned concepts
5. **Verify cross-references** are valid
6. **Suggest gaps** that could be filled with web search

## Page Conventions

### Wiki Page Template

```markdown
---
date: YYYY-MM-DD
tags: [tag1, tag2]
source_count: N
---

# Page Title

## Summary
One-paragraph overview.

## Details
Main content with sections as needed.

## Connections
- [[Related Entity]]
- [[Related Concept]]
- [[Source Document]]

## Sources
- [[Source Name]] (YYYY-MM-DD)
```

### Link Format

- Use wikilinks: `[[Page Name]]`
- Match page titles exactly (case-sensitive)
- Prefer descriptive link text: `[[Entity Name|display text]]`

### Log Entry Format

```markdown
## [YYYY-MM-DD] operation_type | Brief description

- Copied raw source to `raw/papers/filename.pdf` (size)
- Created [[Source Page]]
- Updated [[Entity]]
- Updated [[index.md]]
```

Operation types: `ingest`, `query`, `lint`, `update`

## Tips

- **Always copy raw sources** - Don't just read from original location; copy to `raw/` first
- **Handle large files** - Use Read offset/limit or Grep for PDFs over 100K characters
- **One source at a time** - Stay involved, guide emphasis
- **Obsidian compatibility** - Use wikilinks, YAML frontmatter, local images
- **Download images locally** - Use Obsidian's "Download attachments" feature
- **Graph view** - Use Obsidian's graph view to see wiki structure
- **Git version control** - Commit after each ingest; the wiki is a git repo
- **Check log.md exists** - Ensure log.md is a file, not a directory (use cat/echo to create)
