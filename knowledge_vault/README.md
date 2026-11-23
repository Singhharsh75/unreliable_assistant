# knowledge_vault (SAMPLE FIXTURES)

Invented test content, not real policy. Replace with the real SOPs/catalog.
The first 7 notes are the Gate 3 seed set; 29 more synthetic notes were added on 2026-11-22 so the
250-question golden set (eval/) has enough distinct facts. RAGAS scores measure this synthetic corpus.

Note format (Gate 17): YAML frontmatter `id`, `domain`, `risk_tier`, `version`, optional `tenants`
(list of tenant_ids allowed to read; omit = all tenants), then Markdown body with `[[wikilinks]]`.
