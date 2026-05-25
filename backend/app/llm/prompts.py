"""Prompt templates for the two LLM call sites.

Plain f-string templates (per DESIGN §3 — explicit, not Jinja, because there
are exactly three). All placeholders are `{name}`-style so Python's `str.format`
is the only renderer needed.

Templates:
  SUMMARY_SHORT  — short article path: full cleaned body as input. Placeholder: {body}.
  SUMMARY_LONG   — long article path: structured outline (lede + per-section
                   heading + first sentence). Placeholder: {outline}.
  CHAT_GROUNDED  — chat path: numbered retrieved chunks + question. Placeholders: {chunks}, {question}.
"""

SUMMARY_SHORT = """You are summarising an English Wikipedia article.

Produce a 4-8 sentence summary that reflects the article. Do not invent facts.
Write plain prose, no headings or bullets.

Article body:
{body}

Write the 4-8 sentence summary now.
"""


SUMMARY_LONG = """You are summarising an English Wikipedia article. Below is its structured
outline: the lede paragraph in full, then each section heading with its
opening sentence. Produce a 4-8 sentence summary that reflects the full
article (not just the lede). Do not invent facts. Write plain prose, no
headings or bullets.

Outline:
{outline}

Write the 4-8 sentence summary now.
"""


CHAT_GROUNDED = """You are answering a question about an English Wikipedia article using ONLY
the retrieved passages below. If the passages do not contain the answer,
reply exactly: "not found in the article".

Do not use any outside knowledge. Do not invent facts. Do not emit citation
tags — the application attaches citations from the retriever output.

Retrieved passages:
{chunks}

Question: {question}

Answer:"""
