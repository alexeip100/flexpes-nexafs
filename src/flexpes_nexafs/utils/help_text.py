"""Utilities for loading and rendering packaged help/usage text.

Help files are shipped under docs/ (e.g. usage_controls.md, usage_workflows.md).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional


def _pkg_root_path():
    try:
        from pathlib import Path
        return Path(__file__).resolve().parent.parent
    except Exception:
        return None

def _read_help_markdown(md_filename: str = "help.md"):
    root = _pkg_root_path()
    if not root:
        return None
    md_path = root / "docs" / md_filename
    try:
        return md_path.read_text(encoding="utf-8")
    except Exception:
        return None

def _basic_md_to_html(md: str) -> str:
    """Very small Markdown->HTML fallback.

    This version is careful not to treat every single line break in the
    source file as a separate paragraph. Instead, consecutive non-empty
    lines that are not list items or headings are merged into one
    paragraph, so that soft-wrapped text from the editor behaves like a
    normal flowing paragraph in the help window.

    It supports:
      - #, ##, ### headings
      - unordered lists starting with "- " or "* "
      - ordered lists starting with "1. ", "2. ", ... "9. "
      - **bold**, *italic*, and `inline code`
    """

    import html, re

# Escape HTML first, then re-introduce simple formatting

    text = html.escape(md)

# inline code

    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)

# bold / italic

    text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)

    text = re.sub(r"\*([^*]+)\*", r"<i>\1</i>", text)

# Now process line-by-line with simple block structure

    import re as _re


    def _slugify(s: str) -> str:

        s = (s or "").strip().lower()

        s = _re.sub(r"[\s_]+", "-", s)

        s = _re.sub(r"[^a-z0-9\-]+", "", s)

        s = _re.sub(r"-{2,}", "-", s).strip("-")

        return s or "section"


    _used_ids = {}

    lines_out = []

    in_ul = False

    in_ol = False

    para_buf = []  # accumulate plain paragraph lines

    def flush_para():

        nonlocal para_buf

        if para_buf:

# Join soft-wrapped lines with spaces into one logical paragraph

            lines_out.append(f"<p>{' '.join(para_buf).strip()}</p>")

            para_buf = []

    for raw in text.splitlines():

        line = raw.strip()

        if not line:

# Blank line: end any running paragraph, leave lists open

            flush_para()

            continue

# Headings

        if line.startswith("# ") or line.startswith("## ") or line.startswith("### "):

            flush_para()

            if in_ul:

                lines_out.append("</ul>"); in_ul = False

            if in_ol:

                lines_out.append("</ol>"); in_ol = False

            if line.startswith("# "):

                _title = line[2:].strip()
                _id = _slugify(_title)
                _used_ids[_id] = _used_ids.get(_id, 0) + 1
                if _used_ids[_id] > 1:
                    _id = f"{_id}-{_used_ids[_id]}"
                lines_out.append(f'<h1 id="{_id}">{_title}</h1>')

            elif line.startswith("## "):

                _title = line[3:].strip()
                _id = _slugify(_title)
                _used_ids[_id] = _used_ids.get(_id, 0) + 1
                if _used_ids[_id] > 1:
                    _id = f"{_id}-{_used_ids[_id]}"
                lines_out.append(f'<h2 id="{_id}">{_title}</h2>')

            else:

                lines_out.append(f"<h3>{line[4:].strip()}</h3>")

            continue

# Unordered list item

        if line.startswith("- ") or line.startswith("* "):

            flush_para()

            if in_ol:

                lines_out.append("</ol>"); in_ol = False

            if not in_ul:

                lines_out.append("<ul>"); in_ul = True

            item = line[2:].strip()

            lines_out.append(f"<li>{item}</li>")

            continue

# Ordered list item (1. 2. ... 9.)

        if any(line.startswith(f"{n}. ") for n in range(1, 10)):

            flush_para()

            if in_ul:

                lines_out.append("</ul>"); in_ul = False

            if not in_ol:

                lines_out.append("<ol>"); in_ol = True

            dot = line.find('.')

            item = line[dot+1:].strip()

            lines_out.append(f"<li>{item}</li>")

            continue

# Otherwise part of a normal paragraph: accumulate

        para_buf.append(line)

# Flush trailing paragraph and lists

    flush_para()

    if in_ul:

        lines_out.append("</ul>")

    if in_ol:

        lines_out.append("</ol>")

    return "\n".join(lines_out)
def get_usage_html(md_filename: str = "help.md") -> str:
    """Return help content as HTML from a Markdown file under docs/."""
    md = _read_help_markdown(md_filename)
    if not md:
        return "<p><b>Help file not found.</b></p>"
    try:
        import markdown
        # "toc" adds stable id attributes to headings (used for in-text links and
        # the HelpBrowser right-click "Go to" menu).
        return markdown.markdown(md, extensions=["tables","fenced_code","sane_lists","toc"])  # type: ignore
    except Exception:
        return _basic_md_to_html(md)



# ---------------------------------------------------------------------------
# What's new / changelog rendering
# ---------------------------------------------------------------------------

def _parse_changelog_sections(md: str):
    """Parse Keep-a-Changelog style sections.

    Expected headings:
      ## [Unreleased]
      ## [2.3.8] – YYYY-MM-DD

    Returns:
      (unreleased_block, version_blocks) where version_blocks is a list of
      (version_str, date_str, block_md) in the order they appear in the file.
    """
    import re

    # Split into blocks starting at "## ["
    m = re.search(r"^## \[", md, flags=re.M)
    if not m:
        return None, []

    rest = md[m.start():]

    starts = []
    for match in re.finditer(r"^## \[(?P<tag>[^\]]+)\]\s*[-–—]\s*(?P<date>\d{4}-\d{2}-\d{2}).*$", rest, flags=re.M):
        starts.append((match.group("tag").strip(), match.group("date").strip(), match.start()))

    # Also catch "[Unreleased]" without date
    for match in re.finditer(r"^## \[(?P<tag>Unreleased)\].*$", rest, flags=re.M | re.I):
        # ensure it's not already in starts
        tag = match.group("tag").strip()
        if not any(t.lower() == tag.lower() and s == match.start() for t, _, s in starts):
            starts.append((tag, "", match.start()))

    # sort by position
    starts.sort(key=lambda x: x[2])

    blocks = []
    for i, (tag, d, start) in enumerate(starts):
        end = starts[i+1][2] if i+1 < len(starts) else len(rest)
        blocks.append((tag, d, rest[start:end].strip()))

    unreleased = None
    versions = []
    for tag, d, block in blocks:
        if tag.lower() == "unreleased":
            unreleased = block
        else:
            versions.append((tag, d, block))

    return unreleased, versions


def _semver_key(v: str):
    """Return a sortable key for versions like '2.3.8'."""
    import re
    s = (v or "").strip()
    if s.lower().startswith("v"):
        s = s[1:]
    m = re.match(r"^(\d+)\.(\d+)(?:\.(\d+))?", s)
    if not m:
        return None
    major = int(m.group(1))
    minor = int(m.group(2))
    patch = int(m.group(3) or 0)
    return (major, minor, patch)


def build_whats_new_markdown(current_version: str, max_versions: int = 5) -> tuple[str, str]:
    """Build a markdown document for 'What's new' from docs/CHANGELOG.md.

    Returns:
      (md, latest_version_str)
    """
    md = _read_help_markdown("CHANGELOG.md")
    if not md:
        return ("# What's new\n\n**Changelog could not be loaded.**\n", "")

    _unrel, versions = _parse_changelog_sections(md)
    if not versions:
        return ("# What's new\n\n**No released versions found in changelog.**\n", "")

    # Sort by semver desc when possible, otherwise keep file order
    versions_sorted = []
    versions_unparsable = []
    for tag, d, block in versions:
        key = _semver_key(tag)
        if key is None:
            versions_unparsable.append((tag, d, block))
        else:
            versions_sorted.append((key, tag, d, block))

    versions_sorted.sort(key=lambda x: x[0], reverse=True)
    merged = [(tag, d, block) for _, tag, d, block in versions_sorted] + versions_unparsable
    top = merged[:max_versions]

    latest_version = top[0][0] if top else ""

    out = []
    out.append("# What's new")
    out.append("These are the latest changes recorded in the packaged changelog.")
    out.append("")

    # Create version headings as H3 so TOC shows versions (H3 is used by the Usage TOC style).
    for idx, (tag, d, block) in enumerate(top):
        title = f"{tag} — {d}" if d else tag
        out.append(f"### {title}")
        out.append("")
        # Remove the leading '## [x] ...' line inside block and demote '###' to '####'
        lines = block.splitlines()
        if lines and lines[0].startswith("## "):
            lines = lines[1:]
        content = "\n".join(lines).strip()
        # Demote headings inside each version block to avoid flooding the TOC.
        # Do it consistently for all same-level headings (e.g. "### Fixed" and "### Changed")
        # so equal-level headings render with equal font size.
        import re
        content = re.sub(r"^###\s+", "#### ", content, flags=re.M)
        content = re.sub(r"^##\s+", "#### ", content, flags=re.M)
        out.append(content)
        out.append("")
        if idx != len(top) - 1:
            out.append("---")
            out.append("")
    return ("\n".join(out).strip() + "\n", latest_version)


def get_whats_new_payload(current_version: str, max_versions: int = 5) -> tuple[str, str]:
    """Return (html, latest_version) for the What's new window."""
    md, latest = build_whats_new_markdown(current_version=current_version, max_versions=max_versions)
    try:
        import markdown
        html = markdown.markdown(md, extensions=["tables","fenced_code","sane_lists","toc"])  # type: ignore
    except Exception:
        html = _basic_md_to_html(md)
    return html, latest
