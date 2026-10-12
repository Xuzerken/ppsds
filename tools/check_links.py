#!/usr/bin/env python3
"""Check that the relative links in **the Markdown that ships with the repo** resolve.

Markdown renderers (GitHub, the VS Code preview) resolve relative links against **the
directory the file lives in**; writing them repo-root-relative (e.g. `docs/plans/x.md`)
fails silently -- you click and nothing happens. This script finds such broken links.

**Scope (important)**: this is a **release gate**, so it only covers the documents that
actually go out. **Development records are not checked** -- `docs/logs/`, `docs/plans/`,
`docs/notes/` record what was true at the time and cite the paths of that time; after code
moves or a refactor their links are **necessarily and legitimately** broken (records should not
be rewritten). Including them would keep the gate permanently red, and eventually nobody would
look at it again.

The same script serves **both** repositories: the private development one (which has
``README.md`` and ``CLAUDE.md``) and the public release one (which has the community files and
no ``CLAUDE.md``). :data:`INCLUDE_FILES` is therefore a list of **candidates**, and only the
ones that exist are checked.

Usage::

    python tools/check_links.py         # only report broken links
    python tools/check_links.py -v      # also list skipped targets and passing links
    python tools/check_links.py --pypi-readme
                                        # additionally check README.md against the two rules
                                        # PyPI imposes and GitHub does not

Exit code: 0 = everything passes; 1 = broken or malformed links (or, with ``--pypi-readme``,
a README that PyPI would render wrong).
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import urllib.parse

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# [label](target "optional title")  -- the title part is optional
LINK_RE = re.compile(r'\[[^\]]*\]\(\s*([^)\s]+?)(?:\s+"[^"]*")?\s*\)')

# Loose version: ignore the target's internal structure and just delimit ](...), so that
# **malformed** links can be found. CommonMark says a link target without angle brackets may
# not contain a space -- a bare space ends the target, the whole link fails to parse, and it
# renders as plain text. File names in this repo routinely contain spaces and rely entirely on
# %20 encoding, so this is easy to trip over.
LOOSE_LINK_RE = re.compile(r'\]\(\s*([^)]+?)\s*\)')
TITLE_RE = re.compile(r'\s+("[^"]*"|\'[^\']*\')\s*$')

#: What to check -- **listed explicitly**, rather than "the whole repo minus a few". The
#: reason is that each entry here answers "does this document get published?"; omitting one
#: under-checks a single file, whereas wrongly including one keeps the gate permanently red.
#: Not included: development records (see below), `.claude/` (workflow config), `papers/`
#: (third-party source), `examples/` and `src/` (code).
#:
#: These are **candidates**: only the ones that exist are checked, which is what lets the same
#: script run in the private repository (``README.md``, ``CLAUDE.md`` present; the community
#: files absent) and in the public one (the reverse).
INCLUDE_FILES = (
    'README.md',
    'README.zh-CN.md',
    'CLAUDE.md',
    'CHANGELOG.md',
    'CONTRIBUTING.md',
    'RELEASING.md',
    'SECURITY.md',
    'CODE_OF_CONDUCT.md',
)
INCLUDE_DIRS = ('docs',)

#: **Development-record** subdirectories under `docs/` -- not checked. They record what was
#: true at the time and cite the paths of that time; after code moves or a refactor their
#: links are **necessarily and legitimately** broken (records should not be rewritten).
SKIP_RECORD_DIRS = {'docs/logs', 'docs/plans', 'docs/notes'}

EXTERNAL_PREFIXES = ('http://', 'https://', 'mailto:', 'ftp://', 'data:')


def iter_markdown_files() -> list[str]:
    """Markdown files to check (repo-root-relative, sorted) -- i.e. "the ones that ship"."""
    found = []

    for rel in INCLUDE_FILES:
        if os.path.isfile(os.path.join(REPO_ROOT, rel)):
            found.append(rel)

    for root in INCLUDE_DIRS:
        for dirpath, dirnames, filenames in os.walk(os.path.join(REPO_ROOT, root)):
            reldir = os.path.relpath(dirpath, REPO_ROOT).replace(os.sep, '/')
            dirnames[:] = sorted(d for d in dirnames if f'{reldir}/{d}' not in SKIP_RECORD_DIRS)
            for name in filenames:
                if name.endswith('.md'):
                    found.append(f'{reldir}/{name}')

    return sorted(set(found))


def strip_angle(target: str) -> str:
    """Remove CommonMark's angle-bracket form ``<...>``.

    ``[text](<a b.md>)`` is the **standard** way to write a target containing a space
    (markdown formatters produce it routinely) and is equivalent to ``%20``; both must be
    accepted.
    """
    if target.startswith('<') and target.endswith('>'):
        return target[1:-1]
    return target


def is_path_like(target: str) -> bool:
    """Distinguish real links from notation in the prose that merely looks like one (e.g. ``gamma[0](x,u)``).

    The test: the target contains ``/``, or its basename contains a ``.`` (has an extension).
    Something with neither is essentially never a file link.
    """
    path = target.partition('#')[0]
    if not path:
        return False
    basename = os.path.basename(path)
    return '/' in path or '.' in basename


def line_of(text: str, pos: int) -> int:
    return text.count('\n', 0, pos) + 1


def label_of(text: str, start: int) -> str:
    """The label text at the match site.

    Only search back for ``[`` **within the same line** -- otherwise it picks up unrelated
    brackets from far away (``[0]`` citations in prose, list markers, ...) and the output
    misleads instead of helping. Returns the empty string if none is found.
    """
    line_start = text.rfind('\n', 0, start) + 1
    open_bracket = text.rfind('[', line_start, start)
    return text[open_bracket + 1:start] if open_bracket != -1 else ''


def find_malformed(text: str, strict_spans: list[tuple[int, int]]) -> list[tuple[int, str]]:
    """Find malformed links: ``](...)`` whose target contains an unencoded space.

    The test is "the loose regex matches but the strict one does not" -- the strict regex
    truncates at the space per CommonMark, so such a construct is not a link at all in a
    renderer.
    """
    def already_matched(pos: int) -> bool:
        return any(a <= pos < b for a, b in strict_spans)

    out = []
    for m in LOOSE_LINK_RE.finditer(text):
        if already_matched(m.start()):
            continue
        raw = m.group(1).strip()
        if raw.startswith('<') and raw.endswith('>'):
            continue                        # angle-bracket form: a space is legal, not malformed
        dest = TITLE_RE.sub('', raw).strip()
        if not dest or dest.startswith(('<', '#')) or dest.startswith(EXTERNAL_PREFIXES):
            continue
        if not is_path_like(dest):          # filter out math notation like gamma[0](x, u)
            continue
        if ' ' in dest or '\t' in dest:
            out.append((line_of(text, m.start()), dest))
    return out


def check(verbose: bool) -> tuple[list[tuple[str, int, str, str]], list[tuple[str, int, str]]]:
    """Return (broken links, malformed links).

    Broken: (file, line, label, target); malformed: (file, line, target).
    """
    broken: list[tuple[str, int, str, str]] = []
    malformed: list[tuple[str, int, str]] = []
    n_ok = n_skipped = 0

    for rel in iter_markdown_files():
        try:
            text = open(os.path.join(REPO_ROOT, rel), encoding='utf-8').read()
        except (OSError, UnicodeDecodeError) as exc:
            print(f'  ! cannot read {rel}: {exc}', file=sys.stderr)
            continue

        base = os.path.dirname(rel)
        strict_spans = [m.span() for m in LINK_RE.finditer(text)]

        for m in LINK_RE.finditer(text):
            target = strip_angle(m.group(1))
            if target.startswith(EXTERNAL_PREFIXES) or target.startswith('#'):
                continue
            if not is_path_like(target):
                n_skipped += 1
                if verbose:
                    print(f'  skip (not path-like)  {rel}:{line_of(text, m.start())}  '
                          f'[{label_of(text, m.start())}] -> {target}')
                continue

            path_part, _, anchor = target.partition('#')
            decoded = urllib.parse.unquote(path_part)
            resolved = os.path.normpath(os.path.join(base, decoded))

            if os.path.exists(os.path.join(REPO_ROOT, resolved)):
                n_ok += 1
                if verbose:
                    print(f'  OK   {rel}:{line_of(text, m.start())}  -> {target}')
                continue

            broken.append((rel, line_of(text, m.start()), label_of(text, m.start()), target))

        for lineno, dest in find_malformed(text, strict_spans):
            malformed.append((rel, lineno, dest))

    print(f'\nrelative links {n_ok + len(broken)}: {n_ok} resolve, {len(broken)} broken '
          f'({n_skipped} non-path-like targets skipped)')
    if malformed:
        print(f'WARNING: {len(malformed)} malformed link(s) (target contains an unencoded '
              f'space, so renderers do not treat it as a link)')
    return broken, malformed


# ----------------------------------------------------------------------
# The two rendering rules PyPI imposes and GitHub does not
# ----------------------------------------------------------------------

FENCE_MARKERS = ('```', '~~~')

#: Display-math delimiters. Inline ``$...$`` is deliberately not listed: a single ``$`` shows up
#: in shell snippets and prose often enough that flagging it would make the check noisy, and the
#: README is expected to keep formulas in fenced blocks anyway.
MATH_DELIMITERS = ('$$', r'\[')


def strip_code(text: str) -> str:
    """Blank out fenced code blocks and inline code spans, preserving line and column layout.

    Neither check below may fire on text a reader sees as literal code: a ``$$`` inside a fenced
    block is the very fix this script recommends, and a path-like string inside a code sample is
    not a link. Blanked rather than deleted, so reported line numbers stay true.
    """
    out = []
    fence = None
    for line in text.split('\n'):
        stripped = line.lstrip()
        marker = next((m for m in FENCE_MARKERS if stripped.startswith(m)), None)
        if marker is not None:
            fence = marker if fence is None else (None if marker == fence else fence)
            out.append('')
            continue
        if fence is not None:
            out.append('')
            continue
        out.append(re.sub(r'`[^`]*`', lambda m: ' ' * len(m.group(0)), line))
    return '\n'.join(out)


#: The READMEs that become the project page. ``public/README.md`` is the one that ships; the
#: root ``README.md`` is this repository's own, and is checked so that the rule is enforced
#: wherever a README is being written rather than only where it is finished.
PYPI_README_CANDIDATES = ('public/README.md', 'README.md')


def check_pypi_readme() -> tuple[list[tuple[str, int, str]], list[tuple[str, int, str]]]:
    """Check the README against the two rules PyPI imposes and GitHub does not.

    The README is the wheel's ``long_description``, so it **is** the project page. Two things
    that look fine on GitHub break there:

    1. **Relative links 404.** PyPI renders the README at ``https://pypi.org/project/ppsds/``,
       which is not a directory in the repository, so a relative target resolves nowhere.
    2. **LaTeX is not rendered.** ``$$...$$`` comes out as literal dollar signs.

    Both are invisible until after the upload -- and a version number, once used, can only be
    yanked, never reused. Hence a mechanical check before tagging rather than a careful read.

    Returns ``(relative_links, math_delimiters)`` as ``(file, line, text)`` triples.
    """
    relative: list[tuple[str, int, str]] = []
    math_hits: list[tuple[str, int, str]] = []

    for rel in PYPI_README_CANDIDATES:
        path = os.path.join(REPO_ROOT, rel)
        if not os.path.isfile(path):
            continue
        text = strip_code(open(path, encoding='utf-8').read())

        for match in LINK_RE.finditer(text):
            target = strip_angle(match.group(1))
            if target.startswith(EXTERNAL_PREFIXES) or target.startswith('#'):
                continue
            relative.append((rel, line_of(text, match.start()), target))

        math_hits += [(rel, line_of(text, m.start()), delim)
                      for delim in MATH_DELIMITERS
                      for m in re.finditer(re.escape(delim), text)]

    return relative, sorted(math_hits)


def report_pypi_readme(relative: list[tuple[str, int, str]],
                       math_hits: list[tuple[str, int, str]]) -> None:
    if relative:
        print('\nFAIL: these READMEs have relative links, which 404 on PyPI:')
        for rel, lineno, target in relative:
            print(f'  {rel}:{lineno}')
            print(f'      -> {target}')
        print('      -> PyPI renders the README on the project page, which is not a directory in '
              'the repository, so a relative target resolves nowhere.')
        print('      -> use the absolute form, e.g. '
              'https://github.com/Xuzerken/ppsds/blob/main/<path>')
    if math_hits:
        print('\nFAIL: these READMEs contain LaTeX, which PyPI does not render:')
        for rel, lineno, delim in math_hits:
            print(f'  {rel}:{lineno}   {delim}')
        print('      -> it appears literally on the project page. Keep the LaTeX for GitHub and '
              'put a plain-text form in a fenced code block.')


def main() -> int:
    ap = argparse.ArgumentParser(description='Check that the relative links in Markdown documents resolve.')
    ap.add_argument('-v', '--verbose', action='store_true',
                    help='also list skipped targets and passing links')
    ap.add_argument('--pypi-readme', action='store_true',
                    help='additionally check README.md against the two rendering rules PyPI '
                         'imposes (absolute links only, no LaTeX), because the README is the '
                         'long description and therefore the project page')
    args = ap.parse_args()

    print(f'repo root: {REPO_ROOT}')
    broken, malformed = check(args.verbose)

    relative, math_hits = ([], [])
    if args.pypi_readme:
        relative, math_hits = check_pypi_readme()
        print(f'\nPyPI README check: {len(relative)} relative link(s), '
              f'{len(math_hits)} LaTeX delimiter(s)')

    if not broken and not malformed and not relative and not math_hits:
        print('OK: no broken links and no malformed links.')
        if args.pypi_readme:
            print('OK: README.md satisfies the PyPI rendering rules.')
        return 0

    if broken:
        print('\nFAIL: broken links:')
        for rel, lineno, label, target in broken:
            shown = f'[{label[:70]}] ' if label else ''
            print(f'  {rel}:{lineno}')
            print(f'      {shown}-> {target}')
            # The repo-root-relative form is the most common cause, so call it out separately
            decoded = urllib.parse.unquote(target.partition('#')[0])
            if os.path.exists(os.path.join(REPO_ROOT, decoded)):
                print('      -> the target exists relative to the repo root: this is the '
                      '"repo-root-relative" form, and should be rewritten relative to the '
                      'directory this file lives in')

    if malformed:
        print('\nFAIL: malformed links (target contains an unencoded space):')
        for rel, lineno, dest in malformed:
            print(f'  {rel}:{lineno}')
            print(f'      -> {dest}')
            print(f'      -> a space must be written as %20, otherwise it renders as plain '
                  f'text: {dest.replace(" ", "%20")}')

    if args.pypi_readme:
        report_pypi_readme(relative, math_hits)

    print('\nHow to fix: a link URL must be relative to **the directory this file lives in**; '
          'when the target contains a space, write it as %20, or wrap the whole URL in angle '
          'brackets.')
    return 1


if __name__ == '__main__':
    sys.exit(main())
