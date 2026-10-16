"""Where the *published* content lives, in each of the three trees the suite can run in.

The published content — the README a stranger reads, the documentation site, the figures — is
kept in ``public/`` here and **is the repository root** on the public side: ``tools/export_public.py``
promotes ``public/**`` and prunes the directory itself. A test that writes ``REPO_ROOT / 'public'``
therefore passes in the development repository and fails in the one place the public face actually
matters. That is backwards, and it is not hypothetical: two test modules were red on the public
side for that reason alone.

:func:`public_root` states the layout once instead of leaving each module to guess, and it returns
``None`` rather than falling back to the repository root, because falling back would be worse than
failing -- see the note on the third tree below.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def public_root() -> Path | None:
    """The directory holding ``README.md``, ``docs/`` and the site assets — or ``None``.

    Three trees can be told apart by one marker each, and all three cases are real:

    * **development repository** — ``public/mkdocs.yml``. Use ``public/``.
    * **public repository** — ``mkdocs.yml`` at the root, no ``public/``. Use the root.
    * **anywhere else** (an unpacked sdist, an installed wheel) — neither. Return ``None``, and
      the caller skips.

    ⚠️ The third case is why this cannot simply be "``public/`` if it exists, else the root". An
    sdist carries ``README.md`` — **the development README, in Chinese**, whose links point at
    private ``docs/`` — but neither ``public/`` nor ``mkdocs.yml``. Falling back to the root there
    would point ``test_public_face.py`` at that file and have it assert the shape of the wrong
    document, reporting failures that mean nothing.
    """
    if (REPO_ROOT / 'public' / 'mkdocs.yml').is_file():
        return REPO_ROOT / 'public'
    if (REPO_ROOT / 'mkdocs.yml').is_file():
        return REPO_ROOT
    return None
