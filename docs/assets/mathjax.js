/**
 * MathJax bootstrap for the docs.
 *
 * `pymdownx.arithmatex` runs in `generic` mode: it leaves the delimiters in place and wraps the
 * expression in a span with class `arithmatex`. MathJax is told to process exactly those spans
 * (and only those), so `$` or `\(` in a code block or in prose cannot turn into an equation by
 * accident. The loader is `extra_javascript` in `mkdocs.yml`.
 */
window.MathJax = {
  tex: {
    inlineMath: [['\\(', '\\)']],
    displayMath: [['\\[', '\\]']],
    processEscapes: true,
  },
  options: {
    ignoreHtmlClass: '.*|',
    processHtmlClass: 'arithmatex',
  },
};
