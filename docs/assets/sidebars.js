/**
 * A small menu button at the top of each of the two columns: one folds the navigation, one folds
 * the table of contents.
 *
 * Material ships neither. Its sidebar is a drawer below 76.25em and a fixed column above it, and
 * there is no control for the column -- the community answer is a plugin, which for this project
 * would be a third-party dependency inside the docs build of a published package. This is the same
 * thing in forty lines.
 *
 * The button lives *in* the column it folds, at that column's inner edge, because that is where a
 * reader looks for it: a control for a thing belongs on the thing. Folding does not make the column
 * disappear -- it narrows it to a rail (see the stylesheet), which is what keeps the chevron where
 * the reader last saw it and leaves them something to press to bring the column back.
 *
 * The choice is remembered, because a reader who folds the navigation to read a page does not want
 * it back on the next one. That is `localStorage` and nothing else: no cookie, no request. The
 * game's own promise about keeping nothing is about the game, not about this page's furniture.
 */
(function () {
  'use strict';

  var STORE = 'ppsds-folded-columns';
  // A three-bar menu icon. The columns fold towards different edges, but the icon does not need
  // to point: it is a control for the column, not an arrow.
  var MENU = 'M4 6h16v2H4zM4 11h16v2H4zM4 16h16v2H4z';
  var COLUMNS = {
    nav: { sidebar: '.md-sidebar--primary', body: 'ppsds-nav-folded', what: 'navigation column' },
    toc: { sidebar: '.md-sidebar--secondary', body: 'ppsds-toc-folded', what: 'table of contents' },
  };

  /** Private mode, a full quota, or another site's data: none of them are worth breaking over. */
  function read() {
    var state = { nav: false, toc: false };
    try {
      var saved = JSON.parse(window.localStorage.getItem(STORE) || '{}');
      Object.keys(state).forEach(function (key) { state[key] = saved[key] === true; });
    } catch (error) {
      return { nav: false, toc: false };
    }
    return state;
  }

  function write(state) {
    try {
      window.localStorage.setItem(STORE, JSON.stringify(state));
    } catch (error) {
      // The toggle still works for this page; it just will not be remembered.
    }
  }

  var state = read();

  function apply() {
    Object.keys(COLUMNS).forEach(function (key) {
      var column = COLUMNS[key];
      document.body.classList.toggle(column.body, state[key]);
      var button = document.querySelector('.ppsds-fold--' + key);
      if (!button) return;
      var folded = state[key];
      var label = (folded ? 'Show' : 'Hide') + ' the ' + column.what;
      button.setAttribute('aria-pressed', folded ? 'true' : 'false');
      button.setAttribute('aria-label', label);
      button.title = label;
    });
  }

  // A chevron on every table-of-contents entry that has children. Material's TOC is nested in
  // the DOM but has no control for collapsing a branch; this adds one without touching the
  // theme. The collapsed state is per page, not remembered: it follows the section you are
  // reading, not your habits.
  var CHEVRON_DOWN = 'M7.41 8.59 12 13.17l4.59-4.58L18 10l-6 6-6-6z';

  function buildTocToggles() {
    var items = document.querySelectorAll('.md-nav--secondary li.md-nav__item');
    Array.prototype.forEach.call(items, function (item) {
      var children = item.querySelector(':scope > ul, :scope > nav');
      if (!children) return;
      var button = document.createElement('button');
      button.type = 'button';
      button.className = 'ppsds-toc-toggle';
      button.setAttribute('aria-expanded', 'true');
      button.setAttribute('aria-label', 'Collapse this section');
      button.innerHTML = '<svg viewBox="0 0 24 24"><path d="' + CHEVRON_DOWN + '"/></svg>';
      button.addEventListener('click', function (event) {
        event.preventDefault();
        event.stopPropagation();
        var closed = item.classList.toggle('ppsds-toc-closed');
        button.setAttribute('aria-expanded', closed ? 'false' : 'true');
        button.setAttribute('aria-label', (closed ? 'Expand' : 'Collapse') + ' this section');
      });
      item.appendChild(button);
    });
  }

  function build() {
    buildTocToggles();
    Object.keys(COLUMNS).forEach(function (key) {
      var sidebar = document.querySelector(COLUMNS[key].sidebar);
      if (!sidebar) return;
      var button = document.createElement('button');
      button.type = 'button';
      button.className = 'ppsds-fold ppsds-fold--' + key;
      button.innerHTML = '<svg viewBox="0 0 24 24"><path d="' + MENU + '"/></svg>';
      button.addEventListener('click', function () {
        state[key] = !state[key];
        write(state);
        apply();
      });
      // After the scrollwrap, so that folding the column (which hides the scrollwrap) cannot take
      // the button with it.
      sidebar.appendChild(button);
    });
    apply();
  }

  apply(); // the classes first, so a remembered page does not paint the columns and then drop them
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', build);
  else build();
}());
