// Client-side table sorting for the runs + jobs list pages.
//
// Opt-in: any <table class="sortable"> with <th data-sort="..."> headers
// gets click-to-sort. Supported sort types (data-sort on the <th>):
//   "text" (default) — lexical, case-insensitive
//   "num"            — numeric; strips %, commas, and non-numeric suffixes
//   "time"           — reads datetime attr from a nested <time datetime="..."> cell
//
// Design notes:
//   • No dependencies. Runs on DOMContentLoaded and rewires if rows
//     are replaced (e.g. by htmx later) via a MutationObserver.
//   • Stable sort (Array.prototype.sort is stable in modern browsers).
//   • Empty / missing values always sort last, regardless of direction,
//     so a column with some blank cells doesn't push the meaningful
//     rows out of view.

(function () {
    var INDICATOR = { asc: ' ▲', desc: ' ▼' };

    function cellValue(row, index, type) {
        var td = row.cells[index];
        if (!td) return { missing: true };
        var timeEl = td.querySelector('time[datetime]');
        if (timeEl && (type === 'time' || !type)) {
            var iso = timeEl.getAttribute('datetime');
            if (!iso) return { missing: true };
            var t = Date.parse(iso);
            return isNaN(t) ? { missing: true } : { num: t };
        }
        var raw = (td.textContent || '').trim();
        if (!raw || raw === '—' || raw === '-') return { missing: true };
        if (type === 'num') {
            var cleaned = raw.replace(/[,%$\s]/g, '');
            var n = parseFloat(cleaned);
            return isNaN(n) ? { missing: true } : { num: n };
        }
        return { text: raw.toLowerCase() };
    }

    function compare(a, b) {
        if (a.missing && b.missing) return 0;
        if (a.missing) return 1;   // missing → end
        if (b.missing) return -1;
        if ('num' in a && 'num' in b) return a.num - b.num;
        if (a.text < b.text) return -1;
        if (a.text > b.text) return 1;
        return 0;
    }

    function sortTable(table, colIndex, type, direction) {
        var tbody = table.tBodies[0];
        if (!tbody) return;
        var rows = Array.prototype.slice.call(tbody.rows);
        var indexed = rows.map(function (row, i) {
            return { row: row, key: cellValue(row, colIndex, type), i: i };
        });
        indexed.sort(function (a, b) {
            // Missing values always trail, independent of direction.
            if (a.key.missing !== b.key.missing) {
                return a.key.missing ? 1 : -1;
            }
            var cmp = compare(a.key, b.key);
            if (cmp !== 0) return direction === 'desc' ? -cmp : cmp;
            return a.i - b.i; // stable fallback
        });
        indexed.forEach(function (entry) { tbody.appendChild(entry.row); });
    }

    function clearIndicators(table) {
        table.querySelectorAll('thead th').forEach(function (th) {
            if (th.dataset.origLabel) th.textContent = th.dataset.origLabel;
            th.removeAttribute('aria-sort');
        });
    }

    function wire(table) {
        if (table._sortableWired) return;
        table._sortableWired = true;
        var ths = table.querySelectorAll('thead th');
        ths.forEach(function (th, idx) {
            if (th.dataset.sort === 'off') return;
            th.dataset.origLabel = th.textContent;
            th.style.cursor = 'pointer';
            th.setAttribute('role', 'button');
            th.setAttribute('tabindex', '0');
            var trigger = function () {
                var currentDir = th.getAttribute('aria-sort');
                var nextDir = currentDir === 'ascending' ? 'desc' : 'asc';
                clearIndicators(table);
                sortTable(table, idx, th.dataset.sort || 'text', nextDir);
                th.setAttribute('aria-sort', nextDir === 'asc' ? 'ascending' : 'descending');
                th.textContent = th.dataset.origLabel + INDICATOR[nextDir];
            };
            th.addEventListener('click', trigger);
            th.addEventListener('keydown', function (e) {
                if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault();
                    trigger();
                }
            });
        });
    }

    function init() {
        document.querySelectorAll('table.sortable').forEach(wire);
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
