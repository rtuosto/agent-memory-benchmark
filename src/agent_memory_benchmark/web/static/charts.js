// Chart.js helpers for the run-detail page.
//
// Warm palette = current run. Cool palette = baseline (when compared).
// When a baseline is active, latency and footprint charts collapse to
// mean-only so eight bars per group don't fight for space — the precise
// p50/p95/max numbers stay one click away in the compare table below.

(function () {
    const dataEl = document.getElementById("chart-data");
    if (!dataEl) return;

    let data;
    try {
        data = JSON.parse(dataEl.textContent);
    } catch (e) {
        console.error("chart-data parse failed", e);
        return;
    }

    Chart.defaults.color = "#9ca3af";
    Chart.defaults.borderColor = "#2a2f37";
    Chart.defaults.font.family =
        "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif";

    // Warm palette = this run. Cool palette = baseline.
    const warm = {
        primary: "rgba(251, 146, 60, 0.8)",       // orange-400
        primaryLine: "#fb923c",
        secondary: "rgba(251, 191, 36, 0.8)",     // amber-400
        secondaryLine: "#fbbf24",
        tertiary: "rgba(248, 113, 113, 0.8)",     // red-400
        tertiaryLine: "#f87171",
        quaternary: "rgba(244, 114, 182, 0.8)",   // pink-400
        quaternaryLine: "#f472b6",
    };
    const cool = {
        primary: "rgba(96, 165, 250, 0.55)",      // blue-400
        primaryLine: "#60a5fa",
        secondary: "rgba(45, 212, 191, 0.55)",    // teal-400
        secondaryLine: "#2dd4bf",
        tertiary: "rgba(129, 140, 248, 0.55)",    // indigo-400
        tertiaryLine: "#818cf8",
        quaternary: "rgba(167, 139, 250, 0.55)",  // violet-400
        quaternaryLine: "#a78bfa",
    };

    const gridColor = "rgba(156, 163, 175, 0.15)";
    const tickColor = "#9ca3af";
    const legendColor = "#e5e7eb";
    const scaleOpts = { grid: { color: gridColor }, ticks: { color: tickColor } };

    const hasBaseline = data.has_baseline === true;

    initPerCategoryChart(data.per_category, hasBaseline);
    initLatencyChart(data.latency, hasBaseline);
    initFootprintChart(data.footprint, hasBaseline);
    initEvidenceChart(data.evidence, hasBaseline);

    function initPerCategoryChart(buckets, hasBaseline) {
        const ctx = document.getElementById("chart-per-category");
        if (!ctx || !buckets || !buckets.length) return;
        const datasets = [
            ds("Accuracy (this run)", buckets, (b) => pct(b.accuracy), warm.primary, warm.primaryLine),
            ds("Token-F1 (this run)", buckets, (b) => pct(b.token_f1), warm.secondary, warm.secondaryLine),
        ];
        if (hasBaseline) {
            datasets.push(
                ds("Accuracy (baseline)", buckets, (b) => pct(b.baseline_accuracy), cool.primary, cool.primaryLine),
                ds("Token-F1 (baseline)", buckets, (b) => pct(b.baseline_token_f1), cool.secondary, cool.secondaryLine),
            );
        }
        new Chart(ctx, {
            type: "bar",
            data: { labels: buckets.map((b) => b.name), datasets },
            options: {
                indexAxis: "y",
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        ...scaleOpts,
                        min: 0,
                        max: 100,
                        ticks: { ...scaleOpts.ticks, callback: (v) => v + "%" },
                    },
                    y: scaleOpts,
                },
                plugins: {
                    legend: { labels: { color: legendColor } },
                    tooltip: {
                        callbacks: {
                            label: (ctx) =>
                                `${ctx.dataset.label}: ${Number(ctx.parsed.x).toFixed(2)}%`,
                        },
                    },
                },
            },
        });
    }

    function initLatencyChart(buckets, hasBaseline) {
        const ctx = document.getElementById("chart-latency");
        if (!ctx || !buckets || !buckets.length) return;

        const datasets = hasBaseline
            ? [
                  ds("mean (this run)", buckets, (b) => b.mean, warm.primary, warm.primaryLine),
                  ds("mean (baseline)", buckets, (b) => b.baseline_mean, cool.primary, cool.primaryLine),
              ]
            : [
                  ds("mean", buckets, (b) => b.mean, warm.primary, warm.primaryLine),
                  ds("p50", buckets, (b) => b.p50, warm.secondary, warm.secondaryLine),
                  ds("p95", buckets, (b) => b.p95, warm.tertiary, warm.tertiaryLine),
                  ds("max", buckets, (b) => b.max, warm.quaternary, warm.quaternaryLine),
              ];
        new Chart(ctx, {
            type: "bar",
            data: { labels: buckets.map((b) => b.name), datasets },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: scaleOpts,
                    y: {
                        ...scaleOpts,
                        type: "logarithmic",
                        ticks: { ...scaleOpts.ticks, callback: (v) => v + " ms" },
                    },
                },
                plugins: {
                    legend: { labels: { color: legendColor } },
                    tooltip: {
                        callbacks: {
                            label: (ctx) =>
                                `${ctx.dataset.label}: ${Number(ctx.parsed.y).toFixed(1)} ms`,
                        },
                    },
                },
            },
        });
    }

    function initFootprintChart(buckets, hasBaseline) {
        const ctx = document.getElementById("chart-footprint");
        if (!ctx || !buckets || !buckets.length) return;

        const datasets = hasBaseline
            ? [
                  ds("mean (this run)", buckets, (b) => b.mean, warm.primary, warm.primaryLine),
                  ds("mean (baseline)", buckets, (b) => b.baseline_mean, cool.primary, cool.primaryLine),
              ]
            : [
                  ds("mean", buckets, (b) => b.mean, warm.primary, warm.primaryLine),
                  ds("p50", buckets, (b) => b.p50, warm.secondary, warm.secondaryLine),
                  ds("p95", buckets, (b) => b.p95, warm.tertiary, warm.tertiaryLine),
                  ds("max", buckets, (b) => b.max, warm.quaternary, warm.quaternaryLine),
              ];
        new Chart(ctx, {
            type: "bar",
            data: { labels: buckets.map((b) => b.name), datasets },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: { x: scaleOpts, y: scaleOpts },
                plugins: { legend: { labels: { color: legendColor } } },
            },
        });
    }

    function initEvidenceChart(buckets, hasBaseline) {
        const ctx = document.getElementById("chart-evidence");
        if (!ctx || !buckets || !buckets.length) return;
        const datasets = [
            ds("Completeness (this run)", buckets, (b) => pct(b.completeness), warm.primary, warm.primaryLine),
            ds("Density (this run)", buckets, (b) => pct(b.density), warm.secondary, warm.secondaryLine),
        ];
        if (hasBaseline) {
            datasets.push(
                ds("Completeness (baseline)", buckets, (b) => pct(b.baseline_completeness), cool.primary, cool.primaryLine),
                ds("Density (baseline)", buckets, (b) => pct(b.baseline_density), cool.secondary, cool.secondaryLine),
            );
        }
        new Chart(ctx, {
            type: "bar",
            data: { labels: buckets.map((b) => b.name), datasets },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: scaleOpts,
                    y: {
                        ...scaleOpts,
                        min: 0,
                        max: 100,
                        ticks: { ...scaleOpts.ticks, callback: (v) => v + "%" },
                    },
                },
                plugins: {
                    legend: { labels: { color: legendColor } },
                    tooltip: {
                        callbacks: {
                            label: (ctx) =>
                                `${ctx.dataset.label}: ${Number(ctx.parsed.y).toFixed(2)}%`,
                        },
                    },
                },
            },
        });
    }

    function ds(label, buckets, picker, bg, border) {
        return {
            label,
            data: buckets.map((b) => {
                const v = picker(b);
                return v == null ? null : v;
            }),
            backgroundColor: bg,
            borderColor: border,
            borderWidth: 1,
        };
    }

    function pct(v) {
        return v == null ? null : v * 100;
    }
})();
