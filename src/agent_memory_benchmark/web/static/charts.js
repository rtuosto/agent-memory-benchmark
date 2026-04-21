// Chart.js helpers for the run-detail page.
//
// Data is embedded as JSON <script id="chart-data"> so we avoid a second
// round-trip and keep the page snapshot-testable. The template omits the
// script when there is nothing to chart; each init function bails out
// gracefully if its canvas or data is missing.

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

    // Dark-palette defaults — match dashboard.css variables.
    Chart.defaults.color = "#9ca3af";
    Chart.defaults.borderColor = "#2a2f37";
    Chart.defaults.font.family =
        "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif";

    const palette = {
        primary: "rgba(96, 165, 250, 0.75)",
        primaryLine: "#60a5fa",
        good: "rgba(52, 211, 153, 0.75)",
        goodLine: "#34d399",
        warn: "rgba(251, 146, 60, 0.75)",
        warnLine: "#fb923c",
        purple: "rgba(192, 132, 252, 0.75)",
        purpleLine: "#c084fc",
        grid: "rgba(156, 163, 175, 0.15)",
    };

    const commonScaleOpts = {
        grid: { color: palette.grid },
        ticks: { color: "#9ca3af" },
    };

    initPerCategoryChart(data.per_category, palette, commonScaleOpts);
    initLatencyChart(data.latency, palette, commonScaleOpts);
    initFootprintChart(data.footprint, palette, commonScaleOpts);
    initEvidenceChart(data.evidence, palette, commonScaleOpts);

    function initPerCategoryChart(buckets, palette, scaleOpts) {
        const ctx = document.getElementById("chart-per-category");
        if (!ctx || !buckets || !buckets.length) return;
        new Chart(ctx, {
            type: "bar",
            data: {
                labels: buckets.map((b) => b.name),
                datasets: [
                    {
                        label: "Accuracy",
                        data: buckets.map((b) =>
                            b.accuracy == null ? null : b.accuracy * 100
                        ),
                        backgroundColor: palette.primary,
                        borderColor: palette.primaryLine,
                        borderWidth: 1,
                    },
                    {
                        label: "Token-F1",
                        data: buckets.map((b) =>
                            b.token_f1 == null ? null : b.token_f1 * 100
                        ),
                        backgroundColor: palette.good,
                        borderColor: palette.goodLine,
                        borderWidth: 1,
                    },
                ],
            },
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
                    legend: { labels: { color: "#e5e7eb" } },
                    tooltip: {
                        callbacks: {
                            label: (ctx) =>
                                `${ctx.dataset.label}: ${ctx.parsed.x.toFixed(2)}%`,
                        },
                    },
                },
            },
        });
    }

    function initLatencyChart(buckets, palette, scaleOpts) {
        const ctx = document.getElementById("chart-latency");
        if (!ctx || !buckets || !buckets.length) return;
        new Chart(ctx, {
            type: "bar",
            data: {
                labels: buckets.map((b) => b.name),
                datasets: [
                    makeDataset("mean", buckets, "mean", palette.primary, palette.primaryLine),
                    makeDataset("p50", buckets, "p50", palette.good, palette.goodLine),
                    makeDataset("p95", buckets, "p95", palette.warn, palette.warnLine),
                    makeDataset("max", buckets, "max", palette.purple, palette.purpleLine),
                ],
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: scaleOpts,
                    y: {
                        ...scaleOpts,
                        type: "logarithmic",
                        ticks: {
                            ...scaleOpts.ticks,
                            callback: (v) => v + " ms",
                        },
                    },
                },
                plugins: {
                    legend: { labels: { color: "#e5e7eb" } },
                    tooltip: {
                        callbacks: {
                            label: (ctx) =>
                                `${ctx.dataset.label}: ${ctx.parsed.y.toFixed(1)} ms`,
                        },
                    },
                },
            },
        });
    }

    function initFootprintChart(buckets, palette, scaleOpts) {
        const ctx = document.getElementById("chart-footprint");
        if (!ctx || !buckets || !buckets.length) return;
        new Chart(ctx, {
            type: "bar",
            data: {
                labels: buckets.map((b) => b.name),
                datasets: [
                    makeDataset("mean", buckets, "mean", palette.primary, palette.primaryLine),
                    makeDataset("p50", buckets, "p50", palette.good, palette.goodLine),
                    makeDataset("p95", buckets, "p95", palette.warn, palette.warnLine),
                    makeDataset("max", buckets, "max", palette.purple, palette.purpleLine),
                ],
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: { x: scaleOpts, y: scaleOpts },
                plugins: { legend: { labels: { color: "#e5e7eb" } } },
            },
        });
    }

    function initEvidenceChart(buckets, palette, scaleOpts) {
        const ctx = document.getElementById("chart-evidence");
        if (!ctx || !buckets || !buckets.length) return;
        new Chart(ctx, {
            type: "bar",
            data: {
                labels: buckets.map((b) => b.name),
                datasets: [
                    {
                        label: "Completeness",
                        data: buckets.map((b) =>
                            b.completeness == null ? null : b.completeness * 100
                        ),
                        backgroundColor: palette.primary,
                        borderColor: palette.primaryLine,
                        borderWidth: 1,
                    },
                    {
                        label: "Density",
                        data: buckets.map((b) =>
                            b.density == null ? null : b.density * 100
                        ),
                        backgroundColor: palette.good,
                        borderColor: palette.goodLine,
                        borderWidth: 1,
                    },
                ],
            },
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
                    legend: { labels: { color: "#e5e7eb" } },
                    tooltip: {
                        callbacks: {
                            label: (ctx) =>
                                `${ctx.dataset.label}: ${ctx.parsed.y.toFixed(2)}%`,
                        },
                    },
                },
            },
        });
    }

    function makeDataset(label, buckets, key, bg, border) {
        return {
            label,
            data: buckets.map((b) => (b[key] == null ? null : b[key])),
            backgroundColor: bg,
            borderColor: border,
            borderWidth: 1,
        };
    }
})();
