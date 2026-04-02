// --- GLOBAL STATE ---
let dataset = [];
let metadata = {};
let colorMode = 'original';
let uniqueClasses = [];
let pointById = new Map();
let currentPointSize = 5;
let selectedPoint = null;
let kmeansProjectionSource = 'pca'; // 'pca' or 'mds'

// Filters
let minPrecision = 0;
let minRecall = 0;

// Scales Storage per Plot
const scalesMap = { pca: {}, mds: {}, kmeans: {}, pca2d: {}, mds2d: {} };

// D3 Brushes
const brushPCA = d3.brush();
const brushMDS = d3.brush();
const brushKMeans = d3.brush();
const brushPCA2D = d3.brush();
const brushMDS2D = d3.brush();

// --- SCALES ---
const customTableau = [...d3.schemeTableau10];
customTableau[2] = '#2ca02c'; 
const colorOriginal = d3.scaleOrdinal(customTableau);

const bluesDiscrete = ["#08519c", "#3182bd", "#6baed6", "#9ecae1", "#c6dbef"];
const colorPrecision = d3.scaleQuantize().domain([0, 1]).range(bluesDiscrete); 

const redsDiscrete = ["#a50f15", "#de2d26", "#fb6a4a", "#fc9272", "#fcbba1"];
const colorRecall = d3.scaleQuantize().domain([0, 1]).range(redsDiscrete);

const rdYlGnDiscrete = ["#d73027", "#fdae61", "#ffffbf", "#a6d96a", "#1a9641"];
const colorFScore = d3.scaleQuantize().domain([0, 1]).range(rdYlGnDiscrete);

const colorKMeans = d3.scaleOrdinal(d3.schemeCategory10);

// Gauges
const gauges = { precision: { foreground: null }, recall: { foreground: null }, fscore: { foreground: null } };
const gaugeAngleScale = d3.scaleLinear().domain([0, 1]).range([-Math.PI / 2, Math.PI / 2]);

// --- INITIALIZATION ---
Promise.all([
    d3.json("../json/step2_final_data.json?v=" + Date.now()),
    d3.csv("../../dataset/wine.csv"),
    d3.json("../json/kmeans_results.json?v=" + Date.now()),
    d3.json("../json/kmeans_2d_results.json?v=" + Date.now())
]).then(([data, wineData, kmeansData, kmeans2dData]) => {
    
    const kmeansMap = new Map();
    if(kmeansData && kmeansData.points) {
        kmeansData.points.forEach(p => kmeansMap.set(p.id, p));
    }

    const kmeans2dMap = new Map();
    if(kmeans2dData && kmeans2dData.points) {
        kmeans2dData.points.forEach(p => kmeans2dMap.set(p.id, p));
    }

    data.points.forEach((p, i) => {
        if (wineData[i]) p.attributes = wineData[i];
        const kData = kmeansMap.get(p.id);
        if(kData) {
            p.kmeans_cluster = kData.kmeans_cluster;
            p.is_anomaly = kData.is_anomaly; 
        }
        const k2dData = kmeans2dMap.get(p.id);
        if(k2dData) {
            p.pca_kmeans_cluster = k2dData.pca_kmeans_cluster;
            p.pca_is_anomaly = k2dData.pca_is_anomaly;
            p.mds_kmeans_cluster = k2dData.mds_kmeans_cluster;
            p.mds_is_anomaly = k2dData.mds_is_anomaly;
        }
    });

    dataset = data.points;
    metadata = data.metadata;
    uniqueClasses = Array.from(new Set(dataset.map(d => d.label))).sort();
    pointById = new Map(dataset.map(p => [p.id, p]));

    // Footer Stats
    if(metadata) {
        d3.select("#fb-dataset").text(metadata.dataset || "-");
        if (metadata.global_assessment && metadata.global_assessment.pca && metadata.global_assessment.mds) {
            d3.select("#fb-pca-trust").text((metadata.global_assessment.pca.trustworthiness * 100).toFixed(1) + "%");
            d3.select("#fb-pca-cont").text((metadata.global_assessment.pca.continuity * 100).toFixed(1) + "%");
            d3.select("#fb-mds-trust").text((metadata.global_assessment.mds.trustworthiness * 100).toFixed(1) + "%");
            d3.select("#fb-mds-cont").text((metadata.global_assessment.mds.continuity * 100).toFixed(1) + "%");
        }
        let gFScore = metadata.global_f_score;
        if (gFScore === undefined && metadata.global_assessment) gFScore = metadata.global_assessment.global_f_score || metadata.global_assessment.f_score;
        d3.select("#fb-global-fscore").text(gFScore !== undefined ? (gFScore * 100).toFixed(1) + "%" : "N/A");
    }

    drawPlot("#pca-plot", "pca_x", "pca_y", "pca", brushPCA);
    drawPlot("#mds-plot", "mds_x", "mds_y", "mds", brushMDS);
    drawPlot("#kmeans-plot", kmeansProjectionSource === 'pca' ? 'pca_x' : 'mds_x', kmeansProjectionSource === 'pca' ? 'pca_y' : 'mds_y', "kmeans", brushKMeans, d => {
        if (colorMode === 'original') return d.is_anomaly ? '#e74c3c' : colorOriginal(d.label);
        return getColor(d);
    });
    
    // Temporaneamente mostriamo la tab 2D per permettere a drawPlot di misurare correttamente il DOM
    d3.select("#app-grid").classed("mode-2d", true);
    d3.selectAll(".tab-2d").classed("hidden-panel", false);
    d3.selectAll(".tab-13d").classed("hidden-panel", true);
    drawPlot("#pca-plot-2d", "pca_x", "pca_y", "pca2d", brushPCA2D, d => {
        if (colorMode === 'original') return d.pca_is_anomaly ? '#e74c3c' : colorOriginal(d.label);
        return getColor(d);
    });
    drawPlot("#mds-plot-2d", "mds_x", "mds_y", "mds2d", brushMDS2D, d => {
        if (colorMode === 'original') return d.mds_is_anomaly ? '#e74c3c' : colorOriginal(d.label);
        return getColor(d);
    });

    // Pulisce la griglia 2D dai placeholder e aggiunge il grafico di confronto (se non esiste)
    const grid2d = d3.select("#plots-grid-2d");

    // Identifica i contenitori principali da non rimuovere
    const pcaPlotContainer = d3.select("#pca-plot-2d").node()?.closest('.plot-container');
    const mdsPlotContainer = d3.select("#mds-plot-2d").node()?.closest('.plot-container');

    // Rimuove tutti i .plot-container che non sono i due principali o il comparison plot
    grid2d.selectAll(".plot-container").filter(function() {
        const isComparisonPlot = d3.select(this).attr("id") === "comparison-plot-container";
        return this !== pcaPlotContainer && this !== mdsPlotContainer && !isComparisonPlot;
    }).remove();

    // Aggiunge il contenitore per il grafico di confronto se non esiste già
    // Questo assicura che il comparison plot sia sempre presente e occupi la seconda riga
    // dopo che eventuali placeholder sono stati rimossi.
    // Lo aggiungiamo qui per assicurarci che sia sempre l'ultimo elemento nella griglia,
    // e quindi occupi la seconda riga correttamente grazie a grid-column: 1 / span 2.
    
    if (grid2d.select("#comparison-plot-container").empty()) {
        const newPlot = grid2d.append("div")
            .attr("id", "comparison-plot-container")
            .attr("class", "plot-container")
            .style("grid-column", "1 / span 2"); // Occupa l'intera seconda riga

        newPlot.append("div")
            .attr("class", "plot-title")
            .text("Comparison Plot"); // Titolo del nuovo grafico
        newPlot.append("div").attr("id", "comparison-plot").attr("class", "svg-container"); // Contenitore per il grafico
    }

    // Ripristiniamo la visualizzazione corretta (13D)
    d3.select("#app-grid").classed("mode-2d", false);
    d3.selectAll(".tab-2d").classed("hidden-panel", true);
    d3.selectAll(".tab-13d").classed("hidden-panel", false);

    drawParallelCoordinates("#pc-plot");
    
    setupBrushing();
    
    initGauge("#gauge-precision", gauges.precision);
    initGauge("#gauge-recall", gauges.recall);
    initGauge("#gauge-fscore", gauges.fscore);

    enhanceColorModeSwitcher();

    d3.selectAll("input[name='colorMode']").on("change", function() {
        colorMode = this.value;
        updateColors();
        updateLegend();
    });

    // Slider Size
    d3.select("#point-size-slider").on("input change", function() {
        currentPointSize = +this.value;
        d3.selectAll(".dot").attr("r", currentPointSize);
    });

    // Filters
    d3.select("#filter-prec").on("input change", function() {
        minPrecision = +this.value;
        d3.select("#val-min-prec").text(minPrecision.toFixed(2));
        applyFilters();
    });

    d3.select("#filter-rec").on("input change", function() {
        minRecall = +this.value;
        d3.select("#val-min-rec").text(minRecall.toFixed(2));
        applyFilters();
    });

    d3.select("#show-discrepancies").on("change", function() {
        toggleDiscrepancies(this.checked);
    });

    // K-Means source switcher
    d3.selectAll("input[name='kmeansSource']").on("change", function() {
        kmeansProjectionSource = this.value;
        redrawKMeansPlot();
    });

    // --- TAB SWITCHER LOGIC ---
    d3.selectAll("input[name='mainTab']").on("change", function() {
        const selectedTab = this.value;
        if (selectedTab === '13d') {
            d3.select("#app-grid").classed("mode-2d", false);
            d3.selectAll(".tab-13d").classed("hidden-panel", false);
            d3.selectAll(".tab-2d").classed("hidden-panel", true);
        } else {
            d3.select("#app-grid").classed("mode-2d", true);
            d3.selectAll(".tab-13d").classed("hidden-panel", true);
            d3.selectAll(".tab-2d").classed("hidden-panel", false);
        }
    });

    updateLegend();
}).catch(err => console.error("Error loading JSON/CSV:", err));


// --- KMEANS REDRAW ---
function redrawKMeansPlot() {
    const xKey = kmeansProjectionSource === 'pca' ? 'pca_x' : 'mds_x';
    const yKey = kmeansProjectionSource === 'pca' ? 'pca_y' : 'mds_y';

    // Clear the old plot
    d3.select("#kmeans-plot").html("");

    // Redraw with new coordinates
    drawPlot("#kmeans-plot", xKey, yKey, "kmeans", brushKMeans, d => {
        if (colorMode === 'original') return d.is_anomaly ? '#e74c3c' : colorOriginal(d.label);
        return getColor(d);
    });

    // Re-apply filters and discrepancies if they are active
    applyFilters();
    if (d3.select("#show-discrepancies").property("checked")) {
        toggleDiscrepancies(true);
    }

    // Se c'era un punto selezionato, ripristina la selezione sul nuovo grafico K-Means
    if (selectedPoint) {
        updateSelection(selectedPoint);
    }
}


// --- PLOTTING FUNCTION ---
function drawPlot(containerSelector, xKey, yKey, plotId, brushObj, customColorFn = null) {
    const container = d3.select(containerSelector); 
    const width = container.node().clientWidth;
    const height = container.node().clientHeight;
    const margin = { top: 20, right: 25, bottom: 35, left: 35 };

    const svgRoot = container.append("svg")
        .attr("viewBox", `0 0 ${width} ${height}`)
        .style("width", "100%")
        .style("height", "100%");

    svgRoot.on("mouseleave", resetAllHovers);

    svgRoot.on("click", () => {
        selectedPoint = null;
        d3.selectAll(".dot, .pc-line").style("opacity", 0.9);
        d3.selectAll(".link-group line").remove();
        resetAllHovers();
        updateLiveAnalytics([]); 
        if(d3.select("#show-discrepancies").property("checked")) toggleDiscrepancies(true);
    });

    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;

    const svg = svgRoot.append("g")
        .attr("transform", `translate(${margin.left},${margin.top})`);

    const xScale = d3.scaleLinear().domain(d3.extent(dataset, d => d[xKey])).nice().range([0, innerWidth]);
    const yScale = d3.scaleLinear().domain(d3.extent(dataset, d => d[yKey])).nice().range([innerHeight, 0]);

    scalesMap[plotId] = { xScale, yScale, xKey, yKey };

    svg.append("g").attr("transform", `translate(0,${innerHeight})`).call(d3.axisBottom(xScale).ticks(5));
    svg.append("g").call(d3.axisLeft(yScale).ticks(5));

    svg.append("g").attr("class", "centroid-layer");
    svg.append("g").attr("class", "link-group");
    const brushGroup = svg.append("g").attr("class", "brush-group");

    svg.selectAll(".dot")
        .data(dataset)
        .enter().append("circle")
        .attr("class", d => `dot dot-${plotId} pt-${d.id}`)
        .attr("cx", d => xScale(d[xKey]))
        .attr("cy", d => yScale(d[yKey]))
        .attr("r", currentPointSize)
        .attr("fill", d => customColorFn ? customColorFn(d) : getColor(d))
        .attr("stroke", "rgba(0,0,0,0.4)")
        .attr("stroke-width", 0.8)
        .style("opacity", 0.9)
        .style("cursor", "pointer")
        .on("mouseover", function(event, d) {
            resetAllHovers();
            d3.selectAll(`.dot.pt-${d.id}`)
              .attr("r", currentPointSize * 1.6)
              .attr("stroke", "rgba(0,0,0,0.9)")
              .attr("stroke-width", 2).raise();
            d3.selectAll(`.pc-line.pt-${d.id}`)
              .style("stroke-width", 3).raise();
            showTooltip(event, d);
        })
        .on("mouseout", function() { resetAllHovers(); })
        .on("click", function(event, d) {
            event.stopPropagation(); 
            updateSelection(d);
        });

    brushObj.extent([[0, 0], [innerWidth, innerHeight]]);
    brushGroup.call(brushObj);
    
    brushObj.xScale = xScale;
    brushObj.yScale = yScale;
}

// --- FILTERS LOGIC ---
function applyFilters() {
    d3.selectAll(".dot, .pc-line")
        .classed("filtered-out", d => d.precision < minPrecision || d.recall < minRecall)
        .style("display", d => (d.precision < minPrecision || d.recall < minRecall) ? "none" : null)
        .style("pointer-events", d => (d.precision < minPrecision || d.recall < minRecall) ? "none" : "auto");
        
    if(d3.select("#show-discrepancies").property("checked")) {
        toggleDiscrepancies(true);
    }
}

// --- VISUALIZING ANOMALIES VIA CENTROIDS ---
function toggleDiscrepancies(show) {
    const plotMapping = { pca: '#pca-plot', mds: '#mds-plot', kmeans: '#kmeans-plot', pca2d: '#pca-plot-2d', mds2d: '#mds-plot-2d' };
    if (show) {
        Object.keys(plotMapping).forEach(plotId => {
            const scales = scalesMap[plotId];
            const svg = d3.select(`${plotMapping[plotId]} svg g .centroid-layer`);
            svg.selectAll("*").remove(); 

            const isPCA2D = plotId === 'pca2d';
            const isMDS2D = plotId === 'mds2d';
            const clusterProp = isPCA2D ? 'pca_kmeans_cluster' : (isMDS2D ? 'mds_kmeans_cluster' : 'kmeans_cluster');
            const anomalyProp = isPCA2D ? 'pca_is_anomaly' : (isMDS2D ? 'mds_is_anomaly' : 'is_anomaly');

            const validKMeansClusters = Array.from(new Set(dataset.map(d => d[clusterProp]).filter(c => c !== undefined)));
            const centroids = {};
            validKMeansClusters.forEach(k => centroids[k] = {x:0, y:0, count:0});

            dataset.forEach(d => {
                if(d.precision >= minPrecision && d.recall >= minRecall && d[clusterProp] !== undefined) {
                    centroids[d[clusterProp]].x += scales.xScale(d[scales.xKey]);
                    centroids[d[clusterProp]].y += scales.yScale(d[scales.yKey]);
                    centroids[d[clusterProp]].count += 1;
                }
            });

            validKMeansClusters.forEach(k => {
                if(centroids[k].count > 0) {
                    centroids[k].x /= centroids[k].count;
                    centroids[k].y /= centroids[k].count;
                }
            });

            dataset.forEach(d => {
                if(d.precision < minPrecision || d.recall < minRecall || d[clusterProp] === undefined) return; 
                if(centroids[d[clusterProp]].count === 0) return;
                
                const cx = centroids[d[clusterProp]].x;
                const cy = centroids[d[clusterProp]].y;
                const px = scales.xScale(d[scales.xKey]);
                const py = scales.yScale(d[scales.yKey]);
                
                svg.append("line")
                    .attr("x1", px).attr("y1", py)
                    .attr("x2", cx).attr("y2", cy)
                    .attr("class", d[anomalyProp] ? "centroid-link centroid-anomaly" : "centroid-link centroid-correct");
            });

            validKMeansClusters.forEach(k => {
                if(centroids[k].count === 0) return;
                svg.append("path")
                    .attr("d", d3.symbol().type(d3.symbolCross).size(150)())
                    .attr("transform", `translate(${centroids[k].x}, ${centroids[k].y})`)
                    .attr("fill", colorKMeans(k))
                    .attr("stroke", "black")
                    .attr("stroke-width", 1.5)
                    .append("title").text(`K-Means Centroid ${k}`);
            });
        });

        d3.selectAll(".dot.dot-pca:not(.filtered-out), .dot.dot-mds:not(.filtered-out), .dot.dot-kmeans:not(.filtered-out)")
            .style("opacity", d => d.is_anomaly ? 1.0 : 0.2).attr("stroke-width", d => d.is_anomaly ? 1.5 : 0);
        d3.selectAll(".dot.dot-pca2d:not(.filtered-out)")
            .style("opacity", d => d.pca_is_anomaly ? 1.0 : 0.2).attr("stroke-width", d => d.pca_is_anomaly ? 1.5 : 0);
        d3.selectAll(".dot.dot-mds2d:not(.filtered-out)")
            .style("opacity", d => d.mds_is_anomaly ? 1.0 : 0.2).attr("stroke-width", d => d.mds_is_anomaly ? 1.5 : 0);

        d3.selectAll(".pc-line:not(.filtered-out)")
            .style("opacity", d => d.is_anomaly ? 1.0 : 0.1)
            .style("stroke-width", d => d.is_anomaly ? 2.5 : 1);

        updateConfusionMatrix(dataset.filter(d => d.precision >= minPrecision && d.recall >= minRecall));
    } else {
        Object.values(plotMapping).forEach(selector => d3.select(`${selector} svg g .centroid-layer`).selectAll("*").remove());
        d3.selectAll(".dot:not(.filtered-out)").style("opacity", 0.9).attr("stroke-width", 0.8);
        d3.selectAll(".pc-line:not(.filtered-out)").style("opacity", 0.6).style("stroke-width", 1.5);
        d3.select("#confusion-matrix-container").classed("hidden-panel", true);
        d3.select("#empty-state-placeholder").classed("hidden-panel", false);
    }
}

function updateConfusionMatrix(activePoints) {
    d3.select("#dynamic-panel-title").text("Cluster Discrepancies");
    d3.select("#empty-state-placeholder").classed("hidden-panel", true);
    d3.select("#gauges-container").classed("hidden-panel", true);
    d3.select("#neighbor-graph-container").classed("hidden-panel", true);
    d3.select("#confusion-matrix-container").classed("hidden-panel", false);

    const classes = Array.from(new Set(dataset.map(d => String(d.label)))).sort();
    const matrix = {};
    classes.forEach(r => {
        matrix[r] = {};
        classes.forEach(c => matrix[r][c] = 0);
    });
    
    activePoints.forEach(p => {
        const trueL = String(p.label); 
        const predL = String(p.kmeans_cluster); 
        if (matrix[trueL] && matrix[trueL][predL] !== undefined) {
            matrix[trueL][predL]++;
        }
    });

    let html = "<table style='border-collapse: collapse; width: 100%; text-align: center; font-size: 0.85rem;'>";
    html += "<thead><tr><th style='border-bottom: 1px solid #ccc; font-weight:normal; text-align:left; padding-bottom:5px;'>Abstract ↓ \\ KMeans →</th>";
    
    classes.forEach(c => { html += `<th>Matched C${c}</th>`; });
    html += "</tr></thead><tbody>";
    
    classes.forEach(row => {
        html += `<tr><td style='font-weight:bold; border-right: 1px solid #eee; text-align:left; padding: 8px 0;'>Producer ${row}</td>`;
        classes.forEach(col => {
            const count = matrix[row][col];
            const isDiag = (row === col); 
            const style = isDiag 
                ? "background: #e6fffa; color: #2ca02c; font-weight: bold;" 
                : (count > 0 ? "background: #fff5f5; color: #c0392b; font-weight: bold;" : "color: #bdc3c7;");
            html += `<td style='padding: 8px; border-bottom: 1px solid #eee; ${style}'>${count}</td>`;
        });
        html += "</tr>";
    });
    html += "</tbody></table>";
    d3.select("#cm-table").html(html);
}


// --- CLICK LOGIC (Single Point) ---
function updateSelection(d) {
    selectedPoint = d;

    d3.select("#pca-plot .brush-group").call(brushPCA.move, null);
    d3.select("#mds-plot .brush-group").call(brushMDS.move, null);
    d3.select("#kmeans-plot .brush-group").call(brushKMeans.move, null);
    d3.select("#pca-plot-2d .brush-group").call(brushPCA2D.move, null);
    d3.select("#mds-plot-2d .brush-group").call(brushMDS2D.move, null);

    d3.select("#dynamic-panel-title").text("Neighbor Graph");
    d3.select("#empty-state-placeholder").classed("hidden-panel", true);
    d3.select("#gauges-container").classed("hidden-panel", true);
    d3.select("#confusion-matrix-container").classed("hidden-panel", true);
    d3.select("#neighbor-graph-container").classed("hidden-panel", false);

    const neighborIds = d.neighbors || [];
    const activeIds = new Set([d.id, ...neighborIds]);

    // Livello 1, 2 e 3 per scatterplot e parallel coordinates
    d3.selectAll(".dot:not(.filtered-out)")
        .style("opacity", p => p.id === d.id ? 1 : (activeIds.has(p.id) ? 0.8 : 0.1))
        .attr("r", p => p.id === d.id ? currentPointSize * 1.5 : currentPointSize)
        .attr("stroke-width", p => p.id === d.id ? 2 : 0.8);

    d3.selectAll(".pc-line:not(.filtered-out)")
        .style("opacity", p => p.id === d.id ? 1 : (activeIds.has(p.id) ? 0.6 : 0.05))
        .style("stroke-width", p => p.id === d.id ? 3 : 1.5);

    // Sollevamento SVG: prima i vicini, poi la selezione principale per assicurarsi che stia al top
    d3.selectAll(".pc-line:not(.filtered-out)").filter(p => activeIds.has(p.id) && p.id !== d.id).raise();
    d3.selectAll(".pc-line:not(.filtered-out)").filter(p => p.id === d.id).raise();
    d3.selectAll(".dot:not(.filtered-out)").filter(p => activeIds.has(p.id) && p.id !== d.id).raise();
    d3.selectAll(".dot:not(.filtered-out)").filter(p => p.id === d.id).raise();

    const kmeansXKey = scalesMap['kmeans'].xKey;
    const kmeansYKey = scalesMap['kmeans'].yKey;

    drawLines("pca", d, neighborIds, "pca_x", "pca_y", brushPCA);
    drawLines("mds", d, neighborIds, "mds_x", "mds_y", brushMDS);
    drawLines("kmeans", d, neighborIds, kmeansXKey, kmeansYKey, brushKMeans);
    drawLines("pca2d", d, neighborIds, "pca_x", "pca_y", brushPCA2D);
    drawLines("mds2d", d, neighborIds, "mds_x", "mds_y", brushMDS2D);

    const neighborsData = neighborIds.map(id => pointById.get(id)).filter(Boolean);
    drawNeighborGraph(d, neighborsData);
}

function drawLines(plotId, sourceD, neighborIds, xKey, yKey, scales) {
    const plotMapping = { pca: '#pca-plot', mds: '#mds-plot', kmeans: '#kmeans-plot', pca2d: '#pca-plot-2d', mds2d: '#mds-plot-2d' };
    const linkGroup = d3.select(`${plotMapping[plotId]} .link-group`);
    linkGroup.selectAll("line").remove(); 

    const sourceX = scalesMap[plotId].xScale(sourceD[xKey]);
    const sourceY = scalesMap[plotId].yScale(sourceD[yKey]);

    const linesData = neighborIds.map(id => pointById.get(id)).filter(p => p && p.precision >= minPrecision && p.recall >= minRecall);

    linkGroup.selectAll("line")
        .data(linesData)
        .enter().append("line")
        .attr("x1", sourceX)
        .attr("y1", sourceY)
        .attr("x2", target => scalesMap[plotId].xScale(target[xKey]))
        .attr("y2", target => scalesMap[plotId].yScale(target[yKey]))
        .attr("stroke", target => target.label === sourceD.label ? "#2ca02c" : "#e74c3c")
        .attr("stroke-width", 1.5)
        .attr("opacity", 0.6);
}

// --- BI-DIRECTIONAL BRUSHING ---
function setupBrushing() {
    const brushList = [
        { obj: brushPCA, sel: "#pca-plot", id: "pca" },
        { obj: brushMDS, sel: "#mds-plot", id: "mds" },
        { obj: brushKMeans, sel: "#kmeans-plot", id: "kmeans" },
        { obj: brushPCA2D, sel: "#pca-plot-2d", id: "pca2d" },
        { obj: brushMDS2D, sel: "#mds-plot-2d", id: "mds2d" }
    ];

    brushList.forEach(({obj: brush, id: plotId}) => {
        brush.on("start brush end", function (event) {
            if(event.sourceEvent && event.sourceEvent.type === "mousedown") {
                brushList.filter(b => b.obj !== brush).forEach(b => {
                    d3.select(`${b.sel} .brush-group`).call(b.obj.move, null);
                });
            }
            handleBrush(event, brush, scalesMap[plotId].xKey, scalesMap[plotId].yKey);
        });
    });
}

function handleBrush(event, brushObj, xKey, yKey) {
    if (!event.sourceEvent) return;
    d3.selectAll(".link-group line").remove(); 

    if (!event.selection) {
        if (event.type === "end") {
            selectedPoint = null;
            d3.selectAll(".dot:not(.filtered-out), .pc-line:not(.filtered-out)").style("opacity", 0.9); 
            resetAllHovers(); 
            updateLiveAnalytics([]); 
        }
        return;
    }

    selectedPoint = null;
    const [[x0, y0], [x1, y1]] = event.selection;
    let selectedPoints = [];
    dataset.forEach(d => {
        if(d.precision < minPrecision || d.recall < minRecall) return;
        const cx = brushObj.xScale(d[xKey]);
        const cy = brushObj.yScale(d[yKey]);
        const isSelected = x0 <= cx && cx <= x1 && y0 <= cy && cy <= y1;
        if (isSelected) selectedPoints.push(d);
        d3.selectAll(`.pt-${d.id}:not(.filtered-out)`).style("opacity", isSelected ? 0.9 : 0.15);
    });
    
    updateLiveAnalytics(selectedPoints);
}

// --- LOGIC TO UPDATE GAUGES AND DYNAMIC PANEL ---
function updateLiveAnalytics(selectedPoints) {
    
    if (selectedPoints.length === 0) {
        d3.select("#dynamic-panel-title").text("Live Analytics");
        d3.select("#gauges-container").classed("hidden-panel", true);
        d3.select("#neighbor-graph-container").classed("hidden-panel", true);
        
        if (d3.select("#show-discrepancies").property("checked")) {
            d3.select("#empty-state-placeholder").classed("hidden-panel", true);
            d3.select("#confusion-matrix-container").classed("hidden-panel", false);
        } else {
            d3.select("#confusion-matrix-container").classed("hidden-panel", true);
            d3.select("#empty-state-placeholder").classed("hidden-panel", false);
        }
        return;
    }
    
    d3.select("#dynamic-panel-title").text("Selection Metrics");
    d3.select("#empty-state-placeholder").classed("hidden-panel", true);
    d3.select("#neighbor-graph-container").classed("hidden-panel", true);
    d3.select("#confusion-matrix-container").classed("hidden-panel", true);
    d3.select("#gauges-container").classed("hidden-panel", false);

    d3.select("#stat-count").text(selectedPoints.length);
    
    const avgPrec = d3.mean(selectedPoints, d => d.precision);
    const avgRecall = d3.mean(selectedPoints, d => d.recall);
    const avgFScore = d3.mean(selectedPoints, d => d.f_score);

    updateGauge(gauges.precision, avgPrec, colorFScore(avgPrec), "#val-precision");
    updateGauge(gauges.recall, avgRecall, colorFScore(avgRecall), "#val-recall");
    updateGauge(gauges.fscore, avgFScore, colorFScore(avgFScore), "#val-fscore");
}

// --- HELPERS ---
function resetAllHovers() {
    if (selectedPoint) {
        d3.selectAll(".dot")
            .attr("r", p => p.id === selectedPoint.id ? currentPointSize * 1.5 : currentPointSize)
            .attr("stroke", p => p.id === selectedPoint.id ? "rgba(0,0,0,0.9)" : "rgba(0,0,0,0.4)")
            .attr("stroke-width", p => p.id === selectedPoint.id ? 2 : 0.8);
        d3.selectAll(".pc-line")
            .style("stroke-width", p => p.id === selectedPoint.id ? 3 : 1.5);
    } else {
        d3.selectAll(".dot")
            .attr("r", currentPointSize)
            .attr("stroke", "rgba(0,0,0,0.4)")
            .attr("stroke-width", 0.8);
        d3.selectAll(".pc-line")
            .style("stroke-width", 1.5);
    }
    hideTooltip();
}

function enhanceColorModeSwitcher() {
    const options = [
        { value: 'original', label: 'Original' },
        { value: 'precision', label: 'Precision' },
        { value: 'recall', label: 'Recall' },
        { value: 'fscore', label: 'F-Score' }
    ];

    const container = d3.select(".control-group");
    if (container.empty()) { return; }
    container.html(""); 

    container.classed("control-group", false).classed("segmented-control", true);

    options.forEach((opt, i) => {
        container.append("input").attr("type", "radio").attr("id", `cm-${opt.value}`).attr("name", "colorMode").attr("value", opt.value).property("checked", i === 0); 
        container.append("label").attr("for", `cm-${opt.value}`).text(opt.label);
    });
}

function getColor(d) {
    if (colorMode === 'original') return colorOriginal(d.label);
    if (colorMode === 'precision') return colorPrecision(d.precision);
    if (colorMode === 'recall') return colorRecall(d.recall);
    if (colorMode === 'fscore') return colorFScore(d.f_score);
}

function updateColors() {
    d3.selectAll(".dot.dot-pca").transition().duration(500).attr("fill", d => getColor(d));
    d3.selectAll(".dot.dot-mds").transition().duration(500).attr("fill", d => getColor(d));
    d3.selectAll(".dot.dot-pca2d").transition().duration(500).attr("fill", d => {
        if (colorMode === 'original') return d.pca_is_anomaly ? '#e74c3c' : colorOriginal(d.label);
        return getColor(d);
    });
    d3.selectAll(".dot.dot-mds2d").transition().duration(500).attr("fill", d => {
        if (colorMode === 'original') return d.mds_is_anomaly ? '#e74c3c' : colorOriginal(d.label);
        return getColor(d);
    });
    d3.selectAll(".dot.dot-kmeans").transition().duration(500).attr("fill", d => {
        if (colorMode === 'original') return d.is_anomaly ? '#e74c3c' : colorOriginal(d.label);
        return getColor(d);
    });
    d3.selectAll(".pc-line").transition().duration(500).style("stroke", d => {
        if (colorMode === 'original') return d.is_anomaly ? '#e74c3c' : colorOriginal(d.label);
        return getColor(d);
    });
}

// --- LEGENDA (CORREZIONE ALLINEAMENTO) ---
function updateLegend() {
    const gradient = d3.select("#legend-gradient");
    const labelsDiv = d3.select("#legend-labels");
    
    gradient.selectAll("*").remove(); 
    labelsDiv.selectAll("*").remove();

    if (colorMode === 'original') {
        // Nascondiamo completamente il contenitore delle label per evitare che il margin spinga su i pallini
        labelsDiv.style("display", "none"); 
        
        gradient.style("border", "none").style("background", "transparent").style("justify-content", "flex-end").style("gap", "8px");
        
        uniqueClasses.forEach(cls => {
            const item = gradient.append("div").attr("class", "legend-cluster-item");
            item.append("div").attr("class", "legend-cluster-dot").style("background-color", colorOriginal(cls));
            item.append("span").text(`Prod ${cls}`);
        });
        
        const anomalyItem = gradient.append("div").attr("class", "legend-cluster-item");
        anomalyItem.append("div").attr("class", "legend-cluster-dot").style("background-color", "#e74c3c");
        anomalyItem.append("span").text(`Anomaly`);
        
    } else {
        // Mostriamo il div dei numeri
        labelsDiv.style("display", "flex"); 
        
        gradient.style("border", "1px solid #ccc").style("gap", "0");
        let colors = [];
        let minT = "", maxT = "";

        if (colorMode === 'precision') { colors = bluesDiscrete; minT = "0% (Many FPs)"; maxT = "100% (Pure)"; }
        if (colorMode === 'recall') { colors = redsDiscrete; minT = "0% (Many FNs)"; maxT = "100% (Cohesive)"; }
        if (colorMode === 'fscore') { colors = rdYlGnDiscrete; minT = "0% (Bad)"; maxT = "100% (Perfect)"; }
        
        colors.forEach(c => gradient.append("div").attr("class", "legend-color-block").style("background-color", c));
        labelsDiv.append("span").text(minT);
        labelsDiv.append("span").text(maxT);
    }
}

// --- GAUGES ---
function initGauge(selector, gaugeObj) {
    const svg = d3.select(selector);
    const width = 100, height = 60;

    svg.attr("viewBox", `0 0 ${width} ${height}`)
       .style("width", "100%")
       .style("height", "100%");

    const g = svg.append("g").attr("transform", `translate(${width/2},${height - 5})`);
    const arcBg = d3.arc().innerRadius(30).outerRadius(45).startAngle(-Math.PI / 2).endAngle(Math.PI / 2);
    g.append("path").attr("d", arcBg).attr("fill", "#e0e0e0");

    const arcFg = d3.arc().innerRadius(30).outerRadius(45).startAngle(-Math.PI / 2).cornerRadius(3);
    gaugeObj.foreground = g.append("path").datum({ endAngle: -Math.PI / 2 }).attr("fill", "#bdc3c7").attr("d", arcFg);
}

function updateGauge(gaugeObj, value, color, textSelector) {
    const targetAngle = gaugeAngleScale(value);
    const arcFg = d3.arc().innerRadius(30).outerRadius(45).startAngle(-Math.PI / 2).cornerRadius(3);

    d3.select(textSelector).text((value * 100).toFixed(1) + "%");

    gaugeObj.foreground.transition().duration(750)
        .attrTween("d", function(d) {
            const interpolate = d3.interpolate(d.endAngle, targetAngle);
            return function(t) {
                d.endAngle = interpolate(t);
                return arcFg(d);
            };
        })
        .attr("fill", color);
}

// --- TOOLTIPS ESATTI ---
function showTooltip(event, d) {
    const tooltip = d3.select("#tooltip");
    
    let fpText = d.precision < 0.9 
        ? `🔴 <span style="color: #f1c40f; font-style: italic; font-weight: bold;">False Positive:</span> Attracts <span style="color: white; font-weight: bold;">${((1 - d.precision)*100).toFixed(1)}%</span> of points from other classes.` 
        : `🟢 <span style="color: #f1c40f; font-style: italic; font-weight: bold;">Low FPs:</span> No class mixing.`;
        
    let fnText = d.recall < 0.9 
        ? `🔴 <span style="color: #f1c40f; font-style: italic; font-weight: bold;">False Negative:</span> Disconnected from <span style="color: white; font-weight: bold;">${((1 - d.recall)*100).toFixed(1)}%</span> of points in its own class.` 
        : `🟢 <span style="color: #f1c40f; font-style: italic; font-weight: bold;">Low FNs:</span> Highly cohesive.`;

    let anomalyText = d.is_anomaly ? `<span style='color: #e74c3c;'>⚠️ <strong>Anomaly:</strong> True P${d.label} assigned to KMeans C${d.kmeans_cluster}</span><br>` : "";

    tooltip.html(`
        <strong style="color: #f1c40f;">ID:</strong> ${d.id} | <strong style="color: #f1c40f;">Class:</strong> ${d.label}<br>
        <strong style="color: #f1c40f;">Precision:</strong> ${d.precision === 1 ? "100" : (d.precision*100).toFixed(1)}%<br>
        <strong style="color: #f1c40f;">Recall:</strong> ${d.recall === 1 ? "100" : (d.recall*100).toFixed(1)}%<br>
        <strong style="color: #f1c40f;">F-Score:</strong> ${d.f_score === 1 ? "100" : (d.f_score*100).toFixed(1)}%
        <hr style="border: 0; border-top: 1px solid #555; margin: 8px 0 6px 0;">
        <div style="color: #ecf0f1; line-height: 1.4;">
            ${anomalyText}
            ${fpText}<br>
            ${fnText}
        </div>
    `);

    const tooltipNode = tooltip.node();
    const tooltipWidth = tooltipNode.offsetWidth;
    const tooltipHeight = tooltipNode.offsetHeight;
    const margin = 20;

    let x = event.pageX + margin;
    if (x + tooltipWidth > window.innerWidth) { x = event.pageX - tooltipWidth - margin; }

    let y = event.pageY + margin;
    if (y + tooltipHeight > window.innerHeight) { y = event.pageY - tooltipHeight - margin; }

    tooltip.style("left", x + "px").style("top", y + "px").transition().duration(100).style("opacity", 1);
}

function hideTooltip() {
    d3.select("#tooltip").transition().duration(200).style("opacity", 0);
}

function showAttributeTooltip(event, d) {
    const tooltip = d3.select("#tooltip");
    if (!d.attributes) return showTooltip(event, d); 

    const formatKey = (key) => key.charAt(0).toUpperCase() + key.slice(1).replace(/_/g, ' ');

    const attributesHtml = Object.entries(d.attributes)
        .map(([key, value]) => `<strong><span style="color: #f1c40f;">${formatKey(key)}:</span></strong> <span style="color: white;">${value}</span>`).join('<br>');

    tooltip.html(attributesHtml);

    const tooltipNode = tooltip.node();
    const tooltipWidth = tooltipNode.offsetWidth;
    const tooltipHeight = tooltipNode.offsetHeight;
    const margin = 20;

    let x = event.pageX + margin;
    if (x + tooltipWidth > window.innerWidth) { x = event.pageX - tooltipWidth - margin; }
    let y = event.pageY + margin;
    if (y + tooltipHeight > window.innerHeight) { y = event.pageY - tooltipHeight - margin; }

    tooltip.style("left", x + "px").style("top", y + "px").transition().duration(100).style("opacity", 1);
}

// --- NEIGHBOR GRAPH ---
function drawNeighborGraph(centerNode, neighborNodes) {
    const svg = d3.select("#neighbor-graph-svg");
    svg.selectAll("*").remove();

    const parentBox = document.getElementById("dynamic-svg-box");
    const width = (parentBox.clientWidth || 300) - 20;
    const height = (parentBox.clientHeight || 300) - 20;

    const size = Math.min(width, height);
    const xOffset = (width - size) / 2;
    const yOffset = (height - size) / 2;
    
    const g = svg.append("g").attr("transform", `translate(${xOffset}, ${yOffset})`);

    const graphNodes = [centerNode, ...neighborNodes].map(n => ({...n}));
    const graphLinks = neighborNodes.map(n => ({
        source: centerNode.id,
        target: n.id
    }));

    const centerGraphNode = graphNodes.find(n => n.id === centerNode.id);
    if (centerGraphNode) {
        centerGraphNode.fx = size / 2;
        centerGraphNode.fy = size / 2;
    }

    const simulation = d3.forceSimulation(graphNodes)
        .force("link", d3.forceLink(graphLinks).id(d => d.id).distance(size / 3.5).strength(0.7))
        .force("charge", d3.forceManyBody().strength(-size * 1.8))
        .force("center", d3.forceCenter(size / 2, size / 2));

    const link = g.append("g")
        .selectAll("line")
        .data(graphLinks)
        .join("line")
        .attr("class", "neighbor-link");

    const node = g.append("g")
        .selectAll("g")
        .data(graphNodes)
        .join("g")
        .attr("class", d => d.id === centerNode.id ? "neighbor-node center" : "neighbor-node")
        .call(drag(simulation, centerNode, size));

    node.append("circle")
        .attr("r", d => d.id === centerNode.id ? 20 : 15)
        .attr("fill", d => colorOriginal(d.label))
        .style("cursor", "pointer")
        .on("mouseover", function(event, d) {
            showAttributeTooltip(event, d);
        })
        .on("mouseout", function() {
            hideTooltip();
        });

    simulation.on("tick", () => {
        link.attr("x1", d => d.source.x).attr("y1", d => d.source.y)
            .attr("x2", d => d.target.x).attr("y2", d => d.target.y);
        node.attr("transform", d => `translate(${d.x},${d.y})`);
    });
}

function drag(simulation, centerNode, size) {
    function dragstarted(event, d) {
        if (!event.active) simulation.alphaTarget(0.3).restart();
        d.fx = d.x;
        d.fy = d.y;
    }
    function dragged(event, d) {
        d.fx = Math.max(0, Math.min(size, event.x));
        d.fy = Math.max(0, Math.min(size, event.y));
    }
    function dragended(event, d) {
        if (!event.active) simulation.alphaTarget(0);
        if (d.id !== centerNode.id) { 
            d.fx = null;
            d.fy = null;
        }
    }
    return d3.drag().on("start", dragstarted).on("drag", dragged).on("end", dragended);
}

// --- PARALLEL COORDINATES ---
function drawParallelCoordinates(containerSelector) {
    const container = d3.select(containerSelector); 
    if (container.empty()) return;
    
    container.html("");

    const width = container.node().clientWidth;
    const height = container.node().clientHeight;
    const margin = { top: 30, right: 30, bottom: 20, left: 30 };
    
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;

    const svgRoot = container.append("svg")
        .attr("viewBox", `0 0 ${width} ${height}`)
        .style("width", "100%")
        .style("height", "100%");

    svgRoot.on("mouseleave", resetAllHovers);
    svgRoot.on("click", () => {
        selectedPoint = null;
        d3.selectAll(".dot, .pc-line").style("opacity", 0.9);
        d3.selectAll(".link-group line").remove();
        resetAllHovers();
        updateLiveAnalytics([]); 
        if(d3.select("#show-discrepancies").property("checked")) toggleDiscrepancies(true);
    });

    const svg = svgRoot.append("g")
        .attr("transform", `translate(${margin.left},${margin.top})`);

    if (!dataset || dataset.length === 0 || !dataset[0].attributes) return;

    const features = Object.keys(dataset[0].attributes);

    const x = d3.scalePoint()
        .range([0, innerWidth])
        .padding(0.1)
        .domain(features);

    const y = {};
    features.forEach(f => {
        y[f] = d3.scaleLinear()
            .domain(d3.extent(dataset, d => +d.attributes[f]))
            .range([innerHeight, 0])
            .nice();
    });

    function path(d) {
        return d3.line()(features.map(f => [x(f), y[f](d.attributes[f])]));
    }

    const linesGroup = svg.append("g").attr("class", "pc-lines-group");
    const axesGroup = svg.append("g").attr("class", "pc-axes-group");

    linesGroup.selectAll(".pc-line")
        .data(dataset)
        .enter().append("path")
        .attr("class", d => `pc-line pt-${d.id}`)
        .attr("d", path)
        .style("fill", "none")
        .style("stroke", d => {
            if (colorMode === 'original') return d.is_anomaly ? '#e74c3c' : colorOriginal(d.label);
            return getColor(d);
        })
        .style("stroke-width", 1.5)
        .style("opacity", 0.6)
        .on("mouseover", function(event, d) {
            resetAllHovers();
            d3.selectAll(`.dot.pt-${d.id}`)
              .attr("r", currentPointSize * 1.6)
              .attr("stroke", "rgba(0,0,0,0.9)")
              .attr("stroke-width", 2).raise();
            d3.selectAll(`.pc-line.pt-${d.id}`)
              .style("stroke-width", 3).raise();
            showAttributeTooltip(event, d);
        })
        .on("mouseout", function() { resetAllHovers(); })
        .on("click", function(event, d) {
            event.stopPropagation();
            updateSelection(d);
        });

    const axes = axesGroup.selectAll(".pc-axis")
        .data(features).enter()
        .append("g")
        .attr("class", "pc-axis")
        .attr("transform", f => `translate(${x(f)},0)`);

    axes.each(function(f) {
        d3.select(this).call(d3.axisLeft(y[f]).ticks(5));
    });

    axes.append("text")
        .style("text-anchor", "middle")
        .attr("y", -15)
        .text(f => {
            let name = f.replace(/_/g, ' ');
            return name.charAt(0).toUpperCase() + name.slice(1);
        })
        .style("fill", "#2c3e50")
        .style("font-size", "0.65rem")
        .style("font-weight", "bold");
}