// --- GLOBAL STATE ---
let dataset = [];
let metadata = {};
let colorMode = 'original';
let uniqueClasses = [];
let pointById = new Map();
let currentPointSize = 5;
let selectedPoint = null;
let brushedPointsGlobal = []; 
let kmeansProjectionSource = 'pca'; 
let currentDatasetName = "wine";

// Filters
let minPrecision = 0;
let minRecall = 0;

// Scales Storage
const scalesMap = { pca: {}, mds: {}, kmeans: {}, pca2d: {}, mds2d: {} };

// D3 Brushes
let brushPCA, brushMDS, brushKMeans, brushPCA2D, brushMDS2D;

// Radar Chart Data
let radarDimensions = [];
let radarMinMax = {};
let origClusterAvg = {};
let kmeansClusterAvg = {};

// Color Scales
const customTableau = [...d3.schemeTableau10];
customTableau[2] = '#2ca02c'; 
const colorOriginal = d3.scaleOrdinal(customTableau);
const bluesDiscrete = ["#08519c", "#3182bd", "#6baed6", "#9ecae1", "#c6dbef"];
const colorPrecision = d3.scaleQuantize().domain([0, 1]).range(bluesDiscrete); 
const redsDiscrete = ["#67000d", "#cb181d", "#fb6a4a", "#fcae91", "#fee5d9"];
const colorRecall = d3.scaleQuantize().domain([0, 1]).range(redsDiscrete);
const rdYlGnDiscrete = ["#d73027", "#fdae61", "#ffffbf", "#a6d96a", "#1a9641"];
const colorFScore = d3.scaleQuantize().domain([0, 1]).range(rdYlGnDiscrete);
const colorKMeans = d3.scaleOrdinal(d3.schemeCategory10);

// Gauges
const gauges = { precision: { foreground: null }, recall: { foreground: null }, fscore: { foreground: null } };
const gaugeAngleScale = d3.scaleLinear().domain([0, 1]).range([-Math.PI / 2, Math.PI / 2]);


// --- DYNAMIC DASHBOARD INITIALIZATION ---
function initDashboard(folder) {
    const targetPlotsToClear = [
        "#pca-plot", "#mds-plot", "#kmeans-plot", 
        "#pca-plot-2d", "#mds-plot-2d", 
        "#pc-plot", "#comparison-plot", 
        "#radar-chart-svg-container", "#neighbor-graph-svg"
    ];
    targetPlotsToClear.forEach(selector => d3.select(selector).html(""));
    d3.select("#cm-table").html("");
    
    selectedPoint = null;
    brushedPointsGlobal = [];
    pointById.clear();
    
    brushPCA = d3.brush();
    brushMDS = d3.brush();
    brushKMeans = d3.brush();
    brushPCA2D = d3.brush();
    brushMDS2D = d3.brush();

    d3.select("#empty-state-placeholder").classed("hidden-panel", false);
    d3.select("#gauges-container").classed("hidden-panel", true);
    d3.select("#confusion-matrix-container").classed("hidden-panel", true);
    d3.select("#radar-empty-state").classed("hidden-panel", true);
    d3.select("#radar-chart-container").classed("hidden-panel", true);
    d3.select("#neighbor-graph-container").classed("hidden-panel", true);
    d3.select("#dynamic-panel-title").text("Live Analytics");

    const basePath = `../json/${folder}/`;

    Promise.all([
        d3.json(`${basePath}step2_final_data.json?v=${Date.now()}`),
        d3.csv(`../../dataset/${folder}.csv`),
        d3.json(`${basePath}kmeans_results.json?v=${Date.now()}`),
        d3.json(`${basePath}kmeans_2d_results.json?v=${Date.now()}`)
    ]).then(([data, csvData, kmeansData, kmeans2dData]) => {
        
        if (csvData && csvData.length > 0) {
            const allKeys = Object.keys(csvData[0]);
            radarDimensions = allKeys.filter(k => !['producer', 'label', 'class', 'species', 'variety', 'unnamed', 'uns'].some(sub => k.toLowerCase().includes(sub)));
        }
        
        d3.select("label[for='tab-nd']").text(`MDS vs PCA ${radarDimensions.length}D`);

        const kmeansMap = new Map();
        if(kmeansData?.points) kmeansData.points.forEach(p => kmeansMap.set(p.id, p));

        const kmeans2dMap = new Map();
        if(kmeans2dData?.points) kmeans2dData.points.forEach(p => kmeans2dMap.set(p.id, p));

        const formatLabel = (str) => {
            if (!str || str === "undefined") return "Unknown";
            return String(str).replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
        };

        data.points.forEach((p, i) => {
            if (csvData[i]) p.attributes = csvData[i];

            p.label = formatLabel(p.label);
            
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

        uniqueClasses = Array.from(new Set(dataset.map(d => String(d.label)))).sort();
        pointById = new Map(dataset.map(p => [p.id, p]));

        colorOriginal.domain(uniqueClasses);
        colorKMeans.domain(uniqueClasses);

        radarMinMax = {};
        origClusterAvg = {};
        kmeansClusterAvg = {};

        radarDimensions.forEach(dim => {
            radarMinMax[dim] = {
                min: d3.min(dataset, d => +d.attributes[dim]),
                max: d3.max(dataset, d => +d.attributes[dim])
            };
        });

        uniqueClasses.forEach(c => {
            let pts = dataset.filter(d => String(d.label) === c);
            origClusterAvg[c] = {};
            radarDimensions.forEach(dim => {
                origClusterAvg[c][dim] = d3.mean(pts, d => +d.attributes[dim]);
            });
        });

        const pcaKmeansClasses = Array.from(new Set(dataset.map(d => String(d.pca_kmeans_cluster)).filter(c => c !== "undefined")));
        pcaKmeansClasses.forEach(c => {
            let pts = dataset.filter(d => String(d.pca_kmeans_cluster) === c);
            kmeansClusterAvg[c] = {};
            radarDimensions.forEach(dim => {
                kmeansClusterAvg[c][dim] = d3.mean(pts, d => +d.attributes[dim]);
            });
        });

        if(metadata) {
            if (metadata.global_assessment && metadata.global_assessment.pca && metadata.global_assessment.mds) {
                d3.select("#fb-pca-trust").text((metadata.global_assessment.pca.trustworthiness * 100).toFixed(1) + "%");
                d3.select("#fb-pca-cont").text((metadata.global_assessment.pca.continuity * 100).toFixed(1) + "%");
                d3.select("#fb-mds-trust").text((metadata.global_assessment.mds.trustworthiness * 100).toFixed(1) + "%");
                d3.select("#fb-mds-cont").text((metadata.global_assessment.mds.continuity * 100).toFixed(1) + "%");
            }
            let gFScore = metadata.global_f_score || (metadata.global_assessment ? metadata.global_assessment.global_f_score : 0);
            d3.select("#fb-global-fscore").text(gFScore !== undefined ? (gFScore * 100).toFixed(1) + "%" : "N/A");
        }

        d3.select("#app-grid").classed("mode-2d", false);
        d3.selectAll(".tab-nd").classed("hidden-panel", false);
        d3.selectAll(".tab-2d").classed("hidden-panel", true);
        document.body.clientWidth; 

        drawPlot("#pca-plot", "pca_x", "pca_y", "pca", brushPCA);
        drawPlot("#mds-plot", "mds_x", "mds_y", "mds", brushMDS);
        drawPlot("#kmeans-plot", kmeansProjectionSource === 'pca' ? 'pca_x' : 'mds_x', kmeansProjectionSource === 'pca' ? 'pca_y' : 'mds_y', "kmeans", brushKMeans, d => {
            const showAnon = d3.select("#show-anomalies").property("checked");
            if (colorMode === 'original') return (showAnon && d.is_anomaly) ? 'var(--anomaly-color)' : colorOriginal(d.label);
            return getColor(d);
        });
        drawParallelCoordinates("#pc-plot");

        d3.select("#app-grid").classed("mode-2d", true);
        d3.selectAll(".tab-nd").classed("hidden-panel", true);
        d3.selectAll(".tab-2d").classed("hidden-panel", false);
        document.body.clientWidth; 

        const grid2d = d3.select("#plots-grid-2d");
        if (grid2d.select("#comparison-plot-container").empty()) {
            const newPlot = grid2d.append("div").attr("id", "comparison-plot-container").attr("class", "plot-container").style("grid-column", "1 / span 2");
            newPlot.append("div").attr("class", "plot-title").text("Cluster Agreement Flow"); 
            newPlot.append("div").attr("id", "comparison-plot").attr("class", "svg-container"); 
        }

        drawPlot("#pca-plot-2d", "pca_x", "pca_y", "pca2d", brushPCA2D, d => {
            const showAnon = d3.select("#show-anomalies").property("checked");
            if (colorMode === 'original') return (showAnon && d.pca_is_anomaly) ? 'var(--anomaly-color)' : colorOriginal(d.label);
            return getColor(d);
        });
        drawPlot("#mds-plot-2d", "mds_x", "mds_y", "mds2d", brushMDS2D, d => {
            const showAnon = d3.select("#show-anomalies").property("checked");
            if (colorMode === 'original') return (showAnon && d.mds_is_anomaly) ? 'var(--anomaly-color)' : colorOriginal(d.label);
            return getColor(d);
        });
        drawSankeyDiagram("#comparison-plot", dataset);

        const isND = d3.select("input[name='mainTab']:checked").node().value === 'nd';
        d3.select("#app-grid").classed("mode-2d", !isND);
        d3.selectAll(".tab-nd").classed("hidden-panel", !isND);
        d3.selectAll(".tab-2d").classed("hidden-panel", isND);
        document.body.clientWidth; 

        setupBrushing();
        updateLegend();
        updateLiveAnalytics([]); 
        
        if (d3.select("#show-discrepancies").property("checked")) toggleDiscrepancies(true);

    }).catch(err => {
        console.error(`Error loading dataset [${folder}]:`, err);
        alert(`Failed to load data for dataset: ${folder}. \n\nEnsure that the CSV file is named '${folder}.csv' and that you have run the Python scripts to generate JSON files in '../json/${folder}/'`);
    });
}

// Initial Boot and Theme Init
document.addEventListener("DOMContentLoaded", () => {
    initDashboard(currentDatasetName);
    initGauge("#gauge-precision", gauges.precision);
    initGauge("#gauge-recall", gauges.recall);
    initGauge("#gauge-fscore", gauges.fscore);
    enhanceColorModeSwitcher();
    
    // Theme Toggle Initialization
    const themeToggle = document.getElementById('theme-toggle');
    if (themeToggle) {
        themeToggle.addEventListener('click', () => {
            const currentTheme = document.documentElement.getAttribute('data-theme') || 'light';
            const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
            document.documentElement.setAttribute('data-theme', newTheme);
        });
    }
});

// --- GLOBAL EVENT LISTENERS ---
d3.select("#dataset-selector").on("change", function() {
    currentDatasetName = this.value;
    initDashboard(currentDatasetName);
});

d3.select("#point-size-slider").on("input change", function() {
    currentPointSize = +this.value;
    updateColors(); 
});

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

d3.select("#show-anomalies").on("change", function() {
    updateColors();
});

d3.selectAll("input[name='kmeansSource']").on("change", function() {
    kmeansProjectionSource = this.value;
    redrawKMeansPlot();
});

d3.selectAll("input[name='mainTab']").on("change", function() {
    const selectedTab = this.value;
    
    if (selectedTab === 'nd') {
        d3.select("#app-grid").classed("mode-2d", false);
        d3.selectAll(".tab-nd").classed("hidden-panel", false);
        d3.selectAll(".tab-2d").classed("hidden-panel", true);
    } else {
        d3.select("#app-grid").classed("mode-2d", true);
        d3.selectAll(".tab-nd").classed("hidden-panel", true);
        d3.selectAll(".tab-2d").classed("hidden-panel", false);
        
        const activeData = brushedPointsGlobal.length > 1 
            ? brushedPointsGlobal.filter(d => d.precision >= minPrecision && d.recall >= minRecall)
            : dataset.filter(d => d.precision >= minPrecision && d.recall >= minRecall);
        drawSankeyDiagram("#comparison-plot", activeData);
    }

    if (selectedPoint) {
        updateSelection(selectedPoint);
    } else if (brushedPointsGlobal.length > 0) {
        updateLiveAnalytics(brushedPointsGlobal);
    } else {
        updateLiveAnalytics([]);
    }
    
    if(d3.select("#show-discrepancies").property("checked")) toggleDiscrepancies(true);
});

// --- PLOTTING LOGIC ---
function redrawKMeansPlot() {
    const xKey = kmeansProjectionSource === 'pca' ? 'pca_x' : 'mds_x';
    const yKey = kmeansProjectionSource === 'pca' ? 'pca_y' : 'mds_y';

    d3.select("#kmeans-plot").html("");

    drawPlot("#kmeans-plot", xKey, yKey, "kmeans", brushKMeans, d => {
        const showAnon = d3.select("#show-anomalies").property("checked");
        if (colorMode === 'original') return (showAnon && d.is_anomaly) ? 'var(--anomaly-color)' : colorOriginal(d.label);
        return getColor(d);
    });

    applyFilters();
    if (d3.select("#show-discrepancies").property("checked")) toggleDiscrepancies(true);
    if (selectedPoint) updateSelection(selectedPoint);
}

function drawPlot(containerSelector, xKey, yKey, plotId, brushObj, customColorFn = null) {
    const container = d3.select(containerSelector); 
    const width = container.node().clientWidth || 400; 
    const height = container.node().clientHeight || 300;
    const margin = { top: 20, right: 25, bottom: 35, left: 35 };

    const svgRoot = container.append("svg")
        .attr("viewBox", `0 0 ${width} ${height}`)
        .style("width", "100%")
        .style("height", "100%");

    svgRoot.on("mouseleave", resetAllHovers);
    svgRoot.on("click", () => {
        selectedPoint = null;
        brushedPointsGlobal = [];
        d3.selectAll(".dot, .pc-line").style("opacity", 0.9);
        d3.selectAll(".link-group line").remove();
        resetAllHovers();
        updateLiveAnalytics([]); 
        if(d3.select("#show-discrepancies").property("checked")) toggleDiscrepancies(true);
    });

    const innerWidth = Math.max(10, width - margin.left - margin.right);
    const innerHeight = Math.max(10, height - margin.top - margin.bottom);

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
        .attr("r", d => {
            const showAnon = d3.select("#show-anomalies").property("checked");
            let isAnom = false;
            if (plotId === 'pca2d') isAnom = d.pca_is_anomaly;
            else if (plotId === 'mds2d') isAnom = d.mds_is_anomaly;
            else if (plotId === 'kmeans') isAnom = d.is_anomaly;
            
            return (showAnon && isAnom) ? currentPointSize * 1.8 : currentPointSize;
        })
        .style("fill", d => customColorFn ? customColorFn(d) : getColor(d))
        .style("stroke", "var(--dot-stroke)")
        .style("stroke-width", 0.8)
        .style("opacity", 0.9)
        .style("cursor", "pointer")
        .on("mouseover", function(event, d) {
            resetAllHovers();
            d3.selectAll(`.dot.pt-${d.id}`)
              .attr("r", currentPointSize * 2.0)
              .style("stroke", "var(--hover-stroke)")
              .style("stroke-width", 2).raise();
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

// --- FILTERS & ANOMALIES ---
function applyFilters() {
    d3.selectAll(".dot, .pc-line")
        .classed("filtered-out", d => d.precision < minPrecision || d.recall < minRecall)
        .style("display", d => (d.precision < minPrecision || d.recall < minRecall) ? "none" : null)
        .style("pointer-events", d => (d.precision < minPrecision || d.recall < minRecall) ? "none" : "auto");
        
    if(d3.select("#show-discrepancies").property("checked")) toggleDiscrepancies(true);
    
    const currentTab = d3.select("input[name='mainTab']:checked").node().value;
    if (currentTab === '2d') {
         const activeData = brushedPointsGlobal.length > 1 
            ? brushedPointsGlobal.filter(d => d.precision >= minPrecision && d.recall >= minRecall) 
            : dataset.filter(d => d.precision >= minPrecision && d.recall >= minRecall);
         drawSankeyDiagram("#comparison-plot", activeData);
    }
}

function toggleDiscrepancies(show) {
    const plotMapping = { kmeans: '#kmeans-plot', pca2d: '#pca-plot-2d', mds2d: '#mds-plot-2d' };
    const allPlots = ['#pca-plot', '#mds-plot', '#kmeans-plot', '#pca-plot-2d', '#mds-plot-2d'];
    const currentTab = d3.select("input[name='mainTab']:checked").node().value;

    if (show) {
        allPlots.forEach(selector => d3.select(`${selector} svg g .centroid-layer`).selectAll("*").remove()); 

        Object.keys(plotMapping).forEach(plotId => {
            const scales = scalesMap[plotId];
            const svgContainer = d3.select(`${plotMapping[plotId]} svg g .centroid-layer`);
            if (svgContainer.empty()) return;

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
                
                svgContainer.append("line")
                    .attr("x1", scales.xScale(d[scales.xKey])).attr("y1", scales.yScale(d[scales.yKey]))
                    .attr("x2", centroids[d[clusterProp]].x).attr("y2", centroids[d[clusterProp]].y)
                    .attr("class", (d[anomalyProp] ? "centroid-link centroid-anomaly" : "centroid-link centroid-correct") + ` pt-${d.id}`);
            });

            validKMeansClusters.forEach(k => {
                if(centroids[k].count === 0) return;
                svgContainer.append("path")
                    .attr("d", d3.symbol().type(d3.symbolCross).size(150)())
                    .attr("transform", `translate(${centroids[k].x}, ${centroids[k].y})`)
                    .style("fill", colorKMeans(k))
                    .style("stroke", "var(--sankey-node-stroke)") 
                    .style("stroke-width", 1.5);
            });
        });

        d3.selectAll(".dot.dot-pca:not(.filtered-out), .dot.dot-mds:not(.filtered-out)").style("opacity", 0.9);
        d3.selectAll(".dot.dot-kmeans:not(.filtered-out)").style("opacity", d => d.is_anomaly ? 1.0 : 0.2);
        d3.selectAll(".dot.dot-pca2d:not(.filtered-out)").style("opacity", d => d.pca_is_anomaly ? 1.0 : 0.2);
        d3.selectAll(".dot.dot-mds2d:not(.filtered-out)").style("opacity", d => d.mds_is_anomaly ? 1.0 : 0.2);
        d3.selectAll(".pc-line:not(.filtered-out)").style("opacity", 0.6).style("stroke-width", 1.5);

        let activePoints = brushedPointsGlobal.length > 0 ? brushedPointsGlobal : dataset.filter(d => d.precision >= minPrecision && d.recall >= minRecall);
        if (!selectedPoint) updateConfusionMatrix(activePoints, currentTab);
        
    } else {
        allPlots.forEach(selector => d3.select(`${selector} svg g .centroid-layer`).selectAll("*").remove());
        d3.selectAll(".dot:not(.filtered-out)").style("opacity", 0.9);
        d3.selectAll(".pc-line:not(.filtered-out)").style("opacity", 0.6).style("stroke-width", 1.5);
        d3.select("#confusion-matrix-container").classed("hidden-panel", true);
        if (!selectedPoint && brushedPointsGlobal.length === 0) {
            d3.select("#empty-state-placeholder").classed("hidden-panel", false);
            d3.select("#dynamic-panel-title").text("Live Analytics");
        }
    }
    
    // RESTORE SELECTION
    if (selectedPoint) {
        const currentTab = d3.select("input[name='mainTab']:checked").node().value;
        const activeIds = new Set([selectedPoint.id, ...(selectedPoint.neighbors || [])]);
        
        d3.selectAll(".dot:not(.filtered-out)")
            .style("opacity", p => {
                if (p.id === selectedPoint.id) return 1;
                if (currentTab === 'nd' && activeIds.has(p.id)) return 0.8;
                return 0.1;
            });
            
        d3.selectAll(".pc-line:not(.filtered-out)")
            .style("opacity", p => {
                if (p.id === selectedPoint.id) return 1;
                if (currentTab === 'nd' && activeIds.has(p.id)) return 0.6;
                return 0.05;
            });

        if (currentTab === 'nd') {
            d3.selectAll(".pc-line:not(.filtered-out)").filter(p => activeIds.has(p.id)).raise();
            d3.selectAll(".dot:not(.filtered-out)").filter(p => activeIds.has(p.id)).raise();
        } else {
            d3.selectAll(".dot:not(.filtered-out)").filter(p => p.id === selectedPoint.id).raise();
        }
        
    } else if (brushedPointsGlobal.length > 0) {
        dataset.forEach(d => {
            const isSelected = brushedPointsGlobal.includes(d);
            d3.selectAll(`.pt-${d.id}:not(.filtered-out)`).style("opacity", isSelected ? 0.9 : 0.15);
        });
    }
}

function updateConfusionMatrix(activePoints, tabMode) {
    d3.select("#empty-state-placeholder").classed("hidden-panel", true);
    d3.select("#gauges-container").classed("hidden-panel", true);
    d3.select("#neighbor-graph-container").classed("hidden-panel", true);
    d3.select("#radar-empty-state").classed("hidden-panel", true);
    d3.select("#radar-chart-container").classed("hidden-panel", true);
    d3.select("#confusion-matrix-container").classed("hidden-panel", false);

    const classes = Array.from(new Set(dataset.map(d => String(d.label)))).sort();
    d3.select("#dynamic-panel-title").text("Cluster Discrepancies");
    d3.select("#cm-table").html("");

    const generateTableHtml = (clusterKey, title) => {
        const matrix = {};
        classes.forEach(r => { matrix[r] = {}; classes.forEach(c => matrix[r][c] = 0); });
        
        activePoints.forEach(p => {
            const trueL = String(p.label); 
            const predL = String(p[clusterKey]); 
            if (matrix[trueL] && matrix[trueL][predL] !== undefined) matrix[trueL][predL]++;
        });

        // Use CSS Variables for dynamic styling in matrix
        let html = `<h5 style='margin: 15px 0 5px 0; color: var(--text-title); border-bottom: 1px solid var(--border-color);'>${title}</h5>`;
        html += "<table style='border-collapse: collapse; width: 100%; text-align: center; font-size: 0.75rem; margin-bottom: 10px;'>";
        html += "<thead><tr><th style='border-bottom: 1px solid var(--border-color); color: var(--text-title);'>GT ↓ \\ KMeans →</th>";
        classes.forEach(c => { html += `<th style='border-bottom: 1px solid var(--border-color); color: var(--text-title);'>C${c}</th>`; });
        html += "</tr></thead><tbody>";
        
        classes.forEach(row => {
            html += `<tr><td style='font-weight:bold; border-right: 1px solid var(--border-color); color: var(--text-main);'>Class ${row}</td>`;
            classes.forEach(col => {
                const count = matrix[row][col];
                const isDiag = (row === col); 
                const style = isDiag ? "background: var(--cm-diag-bg); color: var(--cm-diag-text);" : (count > 0 ? "background: var(--cm-err-bg); color: var(--cm-err-text);" : "color: var(--text-muted);");
                html += `<td style='padding: 4px; border-bottom: 1px solid var(--border-color); ${style}'>${count}</td>`;
            });
            html += "</tr>";
        });
        return html + "</tbody></table>";
    };

    let fullHtml = "";
    if (tabMode === 'nd') {
        fullHtml = generateTableHtml('kmeans_cluster', 'Global N-Dim K-Means');
    } else {
        fullHtml = generateTableHtml('pca_kmeans_cluster', 'PCA 2D K-Means');
        fullHtml += generateTableHtml('mds_kmeans_cluster', 'MDS 2D K-Means');
    }
    
    d3.select("#cm-table").html(fullHtml);
}

// --- SYNC & INTERACTION ---
function updateSelection(d) {
    selectedPoint = d;
    brushedPointsGlobal = []; 
    const currentTab = d3.select("input[name='mainTab']:checked").node().value;

    [brushPCA, brushMDS, brushKMeans, brushPCA2D, brushMDS2D].forEach(b => {
        d3.selectAll(".brush-group").call(b.move, null);
    });

    d3.select("#empty-state-placeholder").classed("hidden-panel", true);
    d3.select("#gauges-container").classed("hidden-panel", true);

    if (currentTab === 'nd') {
        d3.select("#dynamic-panel-title").text("Neighbor Graph");
        d3.select("#radar-empty-state").classed("hidden-panel", true);
        d3.select("#radar-chart-container").classed("hidden-panel", true);
        d3.select("#neighbor-graph-container").classed("hidden-panel", false);
        d3.select("#confusion-matrix-container").classed("hidden-panel", true);

        const neighborIds = d.neighbors || [];
        const activeIds = new Set([d.id, ...neighborIds]);

        d3.selectAll(".dot:not(.filtered-out)")
            .style("opacity", p => p.id === d.id ? 1 : (activeIds.has(p.id) ? 0.8 : 0.1))
            .attr("r", function(p) {
                if (p.id === d.id) return currentPointSize * 2.0; 
                const showAnon = d3.select("#show-anomalies").property("checked");
                const plotClass = d3.select(this).attr("class");
                let isAnom = false;
                if (plotClass.includes("pca2d")) isAnom = p.pca_is_anomaly;
                else if (plotClass.includes("mds2d")) isAnom = p.mds_is_anomaly;
                else if (plotClass.includes("kmeans")) isAnom = p.is_anomaly;
                return (showAnon && isAnom) ? currentPointSize * 1.8 : currentPointSize;
            })
            .style("stroke-width", p => p.id === d.id ? 2 : 0.8);

        d3.selectAll(".pc-line:not(.filtered-out)")
            .style("opacity", p => p.id === d.id ? 1 : (activeIds.has(p.id) ? 0.6 : 0.05))
            .style("stroke-width", p => p.id === d.id ? 3 : 1.5);

        d3.selectAll(".pc-line:not(.filtered-out)").filter(p => activeIds.has(p.id)).raise();
        d3.selectAll(".dot:not(.filtered-out)").filter(p => activeIds.has(p.id)).raise();

        const kmeansXKey = scalesMap['kmeans'].xKey;
        const kmeansYKey = scalesMap['kmeans'].yKey;

        drawLines("pca", d, neighborIds, "pca_x", "pca_y", brushPCA);
        drawLines("mds", d, neighborIds, "mds_x", "mds_y", brushMDS);
        drawLines("kmeans", d, neighborIds, kmeansXKey, kmeansYKey, brushKMeans);

        const neighborsData = neighborIds.map(id => pointById.get(id)).filter(Boolean);
        drawNeighborGraph(d, neighborsData);

    } else {
        d3.select("#neighbor-graph-container").classed("hidden-panel", true);
        d3.select("#confusion-matrix-container").classed("hidden-panel", true);
        d3.select("#radar-empty-state").classed("hidden-panel", true);
        d3.select("#radar-chart-container").classed("hidden-panel", false);
        d3.select("#dynamic-panel-title").text("Multidimensional Profile (Radar)");

        drawRadarChart(d);

        d3.selectAll(".dot:not(.filtered-out)")
            .style("opacity", p => p.id === d.id ? 1 : 0.1)
            .attr("r", function(p) {
                if (p.id === d.id) return currentPointSize * 2.0; 
                const showAnon = d3.select("#show-anomalies").property("checked");
                const plotClass = d3.select(this).attr("class");
                let isAnom = false;
                if (plotClass.includes("pca2d")) isAnom = p.pca_is_anomaly;
                else if (plotClass.includes("mds2d")) isAnom = p.mds_is_anomaly;
                else if (plotClass.includes("kmeans")) isAnom = p.is_anomaly;
                return (showAnon && isAnom) ? currentPointSize * 1.8 : currentPointSize;
            })
            .style("stroke-width", p => p.id === d.id ? 2 : 0.8);

        d3.selectAll(".dot:not(.filtered-out)").filter(p => p.id === d.id).raise();
        d3.selectAll(".link-group line").remove();

        const baseDataset = dataset.filter(p => p.precision >= minPrecision && p.recall >= minRecall);
        drawSankeyDiagram("#comparison-plot", baseDataset);
    }
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
        .style("stroke", target => target.label === sourceD.label ? "#2ca02c" : "var(--anomaly-color)") 
        .style("stroke-width", 1.5)
        .style("opacity", 0.6);
}

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
            brushedPointsGlobal = [];
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
    
    brushedPointsGlobal = selectedPoints;
    updateLiveAnalytics(selectedPoints);
}

function updateLiveAnalytics(selectedPoints) {
    const currentTab = d3.select("input[name='mainTab']:checked").node().value;
    const isAnomalyOn = d3.select("#show-discrepancies").property("checked");
    const baseDataset = dataset.filter(d => d.precision >= minPrecision && d.recall >= minRecall);

    if (currentTab === '2d') {
        d3.select("#gauges-container").classed("hidden-panel", true);
        d3.select("#neighbor-graph-container").classed("hidden-panel", true);
        d3.select("#confusion-matrix-container").classed("hidden-panel", true);
        d3.select("#empty-state-placeholder").classed("hidden-panel", true);
        
        if (selectedPoints.length === 1) {
            d3.select("#dynamic-panel-title").text("Multidimensional Profile (Radar)");
            d3.select("#radar-empty-state").classed("hidden-panel", true);
            d3.select("#radar-chart-container").classed("hidden-panel", false);
            drawRadarChart(selectedPoints[0]);
        } else {
            d3.select("#dynamic-panel-title").text("Live Analytics");
            d3.select("#radar-chart-container").classed("hidden-panel", true);
            d3.select("#radar-empty-state").classed("hidden-panel", false);
        }
        
        const activeData = selectedPoints.length > 1 ? selectedPoints : baseDataset;
        drawSankeyDiagram("#comparison-plot", activeData);
        return;
    }

    d3.select("#radar-empty-state").classed("hidden-panel", true);
    d3.select("#radar-chart-container").classed("hidden-panel", true);

    if (selectedPoints.length === 0) {
        d3.select("#dynamic-panel-title").text("Live Analytics");
        d3.select("#gauges-container").classed("hidden-panel", true);
        d3.select("#neighbor-graph-container").classed("hidden-panel", true);
        
        if (isAnomalyOn) {
            d3.select("#empty-state-placeholder").classed("hidden-panel", true);
            d3.select("#confusion-matrix-container").classed("hidden-panel", false);
            updateConfusionMatrix(baseDataset, currentTab);
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
    d3.selectAll(".gauges-wrapper").style("display", "flex");
    
    const avgPrec = d3.mean(selectedPoints, d => d.precision);
    const avgRecall = d3.mean(selectedPoints, d => d.recall);
    const avgFScore = d3.mean(selectedPoints, d => d.f_score);

    updateGauge(gauges.precision, avgPrec, colorFScore(avgPrec), "#val-precision");
    updateGauge(gauges.recall, avgRecall, colorFScore(avgRecall), "#val-recall");
    updateGauge(gauges.fscore, avgFScore, colorFScore(avgFScore), "#val-fscore");
}

// --- RADAR CHART ---
function drawRadarChart(point) {
    const container = d3.select("#radar-chart-svg-container");
    container.html(""); 

    const baseSize = 500;
    const margin = 90; 
    const radius = (baseSize / 2) - margin;

    const svg = container.append("svg")
        .attr("viewBox", `0 0 ${baseSize} ${baseSize}`)
        .attr("preserveAspectRatio", "xMidYMid meet")
        .style("width", "100%")
        .style("height", "100%")
        .style("display", "block") 
        .append("g")
        .attr("transform", `translate(${baseSize / 2},${baseSize / 2})`);

    const normalize = (dim, value) => {
        const {min, max} = radarMinMax[dim];
        if (max === min) return 0;
        return (value - min) / (max - min);
    };

    const getCoordinates = (dataObj) => {
        return radarDimensions.map((dim, i) => {
            let val = normalize(dim, dataObj[dim]);
            let angle = (Math.PI / 2) + (2 * Math.PI * i / radarDimensions.length);
            return {
                x: -(radius * val * Math.cos(angle)), 
                y: -(radius * val * Math.sin(angle))
            };
        });
    };

    const ticks = [0.25, 0.5, 0.75, 1];
    svg.selectAll(".grid-circle")
        .data(ticks).enter()
        .append("circle")
        .attr("r", d => radius * d)
        .style("fill", "none")
        .style("stroke", "var(--radar-grid)") 
        .style("stroke-width", "0.5px");

    const truncateText = (str, maxLength) => str.length > maxLength ? str.substring(0, maxLength) + '...' : str;

    const axes = svg.selectAll(".axis")
        .data(radarDimensions).enter()
        .append("g")
        .attr("class", "axis");

    axes.append("line")
        .attr("x1", 0).attr("y1", 0)
        .attr("x2", (d, i) => -(radius * Math.cos((Math.PI / 2) + (2 * Math.PI * i / radarDimensions.length))))
        .attr("y2", (d, i) => -(radius * Math.sin((Math.PI / 2) + (2 * Math.PI * i / radarDimensions.length))))
        .style("stroke", "var(--radar-axis)") 
        .style("stroke-width", "1px");

    const textNodes = axes.append("text")
        .attr("x", (d, i) => -(radius + 20) * Math.cos((Math.PI / 2) + (2 * Math.PI * i / radarDimensions.length)))
        .attr("y", (d, i) => -(radius + 20) * Math.sin((Math.PI / 2) + (2 * Math.PI * i / radarDimensions.length)))
        .text(d => truncateText(d.replace(/_/g, ' '), 12)) 
        .style("text-anchor", (d, i) => {
            const calcX = -Math.cos((Math.PI / 2) + (2 * Math.PI * i / radarDimensions.length));
            if (calcX > 0.1) return "start"; 
            if (calcX < -0.1) return "end";  
            return "middle";                 
        })
        .style("alignment-baseline", "middle")
        .style("font-size", "11.5px")
        .style("fill", "var(--radar-text)") 
        .style("font-weight", "bold");

    textNodes.append("title")
        .text(d => d.replace(/_/g, ' '));

    const lineBuilder = d3.line().x(d => d.x).y(d => d.y).curve(d3.curveLinearClosed);

    const origData = origClusterAvg[String(point.label)];
    if (origData) {
        svg.append("path")
            .datum(getCoordinates(origData))
            .attr("d", lineBuilder)
            .style("fill", "none")
            .style("stroke", "var(--radar-orig)") 
            .style("stroke-width", "2px");
    }

    const kmeansData = kmeansClusterAvg[String(point.pca_kmeans_cluster)];
    if (kmeansData) {
        svg.append("path")
            .datum(getCoordinates(kmeansData))
            .attr("d", lineBuilder)
            .style("fill", "none")
            .style("stroke", "var(--radar-kmeans)") 
            .style("stroke-width", "2px");
    }

    const pointData = {};
    radarDimensions.forEach(dim => pointData[dim] = +point.attributes[dim]); 
    
    svg.append("path")
        .datum(getCoordinates(pointData))
        .attr("d", lineBuilder)
        .style("fill", "none")
        .style("stroke", "var(--radar-point)") 
        .style("stroke-width", "2.5px");
        
    svg.selectAll(".radar-point")
        .data(getCoordinates(pointData))
        .enter().append("circle")
        .attr("cx", d => d.x).attr("cy", d => d.y)
        .attr("r", 3)
        .style("fill", "var(--radar-point)");
}

// --- UTILITIES ---
function resetAllHovers() {
    const showAnon = d3.select("#show-anomalies").property("checked");
    
    const getBaseR = (nodeClass, d) => {
        let isAnom = false;
        if (nodeClass.includes("pca2d")) isAnom = d.pca_is_anomaly;
        else if (nodeClass.includes("mds2d")) isAnom = d.mds_is_anomaly;
        else if (nodeClass.includes("kmeans")) isAnom = d.is_anomaly;
        return (showAnon && isAnom) ? currentPointSize * 1.8 : currentPointSize;
    };

    if (selectedPoint) {
        d3.selectAll(".dot")
            .attr("r", function(p) { return p.id === selectedPoint.id ? currentPointSize * 2.0 : getBaseR(d3.select(this).attr("class"), p); })
            .style("stroke", p => p.id === selectedPoint.id ? "var(--hover-stroke)" : "var(--dot-stroke)")
            .style("stroke-width", p => p.id === selectedPoint.id ? 2 : 0.8);
        d3.selectAll(".pc-line")
            .style("stroke-width", p => p.id === selectedPoint.id ? 3 : 1.5);
    } else {
        d3.selectAll(".dot")
            .attr("r", function(p) { return getBaseR(d3.select(this).attr("class"), p); })
            .style("stroke", "var(--dot-stroke)")
            .style("stroke-width", 0.8);
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
    if (container.empty()) return;
    container.html(""); 

    container.classed("control-group", false).classed("segmented-control", true);

    options.forEach((opt, i) => {
        container.append("input")
            .attr("type", "radio")
            .attr("id", `cm-${opt.value}`)
            .attr("name", "colorMode")
            .attr("value", opt.value)
            .property("checked", opt.value === colorMode)
            .on("change", function() {
                colorMode = this.value;
                updateColors();
                updateLegend();
            }); 
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
    const showAnon = d3.select("#show-anomalies").property("checked");
    const getRadius = (isAnom) => (showAnon && isAnom) ? currentPointSize * 1.8 : currentPointSize;

    d3.selectAll(".dot.dot-pca").transition().duration(500)
        .style("fill", d => getColor(d))
        .attr("r", currentPointSize);

    d3.selectAll(".dot.dot-mds").transition().duration(500)
        .style("fill", d => getColor(d))
        .attr("r", currentPointSize);
    
    d3.selectAll(".dot.dot-pca2d").transition().duration(500)
        .style("fill", d => {
            if (colorMode === 'original') return (showAnon && d.pca_is_anomaly) ? 'var(--anomaly-color)' : colorOriginal(d.label);
            return getColor(d);
        })
        .attr("r", d => getRadius(d.pca_is_anomaly));
    
    d3.selectAll(".dot.dot-mds2d").transition().duration(500)
        .style("fill", d => {
            if (colorMode === 'original') return (showAnon && d.mds_is_anomaly) ? 'var(--anomaly-color)' : colorOriginal(d.label);
            return getColor(d);
        })
        .attr("r", d => getRadius(d.mds_is_anomaly));
    
    d3.selectAll(".dot.dot-kmeans").transition().duration(500)
        .style("fill", d => {
            if (colorMode === 'original') return (showAnon && d.is_anomaly) ? 'var(--anomaly-color)' : colorOriginal(d.label);
            return getColor(d);
        })
        .attr("r", d => getRadius(d.is_anomaly));
    
    d3.selectAll(".pc-line").transition().duration(500)
        .style("stroke", d => {
            if (colorMode === 'original') return (showAnon && d.is_anomaly) ? 'var(--anomaly-color)' : colorOriginal(d.label);
            return getColor(d);
        });
}

function updateLegend() {
    const gradient = d3.select("#legend-gradient");
    const labelsDiv = d3.select("#legend-labels");
    
    gradient.selectAll("*").remove(); 
    labelsDiv.selectAll("*").remove();

    if (colorMode === 'original') {
        labelsDiv.style("display", "none"); 
        
        gradient.style("border", "none").style("background", "transparent").style("justify-content", "flex-end").style("gap", "8px");
        
        uniqueClasses.forEach(cls => {
            const item = gradient.append("div").attr("class", "legend-cluster-item");
            item.append("div").attr("class", "legend-cluster-dot").style("background-color", colorOriginal(cls));
            item.append("span").text(cls); 
        });
        
    } else {
        labelsDiv.style("display", "flex"); 
        
        gradient.style("border", "1px solid var(--border-color)").style("gap", "0");
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

function initGauge(selector, gaugeObj) {
    const svg = d3.select(selector);
    const width = 100, height = 60;

    svg.attr("viewBox", `0 0 ${width} ${height}`).style("width", "100%").style("height", "100%");
    const g = svg.append("g").attr("transform", `translate(${width/2},${height - 5})`);
    const arcBg = d3.arc().innerRadius(30).outerRadius(45).startAngle(-Math.PI / 2).endAngle(Math.PI / 2);
    g.append("path").attr("d", arcBg).style("fill", "var(--gauge-bg)"); 

    const arcFg = d3.arc().innerRadius(30).outerRadius(45).startAngle(-Math.PI / 2).cornerRadius(3);
    gaugeObj.foreground = g.append("path").datum({ endAngle: -Math.PI / 2 }).style("fill", "var(--correct-color)").attr("d", arcFg);
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
        .style("fill", color);
}

function showTooltip(event, d) {
    const tooltip = d3.select("#tooltip");
    
    let fpText = d.precision < 0.9 
        ? `🔴 <span style="color: #f1c40f; font-style: italic; font-weight: bold;">False Positive:</span> Attracts <span style="color: white; font-weight: bold;">${((1 - d.precision)*100).toFixed(1)}%</span> of points from other classes.` 
        : `🟢 <span style="color: #f1c40f; font-style: italic; font-weight: bold;">Low FPs:</span> No class mixing.`;
        
    let fnText = d.recall < 0.9 
        ? `🔴 <span style="color: #f1c40f; font-style: italic; font-weight: bold;">False Negative:</span> Disconnected from <span style="color: white; font-weight: bold;">${((1 - d.recall)*100).toFixed(1)}%</span> of points in its own class.` 
        : `🟢 <span style="color: #f1c40f; font-style: italic; font-weight: bold;">Low FNs:</span> Highly cohesive.`;

    let anomalyText = d.is_anomaly ? `<span style='color: var(--anomaly-color);'>⚠️ <strong>Anomaly:</strong> True P${d.label} assigned to KMeans C${d.kmeans_cluster}</span><br>` : "";

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
    if (x + tooltipWidth > window.innerWidth) x = event.pageX - tooltipWidth - margin;

    let y = event.pageY + margin;
    if (y + tooltipHeight > window.innerHeight) y = event.pageY - tooltipHeight - margin;

    tooltip.style("left", x + "px").style("top", y + "px").transition().duration(100).style("opacity", 1);
}

function hideTooltip() {
    d3.select("#tooltip").transition().duration(200).style("opacity", 0);
}

function showAttributeTooltip(event, d) {
    const tooltip = d3.select("#tooltip");
    if (!d.attributes) return showTooltip(event, d); 

    const formatKey = (key) => key.charAt(0).toUpperCase() + key.slice(1).replace(/_/g, ' ');

    const attributesHtml = radarDimensions
        .map(key => `<strong><span style="color: #f1c40f;">${formatKey(key)}:</span></strong> <span style="color: white;">${d.attributes[key]}</span>`).join('<br>');

    tooltip.html(attributesHtml);

    const tooltipNode = tooltip.node();
    const tooltipWidth = tooltipNode.offsetWidth;
    const tooltipHeight = tooltipNode.offsetHeight;
    const margin = 20;

    let x = event.pageX + margin;
    if (x + tooltipWidth > window.innerWidth) x = event.pageX - tooltipWidth - margin;
    let y = event.pageY + margin;
    if (y + tooltipHeight > window.innerHeight) y = event.pageY - tooltipHeight - margin;

    tooltip.style("left", x + "px").style("top", y + "px").transition().duration(100).style("opacity", 1);
}

function drawNeighborGraph(centerNode, neighborNodes) {
    const svg = d3.select("#neighbor-graph-svg");
    svg.selectAll("*").remove();

    const baseSize = 300;
    svg.attr("viewBox", `0 0 ${baseSize} ${baseSize}`)
       .attr("preserveAspectRatio", "xMidYMid meet");
    
    const g = svg.append("g");

    const centerRadius = 15; 
    const neighborRadius = 10;

    const graphNodes = [centerNode, ...neighborNodes].map(n => ({...n}));
    const graphLinks = neighborNodes.map(n => ({
        source: centerNode.id,
        target: n.id
    }));

    const centerGraphNode = graphNodes.find(n => n.id === centerNode.id);
    if (centerGraphNode) {
        centerGraphNode.fx = baseSize / 2;
        centerGraphNode.fy = baseSize / 2;
    }

    const simulation = d3.forceSimulation(graphNodes)
        .force("link", d3.forceLink(graphLinks).id(d => d.id).distance(80).strength(0.7))
        .force("charge", d3.forceManyBody().strength(-250))
        .force("center", d3.forceCenter(baseSize / 2, baseSize / 2));

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
        .call(drag(simulation, centerNode, baseSize));

    node.append("circle")
        .attr("r", d => d.id === centerNode.id ? centerRadius : neighborRadius)
        .style("fill", d => colorOriginal(d.label))
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
        d.fx = Math.max(15, Math.min(size - 15, event.x));
        d.fy = Math.max(15, Math.min(size - 15, event.y));
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

function drawSankeyDiagram(selector, activeData) {
    const container = d3.select(selector);
    if (container.empty()) return;
    container.html("");

    if (!activeData || activeData.length === 0) {
        container.append("div")
                 .style("text-align", "center")
                 .style("padding-top", "30px")
                 .style("color", "var(--text-muted)")
                 .style("font-size", "0.9rem")
                 .text("No points selected for comparison.");
        return;
    }

    const width = container.node().clientWidth;
    const height = container.node().clientHeight;
    
    const margin = { top: 15, right: 40, bottom: 50, left: 40 };

    const svg = container.append("svg")
        .attr("viewBox", `0 0 ${width} ${height}`)
        .style("width", "100%")
        .style("height", "100%")
        .append("g")
        .attr("transform", `translate(${margin.left},${margin.top})`);

    const pcaClusters = Array.from(new Set(activeData.map(d => String(d.pca_kmeans_cluster)).filter(c => c !== "undefined"))).sort();
    const mdsClusters = Array.from(new Set(activeData.map(d => String(d.mds_kmeans_cluster)).filter(c => c !== "undefined"))).sort();

    if (pcaClusters.length === 0 || mdsClusters.length === 0) return;

    const nodes = [];
    const nodeMap = new Map();
    let nodeIndex = 0;

    pcaClusters.forEach(c => {
        const name = `PCA C${c}`;
        nodes.push({ name: name, type: 'pca', cluster: c });
        nodeMap.set(name, nodeIndex++);
    });
    mdsClusters.forEach(c => {
        const name = `MDS C${c}`;
        nodes.push({ name: name, type: 'mds', cluster: c });
        nodeMap.set(name, nodeIndex++);
    });

    const linkMap = new Map();
    activeData.forEach(d => {
        if (d.pca_kmeans_cluster !== undefined && d.mds_kmeans_cluster !== undefined) {
            const sourceName = `PCA C${d.pca_kmeans_cluster}`;
            const targetName = `MDS C${d.mds_kmeans_cluster}`;
            const key = `${sourceName}->${targetName}`;
            
            const isDiscrepancy = String(d.pca_kmeans_cluster) !== String(d.mds_kmeans_cluster);
            
            if (!linkMap.has(key)) {
                linkMap.set(key, { 
                    source: nodeMap.get(sourceName), 
                    target: nodeMap.get(targetName), 
                    value: 0,
                    isDiscrepancy: isDiscrepancy,
                    points: [] 
                });
            }
            linkMap.get(key).value += 1;
            linkMap.get(key).points.push(d);
        }
    });

    const links = Array.from(linkMap.values());

    const maxDiscrepancy = d3.max(links.filter(l => l.isDiscrepancy), d => d.value) || 1;
    
    const baseColors = ["#fcae91", "#fb6a4a", "#ef3b2c", "#cb181d", "#99000d"];
    const numColors = Math.min(maxDiscrepancy, 5);
    const severityColors = baseColors.slice(0, numColors);
    
    const step = Math.max(1, Math.ceil(maxDiscrepancy / numColors));
    const thresholds = severityColors.map((_, i) => (i + 1) * step + 1).slice(0, -1);
    
    const severityScale = d3.scaleThreshold()
        .domain(thresholds)
        .range(severityColors);

    const legendHtml = container.append("div")
        .attr("class", "sankey-legend-html")
        .style("position", "absolute")
        .style("bottom", "7px") 
        .style("left", "50%")
        .style("transform", "translateX(-50%)")
        .style("display", "flex")
        .style("flex-direction", "column")
        .style("align-items", "center")
        .style("z-index", "10")
        .style("width", "150px")
        .style("pointer-events", "none"); 

    legendHtml.append("div")
        .style("font-size", "0.6rem")
        .style("font-weight", "bold")
        .style("color", "var(--text-title)") 
        .style("margin-bottom", "3px")
        .text("Discrepancy Severity");

    const colorStrip = legendHtml.append("div")
        .style("display", "flex")
        .style("width", "100%")
        .style("height", "8px") 
        .style("border-radius", "2px")
        .style("overflow", "hidden");

    severityColors.forEach(color => {
        colorStrip.append("div")
            .style("flex", "1")
            .style("height", "100%")
            .style("background-color", color);
    });

    const labelsStrip = legendHtml.append("div")
        .style("display", "flex")
        .style("width", "100%")
        .style("justify-content", "space-between")
        .style("margin-top", "2px");

    let prevVal = 1;
    severityColors.forEach((color, i) => {
        let th = thresholds[i];
        let label = (th === undefined) 
            ? (prevVal >= maxDiscrepancy ? `${prevVal}` : `${prevVal}+`) 
            : (prevVal === th - 1 ? `${prevVal}` : `${prevVal}-${th - 1}`);
        prevVal = th;
        
        labelsStrip.append("span")
            .style("flex", "1")
            .style("text-align", "center")
            .style("font-size", "0.55rem")
            .style("color", "var(--text-muted)") 
            .style("font-weight", "bold")
            .text(label);
    });

    const sankey = d3.sankey()
        .nodeWidth(20)
        .nodePadding(15)
        .extent([[0, 0], [Math.max(10, width - margin.left - margin.right), Math.max(10, height - margin.top - margin.bottom)]]);

    let graph;
    try {
        graph = sankey({
            nodes: nodes.map(d => Object.assign({}, d)),
            links: links.map(d => Object.assign({}, d))
        });
    } catch(e) {
        console.error("Sankey mapping error:", e);
        return;
    }

    graph.links.sort((a, b) => (a.isDiscrepancy === b.isDiscrepancy ? 0 : a.isDiscrepancy ? 1 : -1));

    const linkGroup = svg.append("g")
        .attr("fill", "none")
        .selectAll("g")
        .data(graph.links)
        .enter().append("g");

    linkGroup.append("path")
        .attr("d", d3.sankeyLinkHorizontal())
        .style("stroke", d => d.isDiscrepancy ? severityScale(d.value) : "var(--sankey-link-neutral)") 
        .style("stroke-width", d => d.isDiscrepancy ? 1.5 : Math.max(1, d.width))
        .style("stroke-opacity", d => d.isDiscrepancy ? 0.9 : 0.25)
        .attr("class", "sankey-link")
        .on("mouseover", function(event, d) {
            d3.selectAll(".sankey-link").style("stroke-opacity", 0.1);
            d3.select(this).style("stroke-opacity", 1.0).raise();
        })
        .on("mouseout", function() {
            d3.selectAll(".sankey-link").style("stroke-opacity", l => l.isDiscrepancy ? 0.9 : 0.25);
        })
        .on("click", function(event, d) {
            event.stopPropagation();
            
            d3.select("#show-discrepancies").property("checked", true);
            toggleDiscrepancies(true);
            
            const linkPointIds = new Set(d.points.map(p => p.id));
            
            d3.selectAll(".dot:not(.filtered-out)")
                .style("opacity", p => linkPointIds.has(p.id) ? 1 : 0.05)
                .attr("r", p => linkPointIds.has(p.id) ? currentPointSize * 1.5 : currentPointSize);
                
            d3.selectAll(".pc-line:not(.filtered-out)")
                .style("opacity", p => linkPointIds.has(p.id) ? 1 : 0.05)
                .style("stroke-width", p => linkPointIds.has(p.id) ? 2.5 : 1);
                
            d3.selectAll(".centroid-link")
                .style("display", function() {
                    const cls = d3.select(this).attr("class");
                    const idMatch = cls.match(/pt-(\d+)/);
                    if (idMatch && linkPointIds.has(+idMatch[1])) {
                        return null; 
                    }
                    return "none"; 
                });
                
            d3.selectAll(".dot:not(.filtered-out)").filter(p => linkPointIds.has(p.id)).raise();
        })
        .append("title")
        .text(d => `${d.source.name} → ${d.target.name}\n${d.value} points${d.isDiscrepancy ? ' (DISCREPANCY)' : ''}`);

    svg.append("g")
        .selectAll("rect")
        .data(graph.nodes)
        .enter().append("rect")
        .attr("x", d => d.x0)
        .attr("y", d => d.y0)
        .attr("height", d => Math.max(1, d.y1 - d.y0))
        .attr("width", d => d.x1 - d.x0)
        .style("fill", d => colorKMeans(d.cluster))
        .style("stroke", "var(--sankey-node-stroke)") 
        .attr("class", "sankey-node")
        .append("title")
        .text(d => `${d.name}\n${d.value} points`);

    svg.append("g")
        .style("font-size", "11px")
        .style("font-weight", "bold")
        .selectAll("text")
        .data(graph.nodes)
        .enter().append("text")
        .attr("x", d => d.x0 < width / 2 ? d.x1 + 6 : d.x0 - 6)
        .attr("y", d => (d.y1 + d.y0) / 2)
        .attr("dy", "0.35em")
        .attr("text-anchor", d => d.x0 < width / 2 ? "start" : "end")
        .text(d => `${d.name} (${d.value})`);
}

function drawParallelCoordinates(containerSelector) {
    const container = d3.select(containerSelector); 
    if (container.empty()) return;
    container.html("");

    const width = container.node().clientWidth || 400; 
    const height = container.node().clientHeight || 300;
    const margin = { top: 30, right: 30, bottom: 20, left: 30 };
    
    const innerWidth = Math.max(10, width - margin.left - margin.right);
    const innerHeight = Math.max(10, height - margin.top - margin.bottom);

    const svgRoot = container.append("svg")
        .attr("viewBox", `0 0 ${width} ${height}`)
        .style("width", "100%")
        .style("height", "100%");

    svgRoot.on("mouseleave", resetAllHovers);
    svgRoot.on("click", () => {
        selectedPoint = null;
        brushedPointsGlobal = [];
        d3.selectAll(".dot, .pc-line").style("opacity", 0.9);
        d3.selectAll(".link-group line").remove();
        resetAllHovers();
        updateLiveAnalytics([]); 
        if(d3.select("#show-discrepancies").property("checked")) toggleDiscrepancies(true);
    });

    const svg = svgRoot.append("g").attr("transform", `translate(${margin.left},${margin.top})`);

    if (!dataset || dataset.length === 0 || !dataset[0].attributes) return;

    const features = radarDimensions;
    const x = d3.scalePoint().range([0, innerWidth]).padding(0.1).domain(features);

    const y = {};
    features.forEach(f => {
        const extent = d3.extent(dataset, d => +d.attributes[f]);
        const minVal = Math.min(0, extent[0]);
        const maxVal = Math.max(0, extent[1]);
        y[f] = d3.scaleLinear().domain([minVal, maxVal]).range([innerHeight, 0]).nice();
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
            const showAnon = d3.select("#show-anomalies").property("checked");
            if (colorMode === 'original') return (showAnon && d.is_anomaly) ? 'var(--anomaly-color)' : colorOriginal(d.label);
            return getColor(d);
        })
        .style("stroke-width", 1.5)
        .style("opacity", 0.6)
        .on("mouseover", function(event, d) {
            resetAllHovers();
            d3.selectAll(`.dot.pt-${d.id}`)
              .attr("r", currentPointSize * 2.0)
              .style("stroke", "var(--hover-stroke)")
              .style("stroke-width", 2).raise();
            d3.selectAll(`.pc-line.pt-${d.id}`).style("stroke-width", 3).raise();
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
        .style("fill", "var(--text-title)") 
        .style("font-size", "0.65rem")
        .style("font-weight", "bold");
}