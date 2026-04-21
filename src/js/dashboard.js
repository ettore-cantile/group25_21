// --- MAIN LOGIC & EVENT CONTROLLERS ---

// Initial Core Bootstrapper logic mapped against JSON payloads
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
    
    activePCBrushes.clear();
    pcBrushes.clear();
    
    brushPCA = d3.brush();
    brushMDS = d3.brush();
    brushKMeans = d3.brush();
    brushPCA2D = d3.brush();
    brushMDS2D = d3.brush();

    const currentTab = d3.select("input[name='mainTab']:checked").node().value;
    
    if (currentTab === 'nd') {
        d3.select("#empty-state-placeholder").classed("hidden-panel", false);
        d3.select("#radar-empty-state").classed("hidden-panel", true);
    } else {
        d3.select("#empty-state-placeholder").classed("hidden-panel", true);
        d3.select("#radar-empty-state").classed("hidden-panel", false);
    }

    d3.select("#gauges-container").classed("hidden-panel", true);
    d3.select("#confusion-matrix-container").classed("hidden-panel", true);
    d3.select("#radar-chart-container").classed("hidden-panel", true);
    d3.select("#neighbor-graph-container").classed("hidden-panel", true);
    d3.select("#dynamic-panel-title").text("Live Analytics");

    const basePath = `../json/${folder}/`;

    Promise.all([
        d3.json(`${basePath}step2_final_data.json?v=${Date.now()}`),
        d3.csv(`../../dataset/${folder}.csv`),
        d3.json(`${basePath}kmeans_results.json?v=${Date.now()}`),
        d3.json(`${basePath}kmeans_2d_results.json?v=${Date.now()}`),
        d3.json(`${basePath}step_fp_results.json?v=${Date.now()}`).catch(() => null)
    ]).then(([data, csvData, kmeansData, kmeans2dData, fpData]) => {
        
        if (csvData && csvData.length > 0) {
            const allKeys = Object.keys(csvData[0]);
            radarDimensions = allKeys.filter(k => !['producer', 'label', 'class', 'species', 'variety', 'unnamed', 'uns'].some(sub => k.toLowerCase().includes(sub)));
        }
        
        d3.select("label[for='tab-nd']").text(`MDS vs PCA ${radarDimensions.length}D`);

        const kmeansMap = new Map();
        if(kmeansData?.points) kmeansData.points.forEach(p => kmeansMap.set(p.id, p));

        const kmeans2dMap = new Map();
        if(kmeans2dData?.points) kmeans2dData.points.forEach(p => kmeans2dMap.set(p.id, p));

        // Format IDs to securely match across sources
        const fpPcaSet = new Set((fpData?.false_positive_points_pca || []).map(String));
        const fpMdsSet = new Set((fpData?.false_positive_points_mds || []).map(String));

        const formatLabel = (str) => {
            if (!str || str === "undefined") return "Unknown";
            return String(str).replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
        };

        data.points.forEach((p, i) => {
            if (csvData[i]) p.attributes = csvData[i];

            p.label = formatLabel(p.label);
            
            p.is_fp_pca = fpPcaSet.has(String(p.id));
            p.is_fp_mds = fpMdsSet.has(String(p.id));
            
            const kData = kmeansMap.get(p.id);
            if(kData) {
                p.kmeans_cluster = formatLabel(kData.kmeans_cluster);
                p.is_anomaly = kData.is_anomaly; 
            }
            
            const k2dData = kmeans2dMap.get(p.id);
            if(k2dData) {
                p.pca_kmeans_cluster = formatLabel(k2dData.pca_kmeans_cluster);
                p.pca_is_anomaly = k2dData.pca_is_anomaly;
                p.mds_kmeans_cluster = formatLabel(k2dData.mds_kmeans_cluster);
                p.mds_is_anomaly = k2dData.mds_is_anomaly;
            }
        });

        dataset = data.points;
        metadata = data.metadata;

        uniqueClasses = Array.from(new Set(dataset.map(d => String(d.label)))).sort();
        pointById = new Map(dataset.map(p => [p.id, p]));

        colorOriginal.domain(uniqueClasses);

        radarMinMax = {};
        origClusterAvg = {};
        pcaKmeansAvg = {};
        mdsKmeansAvg = {};

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

        const pcaClasses = Array.from(new Set(dataset.map(d => String(d.pca_kmeans_cluster)).filter(c => c !== "undefined")));
        pcaClasses.forEach(c => {
            let pts = dataset.filter(d => String(d.pca_kmeans_cluster) === c);
            pcaKmeansAvg[c] = {};
            radarDimensions.forEach(dim => pcaKmeansAvg[c][dim] = d3.mean(pts, d => +d.attributes[dim]));
        });

        const mdsClasses = Array.from(new Set(dataset.map(d => String(d.mds_kmeans_cluster)).filter(c => c !== "undefined")));
        mdsClasses.forEach(c => {
            let pts = dataset.filter(d => String(d.mds_kmeans_cluster) === c);
            mdsKmeansAvg[c] = {};
            radarDimensions.forEach(dim => mdsKmeansAvg[c][dim] = d3.mean(pts, d => +d.attributes[dim]));
        });

        if(metadata) {
            if (metadata.global_assessment && metadata.global_assessment.pca && metadata.global_assessment.mds) {
                d3.select("#fb-pca-trust").text((metadata.global_assessment.pca.trustworthiness * 100).toFixed(1) + "%");
                d3.select("#fb-pca-cont").text((metadata.global_assessment.pca.continuity * 100).toFixed(1) + "%");
                d3.select("#fb-mds-trust").text((metadata.global_assessment.mds.trustworthiness * 100).toFixed(1) + "%");
                d3.select("#fb-mds-cont").text((metadata.global_assessment.mds.continuity * 100).toFixed(1) + "%");
                
                let pcaStress = metadata.global_assessment.pca.stress;
                let mdsStress = metadata.global_assessment.mds.stress;
                d3.select("#fb-pca-stress").text(pcaStress !== undefined ? (pcaStress * 100).toFixed(1) + "%" : "N/A");
                d3.select("#fb-mds-stress").text(mdsStress !== undefined ? (mdsStress * 100).toFixed(1) + "%" : "N/A");
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

        recomputeFP();

    }).catch(err => {
        console.error(`Error loading dataset [${folder}]:`, err);
        alert(`Failed to load data for dataset: ${folder}.`);
    });
}

// Function to dynamically call the Public/Local Python microservice to compute False Positives
async function recomputeFP() {
    const baseUrl = USE_LOCAL_API ? LOCAL_API_URL : PUBLIC_API_URL;
    let endpoint = '';
    let payload = { dataset: currentDatasetName };

    if (fpMethod === 'weighted') {
        endpoint = '/api/compute_fp_weighted';
        payload.k = parseInt(d3.select("#fp-k-weighted").node().value) || 15;
        payload.threshold = parseFloat(d3.select("#fp-thresh-weighted").node().value) || 0.5;
    } else if (fpMethod === 'mismatch') {
        endpoint = '/api/compute_fp_mismatch';
        payload.k = parseInt(d3.select("#fp-k-mismatch").node().value) || 15;
        payload.threshold = parseFloat(d3.select("#fp-thresh-mismatch").node().value) || 0.5;
    } else if (fpMethod === 'stress') {
        endpoint = '/api/compute_fp_stress';
        payload.threshold = parseFloat(d3.select("#fp-thresh-stress").node().value) || 0.1;
    } else if (fpMethod === 'centroids') {
        endpoint = '/api/compute_fp_centroids';
        payload.threshold = parseFloat(d3.select("#fp-thresh-centroids").node().value) || 0.1;
    }

    d3.select("#global-loader").classed("hidden-panel", false);

    try {
        const response = await fetch(`${baseUrl}${endpoint}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        
        if (!response.ok) throw new Error("API request failed");
        const data = await response.json();
        
        const fpPcaSet = new Set((data.false_positive_points_pca || []).map(String));
        const fpMdsSet = new Set((data.false_positive_points_mds || []).map(String));
        
        // Cache pseudo centroids coordinates if applicable 
        pseudoCentroidsPCA = data.pseudo_centroids_2d_pca || null;
        pseudoCentroidsMDS = data.pseudo_centroids_2d_mds || null;
        
        dataset.forEach(d => {
            d.is_fp_pca = fpPcaSet.has(String(d.id));
            d.is_fp_mds = fpMdsSet.has(String(d.id));
        });
        
        // Update filtered global stress metrics
        if (data.stress_pca !== undefined) d3.select("#fp-stress-pca").text((data.stress_pca * 100).toFixed(1) + "%");
        if (data.stress_mds !== undefined) d3.select("#fp-stress-mds").text((data.stress_mds * 100).toFixed(1) + "%");
        
        applyFilters();
        
        if (d3.select("#show-pseudo-centroids").property("checked")) {
            togglePseudoCentroids(true);
        }
        
    } catch (err) {
        console.warn("API Error:", err);
    } finally {
        d3.select("#global-loader").classed("hidden-panel", true);
    }
}

// Function to draw and erase pseudo-centroid cross markers generated by the API
function togglePseudoCentroids(show) {
    d3.selectAll(".pseudo-centroid-path").remove();

    if (show) {
        const drawPCACentroids = () => {
            if (!pseudoCentroidsPCA) return;
            const svgContainer = d3.select("#pca-plot svg g .centroid-layer");
            const scales = scalesMap["pca"];
            if (svgContainer.empty() || !scales) return;

            Object.keys(pseudoCentroidsPCA).forEach(k => {
                const coords = pseudoCentroidsPCA[k];
                svgContainer.append("path")
                    .attr("class", "pseudo-centroid-path")
                    .attr("d", d3.symbol().type(d3.symbolCross).size(150)())
                    .attr("transform", `translate(${scales.xScale(coords[0])}, ${scales.yScale(coords[1])})`)
                    .style("fill", colorOriginal(k))
                    .style("stroke", "var(--sankey-node-stroke)")
                    .style("stroke-width", 1.5)
                    .style("pointer-events", "none");
            });
        };
        
        const drawMDSCentroids = () => {
            if (!pseudoCentroidsMDS) return;
            const svgContainer = d3.select("#mds-plot svg g .centroid-layer");
            const scales = scalesMap["mds"];
            if (svgContainer.empty() || !scales) return;

            Object.keys(pseudoCentroidsMDS).forEach(k => {
                const coords = pseudoCentroidsMDS[k];
                svgContainer.append("path")
                    .attr("class", "pseudo-centroid-path")
                    .attr("d", d3.symbol().type(d3.symbolCross).size(150)())
                    .attr("transform", `translate(${scales.xScale(coords[0])}, ${scales.yScale(coords[1])})`)
                    .style("fill", colorOriginal(k))
                    .style("stroke", "var(--sankey-node-stroke)")
                    .style("stroke-width", 1.5)
                    .style("pointer-events", "none");
            });
        };

        drawPCACentroids();
        drawMDSCentroids();
    }
    
    resetAllHovers();
}

// Global Document listeners
document.addEventListener("DOMContentLoaded", () => {
    updateDatasetSelectWidth();
    initDashboard(currentDatasetName);
    initGauge("#gauge-precision", gauges.precision);
    initGauge("#gauge-recall", gauges.recall);
    initGauge("#gauge-fscore", gauges.fscore);
    enhanceColorModeSwitcher();

    // False Positives Toggle Listener
    d3.select("#hide-fp-global").on("change", function() {
        const isChecked = this.checked;
        const currentTab = d3.select("input[name='mainTab']:checked").node().value;
        
        // Auto uncheck overlays if Hide FP is deactivated
        if (!isChecked) {
            d3.select("#show-fp-overlay").property("checked", false);
            d3.select("#show-pseudo-centroids").property("checked", false);
            togglePseudoCentroids(false);
        }

        if (isChecked) {
            // Configuration menu is shown exclusively on Tab 1
            if (currentTab === 'nd') {
                d3.select("#pc-plot").classed("hidden-panel", true);
                d3.select("#fp-config-panel").classed("hidden-panel", false);
            }
            recomputeFP();
        } else {
            if (currentTab === 'nd') {
                d3.select("#pc-plot").classed("hidden-panel", false);
                d3.select("#fp-config-panel").classed("hidden-panel", true);
            }
            applyFilters();
        }
    });
    
    // False Positives Overlay Listener
    d3.select("#show-fp-overlay").on("change", function() {
        applyFilters();
    });

    // Pseudo-Centroids Overlay Listener
    d3.select("#show-pseudo-centroids").on("change", function() {
        togglePseudoCentroids(this.checked);
    });

    // Updates span displays dynamically for the slider inputs
    d3.select("#fp-thresh-stress").on("input", function() {
        d3.select("#val-fp-thresh-stress").text(parseFloat(this.value).toFixed(2));
    });
    d3.select("#fp-thresh-centroids").on("input", function() {
        d3.select("#val-fp-thresh-centroids").text(parseFloat(this.value).toFixed(2));
    });

    d3.select("#fp-method-select").on("change", function() {
        fpMethod = this.value;
        d3.selectAll(".fp-params-group").classed("hidden-panel", true);
        d3.select(`#fp-params-${fpMethod}`).classed("hidden-panel", false);
        
        // Expose Pseudo-Centroids secondary configuration if selected
        if (fpMethod === 'centroids') {
            d3.select("#pseudo-centroids-toggle-container").style("display", "flex");
        } else {
            d3.select("#pseudo-centroids-toggle-container").style("display", "none");
            d3.select("#show-pseudo-centroids").property("checked", false);
            togglePseudoCentroids(false);
        }
        
        recomputeFP();
    });

    d3.selectAll(".fp-param-input").on("change", recomputeFP);

    d3.select("#dataset-selector").on("change", function() {
        updateDatasetSelectWidth();
        currentDatasetName = this.value;
        initDashboard(currentDatasetName);
    });

    d3.select("#search-point-id").on("input", function() {
        const searchId = this.value.trim();
        
        const clearSearchSelection = () => {
            selectedPoint = null;
            brushedPointsGlobal = [];
            clearPCBrushes();
            resetAllHovers();
            updateLiveAnalytics([]); 
            if(d3.select("#show-discrepancies").property("checked")) toggleDiscrepancies(true);
        };

        if (searchId === "") {
            clearSearchSelection();
            return;
        }
        
        let targetPoint = null;
        for (let d of dataset) {
            if (String(d.id) === searchId) {
                targetPoint = d;
                break;
            }
        }

        if (targetPoint) {
            updateSelection(targetPoint);
        } else {
            clearSearchSelection();
        }
    });
    
    const themeToggle = document.getElementById('theme-toggle');
    if (themeToggle) {
        themeToggle.addEventListener('click', () => {
            const currentTheme = document.documentElement.getAttribute('data-theme') || 'light';
            const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
            document.documentElement.setAttribute('data-theme', newTheme);
        });
    }

    let resizeTimer;
    let windowWidth = window.innerWidth;
    let windowHeight = window.innerHeight;
    
    window.addEventListener('resize', () => {
        if (window.innerWidth !== windowWidth || window.innerHeight !== windowHeight) {
            windowWidth = window.innerWidth;
            windowHeight = window.innerHeight;
            clearTimeout(resizeTimer);
            resizeTimer = setTimeout(() => {
                refreshAllVisualizations();
            }, 300);
        }
    });
});

// Clears and redraws everything on resize
function refreshAllVisualizations() {
    if (!dataset || dataset.length === 0) return;

    const currentTab = d3.select("input[name='mainTab']:checked").node().value;
    const hideFpGlobal = d3.select("#hide-fp-global").property("checked");

    brushPCA = d3.brush();
    brushMDS = d3.brush();
    brushKMeans = d3.brush();
    brushPCA2D = d3.brush();
    brushMDS2D = d3.brush();

    const targetPlotsToClear = [
        "#pca-plot", "#mds-plot", "#kmeans-plot", 
        "#pca-plot-2d", "#mds-plot-2d", "#pc-plot"
    ];
    targetPlotsToClear.forEach(selector => d3.select(selector).html(""));

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

    // Reset visibility logic based on current active tab
    if (currentTab === 'nd') {
        d3.select("#app-grid").classed("mode-2d", false);
        d3.selectAll(".tab-nd").classed("hidden-panel", false);
        d3.selectAll(".tab-2d").classed("hidden-panel", true);
        
        if (hideFpGlobal) {
            d3.select("#pc-plot").classed("hidden-panel", true);
            d3.select("#fp-config-panel").classed("hidden-panel", false);
        } else {
            d3.select("#pc-plot").classed("hidden-panel", false);
            d3.select("#fp-config-panel").classed("hidden-panel", true);
        }
    } else {
        d3.select("#app-grid").classed("mode-2d", true);
        d3.selectAll(".tab-nd").classed("hidden-panel", true);
        d3.selectAll(".tab-2d").classed("hidden-panel", false);
        d3.select("#fp-config-panel").classed("hidden-panel", true); 
    }

    setupBrushing();
    applyFilters();

    if (d3.select("#show-discrepancies").property("checked")) toggleDiscrepancies(true);
    if (d3.select("#show-pseudo-centroids").property("checked")) togglePseudoCentroids(true);

    if (selectedPoint) {
        updateSelection(selectedPoint, true);
    } else if (brushedPointsGlobal.length > 0) {
        updateLiveAnalytics(brushedPointsGlobal);
        dataset.forEach(d => {
            const isSelected = brushedPointsGlobal.includes(d);
            d3.selectAll(`.pt-${d.id}:not(.filtered-out)`).style("opacity", isSelected ? 0.9 : 0.15);
        });
    } else {
        updateLiveAnalytics([]);
    }
}

// --- GLOBAL EVENT LISTENERS ---

d3.select("#point-size-slider").on("input change", function() {
    currentPointSize = +this.value;
    d3.selectAll(".dot").attr("d", function(d) {
        const plotClass = d3.select(this).attr("class");
        return getSymbolPath(plotClass, d, (selectedPoint && selectedPoint.id === d.id));
    });
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

d3.selectAll("input[name='sankeyMode']").on("change", function() {
    const currentTab = d3.select("input[name='mainTab']:checked").node().value;
    if (currentTab === '2d') {
        const activeData = brushedPointsGlobal.length > 1 
            ? brushedPointsGlobal.filter(d => d.precision >= minPrecision && d.recall >= minRecall)
            : dataset.filter(d => d.precision >= minPrecision && d.recall >= minRecall);
        drawSankeyDiagram("#comparison-plot", activeData);
    }
});

// Dynamic Tab View Switcher
d3.selectAll("input[name='mainTab']").on("change", function() {
    const selectedTab = this.value;
    const hideFpGlobal = d3.select("#hide-fp-global").property("checked");
    
    if (selectedTab === 'nd') {
        d3.select("#app-grid").classed("mode-2d", false);
        d3.selectAll(".tab-nd").classed("hidden-panel", false);
        d3.selectAll(".tab-2d").classed("hidden-panel", true);
        
        // Return configuration panels if active
        if (hideFpGlobal) {
            d3.select("#pc-plot").classed("hidden-panel", true);
            d3.select("#fp-config-panel").classed("hidden-panel", false);
        } else {
            d3.select("#pc-plot").classed("hidden-panel", false);
            d3.select("#fp-config-panel").classed("hidden-panel", true);
        }
        
    } else {
        d3.select("#app-grid").classed("mode-2d", true);
        d3.selectAll(".tab-nd").classed("hidden-panel", true);
        d3.selectAll(".tab-2d").classed("hidden-panel", false);
        d3.select("#fp-config-panel").classed("hidden-panel", true); // Guarantee hidden state
        
        // Ensure FP configuration interface is never rendered under 2D conditions
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
    if(d3.select("#show-pseudo-centroids").property("checked")) togglePseudoCentroids(true);
});

// --- FILTER & INTERACTION LOGIC ---

function redrawKMeansPlot() {
    const xKey = kmeansProjectionSource === 'pca' ? 'pca_x' : 'mds_x';
    const yKey = kmeansProjectionSource === 'pca' ? 'pca_y' : 'mds_y';

    d3.select("#kmeans-plot").html("");

    drawPlot("#kmeans-plot", xKey, yKey, "kmeans", brushKMeans, d => {
        const showAnon = d3.select("#show-anomalies").property("checked");
        if (colorMode === 'original') return (showAnon && d.is_anomaly) ? 'var(--anomaly-color)' : colorOriginal(d.label);
        return getColor(d);
    });

    if (savedBrushExtent && savedBrushSource === kmeansProjectionSource) {
        d3.select("#kmeans-plot .brush-group").call(brushKMeans.move, savedBrushExtent);
    }

    applyFilters();
    if (d3.select("#show-discrepancies").property("checked")) toggleDiscrepancies(true);
    
    if (selectedPoint) {
        updateSelection(selectedPoint, true);
    } else if (brushedPointsGlobal.length > 0) {
        updateLiveAnalytics(brushedPointsGlobal); 
    }
}

// Applies logical filters and sets CSS classes. Evaluates FP Overlays conditions and updates FP Counters.
function applyFilters() {
    const hideFpGlobal = d3.select("#hide-fp-global").property("checked");
    const showFpOverlay = d3.select("#show-fp-overlay").property("checked");
    
    // Static FP Tables logic 
    const pcaFpCount = dataset.filter(d => d.is_fp_pca && d.precision >= minPrecision && d.recall >= minRecall).length;
    const mdsFpCount = dataset.filter(d => d.is_fp_mds && d.precision >= minPrecision && d.recall >= minRecall).length;
    d3.select("#fp-count-pca").text(pcaFpCount);
    d3.select("#fp-count-mds").text(mdsFpCount);

    d3.selectAll(".dot, .pc-line").each(function(d) {
        const self = d3.select(this);
        let isGeneralFiltered = (d.precision < minPrecision || d.recall < minRecall);
        let isFpFiltered = false;

        // Apply FP hiding logic ONLY to PCA, MDS and Parallel Coordinates. Exclude K-Means and Tab 2 plots.
        if (hideFpGlobal) {
            if (self.classed("dot-pca") && d.is_fp_pca) isFpFiltered = true;
            if (self.classed("dot-mds") && d.is_fp_mds) isFpFiltered = true;
            
            // Explicitly ignore K-Means graphs and Tab 2 elements
            if (self.classed("pc-line") && (d.is_fp_pca || d.is_fp_mds)) isFpFiltered = true;
        }

        self.classed("filtered-fp", isFpFiltered);

        if (isGeneralFiltered) {
            self.classed("filtered-out", true).style("display", "none").style("pointer-events", "none");
        } else if (isFpFiltered) {
            // Evaluates overlay toggles keeping standard points visually present but flagged as errors if requested
            if (showFpOverlay && !self.classed("pc-line")) { 
                self.classed("filtered-out", false).style("display", null);
            } else {
                self.classed("filtered-out", true).style("display", "none").style("pointer-events", "none");
            }
        } else {
            self.classed("filtered-out", false).style("display", null);
        }
    });

    resetAllHovers();
        
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
    const hideFpGlobal = d3.select("#hide-fp-global").property("checked");

    if (show) {
        allPlots.forEach(selector => d3.select(`${selector} svg g .centroid-layer`).selectAll("line, path:not(.pseudo-centroid-path)").remove()); 

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
                    .style("fill", colorOriginal(k)) 
                    .style("stroke", "var(--sankey-node-stroke)") 
                    .style("stroke-width", 1.5);
            });
        });

        resetAllHovers();

        let activePoints = brushedPointsGlobal.length > 0 ? brushedPointsGlobal : dataset.filter(d => d.precision >= minPrecision && d.recall >= minRecall);
        
        if (hideFpGlobal && currentTab === 'nd') {
             activePoints = activePoints.filter(d => !d.is_fp_pca && !d.is_fp_mds);
        }

        if (!selectedPoint) updateConfusionMatrix(activePoints, currentTab);
        
    } else {
        allPlots.forEach(selector => d3.select(`${selector} svg g .centroid-layer`).selectAll("line, path:not(.pseudo-centroid-path)").remove());
        resetAllHovers();
        d3.select("#confusion-matrix-container").classed("hidden-panel", true);
        if (!selectedPoint && brushedPointsGlobal.length === 0) {
            if (currentTab === '2d') {
                d3.select("#radar-empty-state").classed("hidden-panel", false);
                d3.select("#dynamic-panel-title").text("Live Analytics");
            } else {
                d3.select("#empty-state-placeholder").classed("hidden-panel", false);
                d3.select("#dynamic-panel-title").text("Live Analytics");
            }
        }
    }
}

// --- SYNC & INTERACTION ---
function updateSelection(d, skipNeighborGraph = false) {
    selectedPoint = d;
    brushedPointsGlobal = []; 
    const currentTab = d3.select("input[name='mainTab']:checked").node().value;

    [brushPCA, brushMDS, brushKMeans, brushPCA2D, brushMDS2D].forEach(b => {
        d3.selectAll(".brush-group").call(b.move, null);
    });
    
    clearPCBrushes();

    d3.select("#empty-state-placeholder").classed("hidden-panel", true);
    d3.select("#gauges-container").classed("hidden-panel", true);

    resetAllHovers();

    const neighborIds = d.neighbors || [];
    const activeIds = new Set([d.id, ...neighborIds]);
    d3.selectAll(".pc-line:not(.filtered-out)").filter(p => p.id === d.id).raise();
    d3.selectAll(".dot:not(.filtered-out)").filter(p => activeIds.has(p.id)).raise();

    if (currentTab === 'nd') {
        d3.select("#dynamic-panel-title").text("Neighbor Graph");
        d3.select("#radar-empty-state").classed("hidden-panel", true);
        d3.select("#radar-chart-container").classed("hidden-panel", true);
        d3.select("#neighbor-graph-container").classed("hidden-panel", false);
        d3.select("#confusion-matrix-container").classed("hidden-panel", true);

        const kmeansXKey = scalesMap['kmeans'].xKey;
        const kmeansYKey = scalesMap['kmeans'].yKey;

        drawLines("pca", d, neighborIds, "pca_x", "pca_y", brushPCA);
        drawLines("mds", d, neighborIds, "mds_x", "mds_y", brushMDS);
        drawLines("kmeans", d, neighborIds, kmeansXKey, kmeansYKey, brushKMeans);

        const neighborsData = neighborIds.map(id => pointById.get(id)).filter(Boolean);
        
        if (!skipNeighborGraph) {
            drawNeighborGraph(d, neighborsData);
        }

        resetAllHovers();

    } else {
        d3.select("#neighbor-graph-container").classed("hidden-panel", true);
        d3.select("#confusion-matrix-container").classed("hidden-panel", true);
        d3.select("#radar-empty-state").classed("hidden-panel", true);
        d3.select("#radar-chart-container").classed("hidden-panel", false);
        d3.select("#dynamic-panel-title").text("Multidimensional Profile (Radar)");

        drawRadarChart(d);
        d3.selectAll(".link-group line").remove();

        const baseDatasetTab2 = dataset.filter(p => p.precision >= minPrecision && p.recall >= minRecall);
        drawSankeyDiagram("#comparison-plot", baseDatasetTab2);
        
        resetAllHovers();
    }
    
    updateLiveAnalytics([d]);
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
                
                clearPCBrushes();
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
            if (brushObj === brushKMeans) {
                savedBrushExtent = null;
                savedBrushSource = null;
            }
            resetAllHovers(); 
            updateLiveAnalytics([]); 
        }
        return;
    }

    selectedPoint = null;
    const [[x0, y0], [x1, y1]] = event.selection;

    if (brushObj === brushKMeans) {
        savedBrushExtent = [[x0, y0], [x1, y1]];
        savedBrushSource = kmeansProjectionSource;
    }

    let selectedPoints = [];
    
    dataset.forEach(d => {
        if(d.precision < minPrecision || d.recall < minRecall) return;
        
        // Skip hidden points inside brush, EXCEPT if interacting directly with K-Means or Tab 2 plots
        const hideFpGlobal = d3.select("#hide-fp-global").property("checked");
        if (hideFpGlobal && brushObj !== brushKMeans && brushObj !== brushPCA2D && brushObj !== brushMDS2D) {
            if (xKey.includes('pca') && d.is_fp_pca) return;
            if (xKey.includes('mds') && d.is_fp_mds) return;
        }
        
        const cx = brushObj.xScale(d[xKey]);
        const cy = brushObj.yScale(d[yKey]);
        const isSelected = x0 <= cx && cx <= x1 && y0 <= cy && cy <= y1;
        if (isSelected) selectedPoints.push(d);
    });
    
    brushedPointsGlobal = selectedPoints;
    resetAllHovers();
    updateLiveAnalytics(selectedPoints);
}

function updateLiveAnalytics(selectedPoints) {
    const currentTab = d3.select("input[name='mainTab']:checked").node().value;
    const isAnomalyOn = d3.select("#show-discrepancies").property("checked");
    const hideFpGlobal = d3.select("#hide-fp-global").property("checked");
    
    const baseDatasetTab1 = dataset.filter(d => d.precision >= minPrecision && d.recall >= minRecall && (!hideFpGlobal || (!d.is_fp_pca && !d.is_fp_mds)));
    const baseDatasetTab2 = dataset.filter(d => d.precision >= minPrecision && d.recall >= minRecall);

    if (currentTab === '2d') {
        d3.select("#gauges-container").classed("hidden-panel", true);
        d3.select("#neighbor-graph-container").classed("hidden-panel", true);

        if (selectedPoints.length === 1) {
            d3.select("#empty-state-placeholder").classed("hidden-panel", true);
            d3.select("#confusion-matrix-container").classed("hidden-panel", true);
            d3.select("#radar-empty-state").classed("hidden-panel", true);
            d3.select("#radar-chart-container").classed("hidden-panel", false);
            d3.select("#dynamic-panel-title").text("Multidimensional Profile (Radar)");
            drawRadarChart(selectedPoints[0]);
        } else {
            d3.select("#radar-chart-container").classed("hidden-panel", true);
            
            if (isAnomalyOn) {
                d3.select("#empty-state-placeholder").classed("hidden-panel", true);
                d3.select("#radar-empty-state").classed("hidden-panel", true);
                d3.select("#confusion-matrix-container").classed("hidden-panel", false);
                updateConfusionMatrix(selectedPoints.length > 1 ? selectedPoints : baseDatasetTab2, '2d');
            } else {
                d3.select("#confusion-matrix-container").classed("hidden-panel", true);
                d3.select("#empty-state-placeholder").classed("hidden-panel", true);
                d3.select("#radar-empty-state").classed("hidden-panel", false);
                d3.select("#dynamic-panel-title").text("Live Analytics");
            }
        }
        
        const activeData = selectedPoints.length > 1 ? selectedPoints : baseDatasetTab2;
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
            updateConfusionMatrix(baseDatasetTab1, currentTab);
        } else {
            d3.select("#confusion-matrix-container").classed("hidden-panel", true);
            d3.select("#empty-state-placeholder").classed("hidden-panel", false);
        }
        return;
    }
    
    if (!selectedPoint) {
        d3.select("#dynamic-panel-title").text("Selection Metrics");
        d3.select("#empty-state-placeholder").classed("hidden-panel", true);
        d3.select("#neighbor-graph-container").classed("hidden-panel", true);
        d3.select("#confusion-matrix-container").classed("hidden-panel", true);
        d3.select("#gauges-container").classed("hidden-panel", false);
    }

    d3.select("#stat-count").text(selectedPoints.length);
    d3.selectAll(".gauges-wrapper").style("display", "flex");
    
    const avgPrec = d3.mean(selectedPoints, d => d.precision);
    const avgRecall = d3.mean(selectedPoints, d => d.recall);
    const avgFScore = d3.mean(selectedPoints, d => d.f_score);

    updateGauge(gauges.precision, avgPrec, colorFScore(avgPrec), "#val-precision");
    updateGauge(gauges.recall, avgRecall, colorFScore(avgRecall), "#val-recall");
    updateGauge(gauges.fscore, avgFScore, colorFScore(avgFScore), "#val-fscore");
}

// --- UTILITIES ---

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

function updateColors() {
    resetAllHovers();
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