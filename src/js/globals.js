// --- GLOBAL STATE & CONFIGURATION ---

// Environment Configuration
const USE_LOCAL_API = true; // Set to true to switch to local backend
const PUBLIC_API_URL = 'https://matteotwentywings.pythonanywhere.com';
const LOCAL_API_URL = 'http://127.0.0.1:5000';

// Core data and state variables
let dataset = [];
let metadata = {};
let colorMode = 'original';
let uniqueClasses = [];
let pointById = new Map();
let currentPointSize = 3.5;
let selectedPoint = null;
let brushedPointsGlobal = [];
let kmeansProjectionSource = 'pca';
let currentDatasetName = "wine";

// Active filters & parameters
let minPrecision = 0;
let minRecall = 0;
let fpMethod = 'weighted'; // Selected False Positive Method globally scoped

// Server-side calculated pseudo-centroids cache
let pseudoCentroidsPCA = null;
let pseudoCentroidsMDS = null;

// Brush state
let savedBrushExtent = null;
let savedBrushSource = null;

// Plot scales and brushes storage
const scalesMap = { pca: {}, mds: {}, kmeans: {}, pca2d: {}, mds2d: {} };
let brushPCA, brushMDS, brushKMeans, brushPCA2D, brushMDS2D;

// Parallel Coordinates Brush State
let activePCBrushes = new Map(); // Tracks multiple active selections across different axes
let pcBrushes = new Map(); // Stores the unique D3 brush instances for each axis

// Radar Chart pre-calculated metrics
let radarDimensions = [];
let radarMinMax = {};
let origClusterAvg = {};
let pcaKmeansAvg = {};
let mdsKmeansAvg = {};

// Color Scales Configuration
const customTableau = [...d3.schemeTableau10];
customTableau[2] = '#2ca02c';
customTableau[3] = '#9b59b6';

const colorOriginal = d3.scaleOrdinal(customTableau);
const bluesDiscrete = ["#08519c", "#3182bd", "#6baed6", "#9ecae1", "#c6dbef"];
const colorPrecision = d3.scaleQuantize().domain([0, 1]).range(bluesDiscrete);
const redsDiscrete = ["#67000d", "#cb181d", "#fb6a4a", "#fcae91", "#fee5d9"];
const colorRecall = d3.scaleQuantize().domain([0, 1]).range(redsDiscrete);
const rdYlGnDiscrete = ["#d73027", "#fdae61", "#ffffbf", "#a6d96a", "#1a9641"];
const colorFScore = d3.scaleQuantize().domain([0, 1]).range(rdYlGnDiscrete);

// Gauges Configuration
const gauges = { precision: { foreground: null }, recall: { foreground: null }, fscore: { foreground: null } };
const gaugeAngleScale = d3.scaleLinear().domain([0, 1]).range([-Math.PI / 2, Math.PI / 2]);

// --- UTILITIES ---

// Clears all Parallel Coordinates brushes safely and resyncs the tracking maps
function clearPCBrushes() {
    activePCBrushes.clear();
    d3.selectAll(".pc-brush").each(function(f) {
        const b = pcBrushes.get(f);
        if (b) d3.select(this).call(b.move, null);
    });
}

// Returns the stroke color based on whether anomalies are enabled and the active color mode
function getLineColor(d) {
    const showAnon = d3.select("#show-anomalies").property("checked");
    if (colorMode === 'original') return (showAnon && d.is_anomaly) ? 'var(--anomaly-color)' : colorOriginal(d.label);
    return getColor(d);
}

// Custom D3 shape generator for marking False Positives with an 'X'
const symbolX = {
    draw: function(context, size) {
        const r = Math.sqrt(size) * 0.6; // Scale factor mapping to traditional symbol areas
        const w = r * 0.25; // Line thickness
        
        // Define coordinates for a simple cross centered at 0,0
        const pts = [
            [-w, -r], [w, -r], [w, -w], [r, -w], [r, w], [w, w], 
            [w, r], [-w, r], [-w, w], [-r, w], [-r, -w], [-w, -w]
        ];
        
        // Rotate points by 45 degrees to form an X
        const rotated = pts.map(([x, y]) => [
            (x - y) * 0.707, 
            (x + y) * 0.707
        ]);
        
        context.moveTo(rotated[0][0], rotated[0][1]);
        for (let i = 1; i < rotated.length; i++) {
            context.lineTo(rotated[i][0], rotated[i][1]);
        }
        context.closePath();
    }
};

// Generates the SVG path string for a point, handles anomaly triangles, hover scaling, and FP Overlays
function getSymbolPath(plotClass, d, isHovered = false, overrideR = null) {
    const showAnon = d3.select("#show-anomalies").property("checked");
    const hideFpGlobal = d3.select("#hide-fp-global").property("checked");
    const showFpOverlay = d3.select("#show-fp-overlay").property("checked");

    let isAnom = false;
    let isFp = false;
    
    // Evaluate properties based on the specific plot context
    if (plotClass && plotClass.includes("pca2d")) { isAnom = d.pca_is_anomaly; isFp = d.is_fp_pca; }
    else if (plotClass && plotClass.includes("mds2d")) { isAnom = d.mds_is_anomaly; isFp = d.is_fp_mds; }
    else if (plotClass && plotClass.includes("kmeans")) { 
        isAnom = d.is_anomaly; 
        isFp = (kmeansProjectionSource === 'pca' ? d.is_fp_pca : d.is_fp_mds); 
    }
    else if (plotClass && plotClass.includes("pca")) isFp = d.is_fp_pca;
    else if (plotClass && plotClass.includes("mds")) isFp = d.is_fp_mds;
    
    let r = currentPointSize;
    if (overrideR !== null) r = overrideR;
    else if (isHovered) r = currentPointSize * 2.0;
    
    const area = Math.PI * Math.pow(r, 2) * ((showAnon && isAnom) ? 1.5 : 1);
    
    let type = d3.symbolCircle;
    
    // Check specific conditions for overlay: exclude K-means scatter and 2D charts entirely from visual alteration
    const isTab2Plot = plotClass && (plotClass.includes("pca2d") || plotClass.includes("mds2d"));
    const isKmeansPlot = plotClass && plotClass.includes("kmeans");

    if (hideFpGlobal && showFpOverlay && isFp && !isKmeansPlot && !isTab2Plot) {
        type = symbolX;
    } else if (showAnon && isAnom) {
        type = d3.symbolTriangle;
    }
    
    return d3.symbol().type(type).size(area)();
}

// Maps data metrics to specific color scales based on the current UI mode
function getColor(d) {
    if (colorMode === 'original') return colorOriginal(d.label);
    if (colorMode === 'precision') return colorPrecision(d.precision);
    if (colorMode === 'recall') return colorRecall(d.recall);
    if (colorMode === 'fscore') return colorFScore(d.f_score);
}

// Shows fundamental metrics (ID, Class, Prec, Recall, F-Score)
function showTooltip(event, d) {
    const tooltip = d3.select("#tooltip");
    
    tooltip.html(`
        <strong style="color: var(--control-sel-text);">ID:</strong> ${d.id} | <strong style="color: var(--control-sel-text);">Class:</strong> ${d.label}<br>
        <strong style="color: var(--control-sel-text);">Precision:</strong> ${d.precision === 1 ? "100" : (d.precision*100).toFixed(1)}%<br>
        <strong style="color: var(--control-sel-text);">Recall:</strong> ${d.recall === 1 ? "100" : (d.recall*100).toFixed(1)}%<br>
        <strong style="color: var(--control-sel-text);">F-Score:</strong> ${d.f_score === 1 ? "100" : (d.f_score*100).toFixed(1)}%
    `);

    const tooltipNode = tooltip.node();
    const tooltipWidth = tooltipNode.offsetWidth;
    const tooltipHeight = tooltipNode.offsetHeight;
    const margin = 20;

    let x = event.pageX + margin;
    if (x + tooltipWidth > window.innerWidth) x = event.pageX - tooltipWidth - margin;

    let y = event.pageY + margin;
    if (y + tooltipHeight > window.innerHeight) y = event.pageY - tooltipHeight - margin;

    tooltip.style("left", x + "px")
           .style("top", y + "px")
           .transition().duration(100).style("opacity", 1);
}

// Hides the tooltip smoothly
function hideTooltip() {
    d3.select("#tooltip").transition().duration(200).style("opacity", 0);
}

// Displays a detailed attribute tooltip or defaults to standard metrics if attributes are missing
function showAttributeTooltip(event, d) {
    const tooltip = d3.select("#tooltip");
    if (!d.attributes) return showTooltip(event, d);

    const formatKey = (key) => key.charAt(0).toUpperCase() + key.slice(1).replace(/_/g, ' ');

    const attributesHtml = radarDimensions
        .map(key => `<strong><span style="color: var(--control-sel-text);">${formatKey(key)}:</span></strong> <span>${d.attributes[key]}</span>`).join('<br>');

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

// Adjusts the width of the dataset selector to fit the currently selected text
function updateDatasetSelectWidth() {
    const select = document.getElementById("dataset-selector");
    if (!select) return;
    const temp = document.createElement("span");
    temp.style.font = window.getComputedStyle(select).font;
    temp.style.visibility = "hidden";
    temp.style.whiteSpace = "pre";
    temp.style.position = "absolute";
    temp.textContent = select.options[select.selectedIndex].text;
    document.body.appendChild(temp);
    select.style.width = (temp.offsetWidth + 42) + "px";
    document.body.removeChild(temp);
}

// --- MASTER VISUAL STATE CONTROLLER ---
// Dynamically evaluates and applies Fill, Stroke, and Opacity to all points
function resetAllHovers() {
    const showAnon = d3.select("#show-anomalies").property("checked");
    const hideFpGlobal = d3.select("#hide-fp-global").property("checked");
    const showFpOverlay = d3.select("#show-fp-overlay").property("checked");
    const showCentroids = d3.select("#show-discrepancies").property("checked");
    const showPseudoCentroids = d3.select("#show-pseudo-centroids").property("checked") && hideFpGlobal && fpMethod === 'centroids';
    const currentTab = d3.select("input[name='mainTab']:checked").node().value;

    d3.selectAll(".dot").each(function(p) {
        const self = d3.select(this);
        const plotClass = self.attr("class");
        
        let isAnom = false;
        let isFp = false;
        
        if (plotClass && plotClass.includes("pca2d")) { isAnom = p.pca_is_anomaly; isFp = p.is_fp_pca; }
        else if (plotClass && plotClass.includes("mds2d")) { isAnom = p.mds_is_anomaly; isFp = p.is_fp_mds; }
        else if (plotClass && plotClass.includes("kmeans")) { 
            isAnom = p.is_anomaly; 
            isFp = (kmeansProjectionSource === 'pca' ? p.is_fp_pca : p.is_fp_mds);
        }
        else if (plotClass && plotClass.includes("pca")) isFp = p.is_fp_pca;
        else if (plotClass && plotClass.includes("mds")) isFp = p.is_fp_mds;

        // Skip 'X' symbol rendering on K-Means and Tab 2 charts entirely
        const isTab2Plot = plotClass && (plotClass.includes("pca2d") || plotClass.includes("mds2d"));
        const isKmeansPlot = plotClass && plotClass.includes("kmeans");
        const isTab1Plot = plotClass && (plotClass.includes("pca") || plotClass.includes("mds")) && !isTab2Plot && !isKmeansPlot;
        const isFpOverlayTarget = hideFpGlobal && showFpOverlay && isFp && !isKmeansPlot && !isTab2Plot;

        let isHighlighted = false;
        let targetOpacity = 0.9;
        let isSelected = false;

        // Determine Selection/Highlight Context
        if (selectedPoint) {
            const activeIds = new Set([selectedPoint.id, ...(selectedPoint.neighbors || [])]);
            isSelected = p.id === selectedPoint.id;
            if (activeIds.has(p.id)) isHighlighted = true;
            
            if (isSelected) targetOpacity = 1;
            else if (currentTab === 'nd' && isHighlighted) targetOpacity = 0.8;
            else targetOpacity = 0.1;
        } 
        else if (brushedPointsGlobal && brushedPointsGlobal.length > 0) {
            if (brushedPointsGlobal.includes(p)) {
                isHighlighted = true;
                targetOpacity = 0.9;
            } else {
                targetOpacity = 0.15;
            }
        } 
        else {
            if (showCentroids) {
                if (plotClass && (plotClass.includes("kmeans") || plotClass.includes("pca2d") || plotClass.includes("mds2d"))) {
                    if (isAnom) {
                        isHighlighted = true;
                        targetOpacity = 1.0;
                    } else {
                        targetOpacity = 0.2;
                    }
                } else {
                    targetOpacity = 0.9;
                }
            } else if (showPseudoCentroids && isTab1Plot) {
                // Focus pseudo-centroid FPs: diminish non-FPs in Tab1
                if (isFp) {
                    isHighlighted = true;
                    targetOpacity = 1.0;
                } else {
                    targetOpacity = 0.2;
                }
            } else {
                targetOpacity = 0.9;
            }
        }

        // Calculate Base Colors based on interactions and overlay states
        let baseColor = getColor(p);
        if (colorMode === 'original' && showAnon && isAnom) baseColor = 'var(--anomaly-color)';
        if (isFpOverlayTarget) baseColor = 'var(--fp-x-color)';

        let sColor = isSelected ? "var(--hover-stroke)" : "var(--dot-stroke)";
        if (isFpOverlayTarget) sColor = 'var(--fp-x-color)';

        // Apply visual properties
        self.attr("d", getSymbolPath(plotClass, p, isSelected))
            .style("fill", baseColor)
            .style("stroke", sColor)
            .style("stroke-width", isSelected ? 2 : 0.8)
            .style("opacity", targetOpacity)
            .style("pointer-events", targetOpacity > 0 ? "auto" : "none"); 
    });

    d3.selectAll(".pc-line").each(function(p) {
        const self = d3.select(this);
        
        let targetOpacity = 0.6;
        let sw = (showAnon && p.is_anomaly && colorMode === 'original') ? 2.5 : 1.5;
        let isHighlighted = false;

        if (selectedPoint) {
            if (p.id === selectedPoint.id) { targetOpacity = 1; sw = 3; isHighlighted = true; }
            else { targetOpacity = 0.05; }
        } else if (brushedPointsGlobal && brushedPointsGlobal.length > 0) {
            if (brushedPointsGlobal.includes(p)) { targetOpacity = 0.9; isHighlighted = true; }
            else { targetOpacity = 0.05; }
        } else {
            targetOpacity = (showAnon && p.is_anomaly && colorMode === 'original') ? 0.9 : 0.6;
        }

        self.style("stroke", getLineColor(p))
            .style("stroke-width", sw)
            .style("opacity", targetOpacity)
            .style("pointer-events", targetOpacity > 0 ? "visibleStroke" : "none"); 
    });

    hideTooltip();
}