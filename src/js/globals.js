// --- GLOBAL STATE & CONFIGURATION ---

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

// Active filters
let minPrecision = 0;
let minRecall = 0;

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

// Generates the SVG path string for a point, handles anomaly triangles and hover scaling
function getSymbolPath(plotClass, d, isHovered = false, overrideR = null) {
    const showAnon = d3.select("#show-anomalies").property("checked");
    let isAnom = false;
    
    if (plotClass && plotClass.includes("pca2d")) isAnom = d.pca_is_anomaly;
    else if (plotClass && plotClass.includes("mds2d")) isAnom = d.mds_is_anomaly;
    else if (plotClass && plotClass.includes("kmeans")) isAnom = d.is_anomaly;
    
    let r = currentPointSize;
    if (overrideR !== null) r = overrideR;
    else if (isHovered) r = currentPointSize * 2.0;
    
    const area = Math.PI * Math.pow(r, 2) * ((showAnon && isAnom) ? 1.5 : 1);
    const type = (showAnon && isAnom) ? d3.symbolTriangle : d3.symbolCircle;
    
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
    const showCentroids = d3.select("#show-discrepancies").property("checked");
    const currentTab = d3.select("input[name='mainTab']:checked").node().value;

    d3.selectAll(".dot").each(function(p) {
        const self = d3.select(this);
        const isFilteredFP = self.classed("filtered-fp");
        const plotClass = self.attr("class");
        
        let isAnom = false;
        if (plotClass && plotClass.includes("pca2d")) isAnom = p.pca_is_anomaly;
        else if (plotClass && plotClass.includes("mds2d")) isAnom = p.mds_is_anomaly;
        else if (plotClass && plotClass.includes("kmeans")) isAnom = p.is_anomaly;

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
            } else {
                targetOpacity = 0.9;
            }
        }

        // Calculate Base Colors
        let baseColor = getColor(p);
        if (colorMode === 'original' && showAnon && isAnom) baseColor = 'var(--anomaly-color)';

        // --- GHOST POINT LOGIC --- 
        const finalFill = (isFilteredFP && isHighlighted) ? "transparent" : baseColor;
        const finalStroke = isSelected ? "var(--hover-stroke)" : (isFilteredFP && isHighlighted ? baseColor : "var(--dot-stroke)");
        const finalStrokeWidth = isSelected ? 2 : (isFilteredFP && isHighlighted ? 1.5 : 0.8);
        const finalOpacity = isFilteredFP ? (isHighlighted ? targetOpacity : 0) : targetOpacity;

        self.attr("d", getSymbolPath(plotClass, p, isSelected))
            .style("fill", finalFill)
            .style("stroke", finalStroke)
            .style("stroke-width", finalStrokeWidth)
            .style("opacity", finalOpacity);
    });

    d3.selectAll(".pc-line").each(function(p) {
        const self = d3.select(this);
        const isFilteredFP = self.classed("filtered-fp");
        
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

        const finalOpacity = isFilteredFP ? (isHighlighted ? targetOpacity : 0) : targetOpacity;

        self.style("stroke", getLineColor(p))
            .style("stroke-width", sw)
            .style("opacity", finalOpacity);
    });

    hideTooltip();
}