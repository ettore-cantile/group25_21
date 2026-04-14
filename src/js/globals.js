// --- GLOBAL STATE ---

// Main dataset array
let dataset = [];
// Metadata object for global metrics
let metadata = {};
// Current color mode for visualizations
let colorMode = 'original';
// Array of unique classes in the dataset
let uniqueClasses = [];
// Map to quickly access points by their ID
let pointById = new Map();
// Default visual size for data points
let currentPointSize = 3.5;
// Currently selected data point
let selectedPoint = null;
// Array of points selected via brush interactions
let brushedPointsGlobal = []; 
// Source projection for KMeans ('pca' or 'mds')
let kmeansProjectionSource = 'pca'; 
// Name of the currently active dataset
let currentDatasetName = "wine";

// Filters
// Minimum precision value filter
let minPrecision = 0;
// Minimum recall value filter
let minRecall = 0;

// Saved boundaries of the active brush
let savedBrushExtent = null;
// Identifier for the chart where the brush is active
let savedBrushSource = null;

// Scales Storage
// Map storing D3 scales for X and Y axes across all plots
const scalesMap = { pca: {}, mds: {}, kmeans: {}, pca2d: {}, mds2d: {} };

// D3 Brushes
// D3 brush instances for each scatter plot
let brushPCA, brushMDS, brushKMeans, brushPCA2D, brushMDS2D;

// Radar Chart Data
// Keys for the attributes used in the radar chart
let radarDimensions = [];
// Minimum and maximum values for each radar dimension
let radarMinMax = {};
// Average attribute values for the original ground-truth clusters
let origClusterAvg = {};
// Average attribute values for PCA KMeans clusters
let pcaKmeansAvg = {}; 
// Average attribute values for MDS KMeans clusters
let mdsKmeansAvg = {};

// Color Scales
// Custom Tableau 10 color scheme array
const customTableau = [...d3.schemeTableau10];
customTableau[2] = '#2ca02c'; 
customTableau[3] = '#9b59b6'; 

// Ordinal color scale for ground-truth classes
const colorOriginal = d3.scaleOrdinal(customTableau);
// Discrete color palette for precision (blues)
const bluesDiscrete = ["#08519c", "#3182bd", "#6baed6", "#9ecae1", "#c6dbef"];
// Quantize scale mapping precision values to blue colors
const colorPrecision = d3.scaleQuantize().domain([0, 1]).range(bluesDiscrete); 
// Discrete color palette for recall (reds)
const redsDiscrete = ["#67000d", "#cb181d", "#fb6a4a", "#fcae91", "#fee5d9"];
// Quantize scale mapping recall values to red colors
const colorRecall = d3.scaleQuantize().domain([0, 1]).range(redsDiscrete);
// Discrete color palette for F-Score (red to green)
const rdYlGnDiscrete = ["#d73027", "#fdae61", "#ffffbf", "#a6d96a", "#1a9641"];
// Quantize scale mapping F-Score values to red-yellow-green colors
const colorFScore = d3.scaleQuantize().domain([0, 1]).range(rdYlGnDiscrete);

// Gauges
// Configuration objects storing foreground elements for gauges
const gauges = { precision: { foreground: null }, recall: { foreground: null }, fscore: { foreground: null } };
// Linear scale to map metric values (0-1) to gauge angles
const gaugeAngleScale = d3.scaleLinear().domain([0, 1]).range([-Math.PI / 2, Math.PI / 2]);


// --- UTILITIES ---

// Function to determine the stroke color for lines based on the active color mode
function getLineColor(d) {
    const showAnon = d3.select("#show-anomalies").property("checked");
    if (colorMode === 'original') return (showAnon && d.is_anomaly) ? 'var(--anomaly-color)' : colorOriginal(d.label);
    return getColor(d);
}

// Function to generate the SVG path for data points (circle or triangle for anomalies)
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

// Function to determine the fill color of a point based on the active color mode
function getColor(d) {
    if (colorMode === 'original') return colorOriginal(d.label);
    if (colorMode === 'precision') return colorPrecision(d.precision);
    if (colorMode === 'recall') return colorRecall(d.recall);
    if (colorMode === 'fscore') return colorFScore(d.f_score);
}

// Function to display the standard tooltip with precision, recall, and F-Score data
function showTooltip(event, d) {
    const tooltip = d3.select("#tooltip");
    
    let fpText = d.precision < 0.9 
        ? `🔴 <span style="color: var(--control-sel-text); font-style: italic; font-weight: bold;">False Positive:</span> Attracts <span style="font-weight: bold;">${((1 - d.precision)*100).toFixed(1)}%</span> of points from other classes.` 
        : `🟢 <span style="color: var(--control-sel-text); font-style: italic; font-weight: bold;">Low FPs:</span> No class mixing.`;
        
    let fnText = d.recall < 0.9 
        ? `🔴 <span style="color: var(--control-sel-text); font-style: italic; font-weight: bold;">False Negative:</span> Disconnected from <span style="font-weight: bold;">${((1 - d.recall)*100).toFixed(1)}%</span> of points in its own class.` 
        : `🟢 <span style="color: var(--control-sel-text); font-style: italic; font-weight: bold;">Low FNs:</span> Highly cohesive.`;

    let anomalyText = d.is_anomaly ? `<span style='color: var(--anomaly-color);'>⚠️ <strong>Anomaly:</strong> True P${d.label} assigned to KMeans C${d.kmeans_cluster}</span><br>` : "";

    tooltip.html(`
        <strong style="color: var(--control-sel-text);">ID:</strong> ${d.id} | <strong style="color: var(--control-sel-text);">Class:</strong> ${d.label}<br>
        <strong style="color: var(--control-sel-text);">Precision:</strong> ${d.precision === 1 ? "100" : (d.precision*100).toFixed(1)}%<br>
        <strong style="color: var(--control-sel-text);">Recall:</strong> ${d.recall === 1 ? "100" : (d.recall*100).toFixed(1)}%<br>
        <strong style="color: var(--control-sel-text);">F-Score:</strong> ${d.f_score === 1 ? "100" : (d.f_score*100).toFixed(1)}%
        <hr style="border: 0; border-top: 1px solid var(--border-color); margin: 8px 0 6px 0;">
        <div style="color: var(--text-main); line-height: 1.4;">
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

// Function to hide the standard tooltip smoothly
function hideTooltip() {
    d3.select("#tooltip").transition().duration(200).style("opacity", 0);
}

// Function to display an extended tooltip containing radar chart attributes
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

// Function to adjust the width of the dataset selector dropdown dynamically
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

// Function to reset all hover states across visualizations
function resetAllHovers() {
    const showAnon = d3.select("#show-anomalies").property("checked");

    if (selectedPoint) {
        const currentTab = d3.select("input[name='mainTab']:checked").node().value;
        const neighborIds = selectedPoint.neighbors || [];
        const activeIds = new Set([selectedPoint.id, ...neighborIds]);

        d3.selectAll(".dot")
            .attr("d", function(p) { return getSymbolPath(d3.select(this).attr("class"), p, p.id === selectedPoint.id); })
            .style("stroke", p => p.id === selectedPoint.id ? "var(--hover-stroke)" : "var(--dot-stroke)")
            .style("stroke-width", p => p.id === selectedPoint.id ? 2 : 0.8)
            .style("opacity", p => {
                if (p.id === selectedPoint.id) return 1;
                if (currentTab === 'nd' && activeIds.has(p.id)) return 0.8;
                return 0.1;
            });

        // PC Line explicitly highlights ONLY the selected point, completely ignoring activeIds (neighbors)
        d3.selectAll(".pc-line")
            .style("stroke-width", p => p.id === selectedPoint.id ? 3 : 1.5)
            .style("stroke", d => getLineColor(d))
            .style("opacity", p => {
                if (p.id === selectedPoint.id) return 1;
                return 0.05; // Only selected point is visible
            });

    } else if (brushedPointsGlobal && brushedPointsGlobal.length > 0) {
        d3.selectAll(".dot")
            .attr("d", function(p) { return getSymbolPath(d3.select(this).attr("class"), p, false); })
            .style("stroke", "var(--dot-stroke)")
            .style("stroke-width", 0.8)
            .style("opacity", p => brushedPointsGlobal.includes(p) ? 0.9 : 0.15);

        d3.selectAll(".pc-line")
            .style("stroke", d => getLineColor(d))
            .style("opacity", p => brushedPointsGlobal.includes(p) ? 0.9 : 0.05);
    } else {
        const showCentroids = d3.select("#show-discrepancies").property("checked");

        d3.selectAll(".dot")
            .attr("d", function(p) { return getSymbolPath(d3.select(this).attr("class"), p, false); })
            .style("stroke", "var(--dot-stroke)")
            .style("stroke-width", 0.8)
            .style("opacity", function(p) {
                const plotClass = d3.select(this).attr("class");
                if (showCentroids) {
                    if (plotClass && plotClass.includes("kmeans")) return p.is_anomaly ? 1.0 : 0.2;
                    if (plotClass && plotClass.includes("pca2d")) return p.pca_is_anomaly ? 1.0 : 0.2;
                    if (plotClass && plotClass.includes("mds2d")) return p.mds_is_anomaly ? 1.0 : 0.2;
                }
                return 0.9;
            });

        d3.selectAll(".pc-line")
            .style("stroke", d => getLineColor(d))
            .style("stroke-width", p => (showAnon && p.is_anomaly && colorMode === 'original') ? 2.5 : 1.5)
            .style("opacity", p => (showAnon && p.is_anomaly && colorMode === 'original') ? 0.9 : 0.6);
    }
    hideTooltip();
}