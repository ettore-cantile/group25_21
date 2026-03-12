// ==========================================
// 1. GLOBAL VARIABLES AND SETUP
// ==========================================
let globalData = [];
let globalStats = {};
let currentMode = 'error'; // Default mode: show errors (False Positives/Negatives)
let colorScaleError;
const colorScaleClass = d3.scaleOrdinal(d3.schemeCategory10);

// Dictionary to save scales (x, y) of both charts to draw lines
const chartScales = { pca: null, mds: null };

// JSON Path (Relative to HTML location in ./src/html/)
const DATA_URL = "../../dataset/firstExperiment.json";

// ==========================================
// 2. DATA LOADING (D3.js)
// ==========================================
d3.json(DATA_URL).then(function(data) {
    // JSON is divided into data.stats and data.points
    globalData = data.points;
    globalStats = data.stats;
    
    // --- POPULATE DASHBOARD STATISTICS ---
    d3.select("#stat-pca-err").text(globalStats.pca_global);
    d3.select("#stat-pca-match").text(globalStats.pca_matches);
    d3.select("#stat-mds-err").text(globalStats.mds_global);
    d3.select("#stat-mds-match").text(globalStats.mds_matches);
    
    // --- DISCRETE ERROR SCALE (ColorBrewer RdBu 7-classes) ---
    // [Dark Red, Red, Light Red, Neutral(Gray), Light Blue, Blue, Dark Blue]
    const colorBrewerRdBu7 = ["#b2182b", "#ef8a62", "#fddbc7", "#f7f7f7", "#d1e5f0", "#67a9cf", "#2166ac"];

    // Find max absolute error to center scale on 0
    const maxAbsScore = d3.max(globalData, d => Math.max(Math.abs(d.score_pca), Math.abs(d.score_mds)));
    
    // scaleQuantize automatically divides domain into 7 perfect "steps"
    colorScaleError = d3.scaleQuantize()
        .domain([-maxAbsScore, maxAbsScore]) 
        .range(colorBrewerRdBu7);

    // Dynamically draw blocks in HTML legend for discrete scale
    const legendBar = d3.select("#discrete-error-legend");
    legendBar.html(""); 
    colorBrewerRdBu7.forEach(color => {
        legendBar.append("div")
            .attr("class", "discrete-step")
            .style("background-color", color);
    });

    // --- SETUP CLASS LEGEND (Wines) ---
    const classes = [...new Set(globalData.map(d => d.class_name))].sort();
    const legendClassDiv = d3.select("#legend-class");
    classes.forEach(c => {
        const box = legendClassDiv.append("div").attr("class", "legend-box");
        box.append("div").attr("class", "color-dot").style("background", colorScaleClass(c));
        box.append("span").text(c);
    });

    // Draw charts for the first time
    drawCharts();
    
}).catch(err => {
    console.error("D3 loading error:", err);
    alert("Error loading JSON.\nDid you start the local Python server? (e.g., python -m http.server)");
});

// ==========================================
// 3. INTERFACE AND MODE MANAGEMENT
// ==========================================
function setMode(mode) {
    currentMode = mode;
    
    // Update Button UI
    d3.selectAll("button").classed("active", false);
    d3.select("#btn-" + mode).classed("active", true);
    
    // Show/Hide correct legends
    d3.select("#legend-error").style("display", mode === 'error' ? 'flex' : 'none');
    d3.select("#legend-class").style("display", mode === 'class' ? 'flex' : 'none');
    
    // Animate point color change
    d3.select("#chart-pca").selectAll("circle").transition().duration(500)
        .attr("fill", d => currentMode === 'error' ? colorScaleError(d.score_pca) : colorScaleClass(d.class_name));
        
    d3.select("#chart-mds").selectAll("circle").transition().duration(500)
        .attr("fill", d => currentMode === 'error' ? colorScaleError(d.score_mds) : colorScaleClass(d.class_name));
}

// ==========================================
// 4. DRAWING CHARTS
// ==========================================
function drawCharts() {
    drawScatter("#chart-pca", "pca", "score_pca", "container-pca");
    drawScatter("#chart-mds", "mds", "score_mds", "container-mds");
}

function drawScatter(selector, type, scoreProp, containerId) {
    const container = document.getElementById(containerId);
    const rect = container.getBoundingClientRect();
    
    // Prevent negative height bug in case of delayed CSS loading
    const w = Math.max(300, rect.width - 40);
    const h = Math.max(340, rect.height - 60); 
    
    const xProp = type + "_x";
    const yProp = type + "_y";

    // Clean previous SVG on window resize
    d3.select(selector).html("");

    const svg = d3.select(selector).append("svg")
        .attr("width", w)
        .attr("height", h)
        .style("overflow", "visible"); 
    
    // Scales (adapted to min and max data)
    const x = d3.scaleLinear().domain(d3.extent(globalData, d => d[xProp])).nice().range([30, w-20]);
    const y = d3.scaleLinear().domain(d3.extent(globalData, d => d[yProp])).nice().range([h-30, 20]);

    // Save scales globally to use them later for neighbor lines
    chartScales[type] = { x: x, y: y };

    // Draw Axes
    svg.append("g").attr("transform", `translate(0,${h-30})`).call(d3.axisBottom(x).ticks(5));
    svg.append("g").attr("transform", `translate(30,0)`).call(d3.axisLeft(y).ticks(5));

    // Create two layers to prevent lines from covering circles
    const linksLayer = svg.append("g").attr("class", "links-layer");
    const nodesLayer = svg.append("g").attr("class", "nodes-layer");

    // Draw Points (Nodes)
    nodesLayer.selectAll("circle")
        .data(globalData)
        .enter().append("circle")
        .attr("cx", d => x(d[xProp]))
        .attr("cy", d => y(d[yProp]))
        .attr("r", 6)
        .attr("fill", d => currentMode === 'error' ? colorScaleError(d[scoreProp]) : colorScaleClass(d.class_name))
        .attr("class", d => "dot-" + d.id)
        
        // --- INTERACTIONS: Brushing, Linking and Tooltip ---
        .on("mouseover", function(event, d) {
            // Highlight same point on both charts
            d3.selectAll(".dot-" + d.id).classed("hovered", true).raise();
            
            // Dim all other points
            d3.selectAll("circle:not(.dot-" + d.id + ")").classed("dimmed", true);

            // Draw lines to true neighbors in 13D space
            drawNeighborLines("pca", d);
            drawNeighborLines("mds", d);

            // Show Tooltip
            const t = d3.select("#tooltip");
            t.style("opacity", 1);
            t.html(`
                <strong style="color:yellow">${d.class_name}</strong><br>
                ID: ${d.id}<br>
                Err: ${d[scoreProp].toFixed(2)}
            `)
            .style("left", event.pageX + "px")
            .style("top", event.pageY + "px");
        })
        .on("mouseout", function() {
            // Restore point opacity
            d3.selectAll("circle").classed("hovered", false).classed("dimmed", false);
            
            // Hide Tooltip
            d3.select("#tooltip").style("opacity", 0);
            
            // Clear all drawn neighbor lines
            d3.selectAll(".links-layer").html("");
        });
}

// ==========================================
// 5. DRAWING LINKS (TRUE NEIGHBORS)
// ==========================================
function drawNeighborLines(chartType, sourceData) {
    // Select lines layer of the correct chart
    const layer = d3.select(`#chart-${chartType} .links-layer`);
    
    // Retrieve scales for that chart
    const scaleX = chartScales[chartType].x;
    const scaleY = chartScales[chartType].y;

    // Coordinates of the point hovered
    const sourceX = scaleX(sourceData[`${chartType}_x`]);
    const sourceY = scaleY(sourceData[`${chartType}_y`]);

    // Iterate over true neighbor IDs (calculated by Python in R^13)
    sourceData.neighbors.forEach(targetId => {
        // Find neighbor data
        const targetData = globalData.find(p => p.id === targetId);
        
        if(targetData) {
            // Calculate neighbor position on chart
            const targetX = scaleX(targetData[`${chartType}_x`]);
            const targetY = scaleY(targetData[`${chartType}_y`]);

            // Draw line
            layer.append("line")
                .attr("class", "link-line")
                .attr("x1", sourceX)
                .attr("y1", sourceY)
                .attr("x2", targetX)
                .attr("y2", targetY);
        }
    });
}

// Redraw charts if browser window resizes
window.addEventListener('resize', drawCharts);