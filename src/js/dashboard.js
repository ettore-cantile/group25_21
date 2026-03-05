// --- GLOBAL STATE ---
let dataset = [];
let metadata = {};
let colorMode = 'original';

// --- COLOR SCALES ---
// Original labels: standard categorical colors
const colorOriginal = d3.scaleOrdinal(d3.schemeCategory10);
// False Positives (Precision): The lower the precision, the higher the FP. Low = Dark Blue, High = Light
const colorPrecision = d3.scaleSequential(d3.interpolateBlues).domain([1, 0]); 
// False Negatives (Recall): The lower the recall, the higher the FN. Low = Dark Red, High = Light
const colorRecall = d3.scaleSequential(d3.interpolateReds).domain([1, 0]);
// F-Score: Viridis. Low = Purple/Dark, High = Yellow/Bright
const colorFScore = d3.scaleSequential(d3.interpolateViridis).domain([0, 1]);

// --- INITIALIZATION ---
d3.json("../json/step2_final_data.json").then(data => {
    dataset = data.points;
    metadata = data.metadata;

    // Update Header
    d3.select("#global-assessment").html(`
        Dataset: ${metadata.dataset} <br>
        <span style="color: #2980b9;">${metadata.global_assessment.message}</span>
    `);

    // Draw Plots
    drawPlot("#pca-plot", "pca_x", "pca_y", "pca");
    drawPlot("#mds-plot", "mds_x", "mds_y", "mds");
    
    // Add Brushing to PCA
    addBrush("#pca-plot", "pca_x", "pca_y");

    // Listeners for Radio buttons
    d3.selectAll("input[name='colorMode']").on("change", function() {
        colorMode = this.value;
        updateColors();
        updateLegend();
    });

    updateLegend();
}).catch(err => console.error("Error loading JSON:", err));


// --- PLOTTING FUNCTION ---
function drawPlot(containerSelector, xKey, yKey, plotId) {
    const container = d3.select(containerSelector);
    const width = container.node().clientWidth;
    const height = container.node().clientHeight;
    const margin = { top: 20, right: 20, bottom: 30, left: 40 };

    const svg = container.append("svg")
        .attr("width", width)
        .attr("height", height)
        .append("g")
        .attr("transform", `translate(${margin.left},${margin.top})`);

    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;

    // Scales
    const xScale = d3.scaleLinear()
        .domain(d3.extent(dataset, d => d[xKey])).nice()
        .range([0, innerWidth]);

    const yScale = d3.scaleLinear()
        .domain(d3.extent(dataset, d => d[yKey])).nice()
        .range([innerHeight, 0]);

    // Axes
    svg.append("g").attr("transform", `translate(0,${innerHeight})`).call(d3.axisBottom(xScale));
    svg.append("g").call(d3.axisLeft(yScale));

    // Points
    svg.selectAll(".dot")
        .data(dataset)
        .enter().append("circle")
        .attr("class", `dot dot-${plotId}`)
        .attr("id", d => `dot-${d.id}`)
        .attr("cx", d => xScale(d[xKey]))
        .attr("cy", d => yScale(d[yKey]))
        .attr("r", 5)
        .attr("fill", d => getColor(d))
        .attr("stroke", "#333")
        .attr("stroke-width", 0.5)
        .attr("opacity", 0.8)
        // Synchronized Hover
        .on("mouseover", function(event, d) {
            d3.selectAll(`#dot-${d.id}`).attr("r", 8).attr("stroke-width", 2).attr("stroke", "black");
            showTooltip(event, d);
        })
        .on("mouseout", function(event, d) {
            d3.selectAll(`#dot-${d.id}`).attr("r", 5).attr("stroke-width", 0.5).attr("stroke", "#333");
            hideTooltip();
        });
}

// --- COLOR LOGIC ---
function getColor(d) {
    if (colorMode === 'original') return colorOriginal(d.label);
    if (colorMode === 'precision') return colorPrecision(d.precision);
    if (colorMode === 'recall') return colorRecall(d.recall);
    if (colorMode === 'fscore') return colorFScore(d.f_score);
}

function updateColors() {
    d3.selectAll("circle.dot")
        .transition().duration(500)
        .attr("fill", d => getColor(d));
}

function updateLegend() {
    const gradient = d3.select("#legend-gradient");
    const desc = d3.select("#legend-description");
    const minLabel = d3.select("#legend-min");
    const maxLabel = d3.select("#legend-max");

    if (colorMode === 'original') {
        gradient.style("background", "linear-gradient(to right, #1f77b4, #ff7f0e, #2ca02c, #d62728)");
        minLabel.text("Class 1"); maxLabel.text("Class N");
        desc.text("Standard categorical colors mapping to original dataset classes.");
    } else if (colorMode === 'precision') {
        gradient.style("background", "linear-gradient(to right, #08306b, #c6dbef)");
        minLabel.text("Low Prec (High FPs)"); maxLabel.text("High Prec (0 FPs)");
        desc.html("Dark Blue reveals mapping artifacts where points are mistakenly grouped with wrong classes[cite: 15].");
    } else if (colorMode === 'recall') {
        gradient.style("background", "linear-gradient(to right, #67000d, #fcbba1)");
        minLabel.text("Low Recall (High FNs)"); maxLabel.text("High Recall (0 FNs)");
        desc.html("Dark Red reveals artifacts where points of the same class are incorrectly split apart by the projection[cite: 38].");
    } else if (colorMode === 'fscore') {
        gradient.style("background", "linear-gradient(to right, #440154, #fde725)");
        minLabel.text("0 (Bad)"); maxLabel.text("1 (Perfect)");
        desc.text("Combined structural preservation score.");
    }
}

// --- BRUSHING & LINKING LOGIC ---
function addBrush(containerSelector, xKey, yKey) {
    const container = d3.select(containerSelector).select("svg g");
    
    // We recreate scales to map pixel coordinates back to data values
    const width = d3.select(containerSelector).node().clientWidth - 60;
    const height = d3.select(containerSelector).node().clientHeight - 50;
    
    const xScale = d3.scaleLinear().domain(d3.extent(dataset, d => d[xKey])).nice().range([0, width]);
    const yScale = d3.scaleLinear().domain(d3.extent(dataset, d => d[yKey])).nice().range([height, 0]);

    const brush = d3.brush()
        .extent([[0, 0], [width, height]])
        .on("brush end", function(event) {
            if (!event.selection) {
                // Reset if click outside
                d3.selectAll("circle.dot").attr("opacity", 0.8);
                d3.select("#brush-stats").classed("hidden", true);
                return;
            }

            const [[x0, y0], [x1, y1]] = event.selection;
            let selectedPoints = [];

            d3.selectAll("circle.dot-pca").each(function(d) {
                const cx = xScale(d[xKey]);
                const cy = yScale(d[yKey]);
                const isSelected = x0 <= cx && cx <= x1 && y0 <= cy && cy <= y1;
                
                if (isSelected) selectedPoints.push(d);
                
                // Link opacity change to BOTH plots using the ID
                d3.selectAll(`#dot-${d.id}`).attr("opacity", isSelected ? 1 : 0.1);
            });

            updateLiveAnalytics(selectedPoints);
        });

    container.append("g").attr("class", "brush").call(brush);
}

// --- ANALYTICS AND TOOLTIP HELPERS ---
function updateLiveAnalytics(selectedPoints) {
    const statsBox = d3.select("#brush-stats");
    if (selectedPoints.length === 0) {
        statsBox.classed("hidden", true);
        return;
    }
    
    statsBox.classed("hidden", false);
    d3.select("#stat-count").text(selectedPoints.length);
    
    const avgPrec = d3.mean(selectedPoints, d => d.precision).toFixed(3);
    const avgRecall = d3.mean(selectedPoints, d => d.recall).toFixed(3);
    const avgFScore = d3.mean(selectedPoints, d => d.f_score).toFixed(3);

    d3.select("#stat-precision").text(avgPrec);
    d3.select("#stat-recall").text(avgRecall);
    d3.select("#stat-fscore").text(avgFScore);
}

function showTooltip(event, d) {
    const tooltip = d3.select("#tooltip");
    tooltip.transition().duration(100).style("opacity", 1);
    
    let diagnosis = "Structurally sound.";
    if (d.f_score < 0.3) diagnosis = "Severe mapping artifact.";
    else if (d.precision < 0.5) diagnosis = "Mixed with wrong classes.";
    else if (d.recall < 0.5) diagnosis = "Separated from its true class.";

    tooltip.html(`
        <strong>ID:</strong> ${d.id} | <strong>Class:</strong> ${d.label}<br>
        <strong>Precision:</strong> ${d.precision.toFixed(3)}<br>
        <strong>Recall:</strong> ${d.recall.toFixed(3)}<br>
        <strong>F-Score:</strong> ${d.f_score.toFixed(3)}<br>
        <hr style="border: 0.5px solid #7f8c8d; margin: 4px 0;">
        <em>${diagnosis}</em>
    `)
    .style("left", (event.pageX + 15) + "px")
    .style("top", (event.pageY - 28) + "px");
}

function hideTooltip() {
    d3.select("#tooltip").transition().duration(200).style("opacity", 0);
}