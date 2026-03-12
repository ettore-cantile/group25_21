// --- GLOBAL STATE ---
let dataset = [];
let metadata = {};
let colorMode = 'original';
let uniqueClasses = [];
let pointById = new Map();
let currentPointSize = 5;

// D3 Brushes
const brushPCA = d3.brush();
const brushMDS = d3.brush();

// --- OPTIMIZED DISCRETE COLOR SCALES ---
const customTableau = [...d3.schemeTableau10];
customTableau[2] = '#2ca02c'; 
const colorOriginal = d3.scaleOrdinal(customTableau);

const bluesDiscrete = ["#08519c", "#3182bd", "#6baed6", "#9ecae1", "#c6dbef"];
const colorPrecision = d3.scaleQuantize().domain([0, 1]).range(bluesDiscrete); 

const redsDiscrete = ["#a50f15", "#de2d26", "#fb6a4a", "#fc9272", "#fcbba1"];
const colorRecall = d3.scaleQuantize().domain([0, 1]).range(redsDiscrete);

const rdYlGnDiscrete = ["#d73027", "#fdae61", "#ffffbf", "#a6d96a", "#1a9641"];
const colorFScore = d3.scaleQuantize().domain([0, 1]).range(rdYlGnDiscrete);

// Gauges
const gauges = {
    precision: { foreground: null },
    recall: { foreground: null },
    fscore: { foreground: null }
};
const gaugeAngleScale = d3.scaleLinear().domain([0, 1]).range([-Math.PI / 2, Math.PI / 2]);

// --- INITIALIZATION ---
Promise.all([
    d3.json("../json/step2_final_data.json?v=" + Date.now()),
    d3.csv("../../dataset/wine.csv")
]).then(([data, wineData]) => {
    data.points.forEach((p, i) => {
        if (wineData[i]) p.attributes = wineData[i];
    });

    dataset = data.points;
    metadata = data.metadata;
    uniqueClasses = Array.from(new Set(dataset.map(d => d.label))).sort();
    pointById = new Map(dataset.map(p => [p.id, p]));

    // 1. Inject Global Data in Footer "Buttons"
    if(metadata) {
        d3.select("#fb-dataset").text(metadata.dataset || "-");
        
        // Extract base statistics
        if (metadata.global_assessment && metadata.global_assessment.pca && metadata.global_assessment.mds) {
            d3.select("#fb-pca-trust").text((metadata.global_assessment.pca.trustworthiness * 100).toFixed(1) + "%");
            d3.select("#fb-pca-cont").text((metadata.global_assessment.pca.continuity * 100).toFixed(1) + "%");
            d3.select("#fb-mds-trust").text((metadata.global_assessment.mds.trustworthiness * 100).toFixed(1) + "%");
            d3.select("#fb-mds-cont").text((metadata.global_assessment.mds.continuity * 100).toFixed(1) + "%");
        }

        // Extract global_f_score and insert it into the last button
        let gFScore = metadata.global_f_score;
        if (gFScore === undefined && metadata.global_assessment) {
            gFScore = metadata.global_assessment.global_f_score || metadata.global_assessment.f_score;
        }
        
        if (gFScore !== undefined) {
            d3.select("#fb-global-fscore").text((gFScore * 100).toFixed(1) + "%");
        } else {
            d3.select("#fb-global-fscore").text("N/A");
        }
    }

    // Initialize main charts
    drawPlot("#pca-plot", "pca_x", "pca_y", "pca", brushPCA);
    drawPlot("#mds-plot", "mds_x", "mds_y", "mds", brushMDS);
    setupBrushing();
    
    // Initialize gauges
    initGauge("#gauge-precision", gauges.precision);
    initGauge("#gauge-recall", gauges.recall);
    initGauge("#gauge-fscore", gauges.fscore);

    // Color and Size Switcher Management
    enhanceColorModeSwitcher();

    d3.selectAll("input[name='colorMode']").on("change", function() {
        colorMode = this.value;
        updateColors();
        updateLegend();
    });

    d3.select("#point-size-slider").on("input", function() {
        currentPointSize = +this.value;
        d3.selectAll("circle.dot").attr("r", currentPointSize);
    });

    updateLegend();
}).catch(err => console.error("Error loading JSON:", err));


// --- PLOTTING FUNCTION ---
function drawPlot(containerSelector, xKey, yKey, plotId, brushObj) {
    const container = d3.select(containerSelector); 
    const width = container.node().clientWidth;
    const height = container.node().clientHeight;
    // Margins to let axes breathe and avoid clipping
    const margin = { top: 20, right: 25, bottom: 35, left: 35 };

    const svgRoot = container.append("svg")
        .attr("viewBox", `0 0 ${width} ${height}`)
        .attr("preserveAspectRatio", "none")
        .style("width", "100%")
        .style("height", "100%");

    svgRoot.on("mouseleave", resetAllHovers);

    svgRoot.on("click", () => {
        d3.selectAll("circle.dot").attr("opacity", 0.9);
        d3.selectAll(".link-group line").remove();
        updateLiveAnalytics([]); 
    });

    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;

    const svg = svgRoot.append("g")
        .attr("transform", `translate(${margin.left},${margin.top})`);

    const xScale = d3.scaleLinear().domain(d3.extent(dataset, d => d[xKey])).nice().range([0, innerWidth]);
    const yScale = d3.scaleLinear().domain(d3.extent(dataset, d => d[yKey])).nice().range([innerHeight, 0]);

    svg.append("g").attr("transform", `translate(0,${innerHeight})`).call(d3.axisBottom(xScale).ticks(5));
    svg.append("g").call(d3.axisLeft(yScale).ticks(5));

    svg.append("g").attr("class", "link-group");
    const brushGroup = svg.append("g").attr("class", "brush-group");

    svg.selectAll(".dot")
        .data(dataset)
        .enter().append("circle")
        .attr("class", `dot dot-${plotId}`)
        .attr("id", d => `dot-${d.id}`)
        .attr("cx", d => xScale(d[xKey]))
        .attr("cy", d => yScale(d[yKey]))
        .attr("r", currentPointSize)
        .attr("fill", d => getColor(d))
        .attr("stroke", "rgba(0,0,0,0.4)")
        .attr("stroke-width", 0.8)
        .attr("opacity", 0.9)
        .style("cursor", "pointer")
        .on("mouseover", function(event, d) {
            resetAllHovers();
            d3.selectAll(`#dot-${d.id}`).attr("r", currentPointSize * 1.6).attr("stroke", "rgba(0,0,0,0.9)").attr("stroke-width", 2).raise();
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

// --- CLICK LOGIC (Single Point) ---
function updateSelection(d) {
    d3.select("#pca-plot .brush-group").call(brushPCA.move, null);
    d3.select("#mds-plot .brush-group").call(brushMDS.move, null);

    d3.select("#dynamic-panel-title").text("Neighbor Graph");
    d3.select("#empty-state-placeholder").classed("hidden-panel", true);
    d3.select("#gauges-container").classed("hidden-panel", true);
    d3.select("#neighbor-graph-container").classed("hidden-panel", false);

    const neighborIds = d.neighbors || [];
    const activeIds = new Set([d.id, ...neighborIds]);

    d3.selectAll("circle.dot").attr("opacity", p => activeIds.has(p.id) ? 0.9 : 0.15);

    drawLines("pca", d, neighborIds, "pca_x", "pca_y", brushPCA);
    drawLines("mds", d, neighborIds, "mds_x", "mds_y", brushMDS);

    const neighborsData = neighborIds.map(id => pointById.get(id)).filter(Boolean);
    drawNeighborGraph(d, neighborsData);
}

function drawLines(plotId, sourceD, neighborIds, xKey, yKey, scales) {
    const linkGroup = d3.select(`#${plotId}-plot .link-group`);
    linkGroup.selectAll("line").remove(); 

    const sourceX = scales.xScale(sourceD[xKey]);
    const sourceY = scales.yScale(sourceD[yKey]);

    const linesData = neighborIds.map(id => pointById.get(id)).filter(Boolean);

    linkGroup.selectAll("line")
        .data(linesData)
        .enter().append("line")
        .attr("x1", sourceX)
        .attr("y1", sourceY)
        .attr("x2", target => scales.xScale(target[xKey]))
        .attr("y2", target => scales.yScale(target[yKey]))
        .attr("stroke", target => target.label === sourceD.label ? "#2ca02c" : "#e74c3c")
        .attr("stroke-width", 1.5)
        .attr("opacity", 0.6);
}

// --- BI-DIRECTIONAL BRUSHING ---
function setupBrushing() {
    brushPCA.on("start brush end", function(event) {
        if(event.sourceEvent && event.sourceEvent.type === "mousedown") d3.select("#mds-plot .brush-group").call(brushMDS.move, null);
        handleBrush(event, brushPCA, "pca_x", "pca_y");
    });

    brushMDS.on("start brush end", function(event) {
        if(event.sourceEvent && event.sourceEvent.type === "mousedown") d3.select("#pca-plot .brush-group").call(brushPCA.move, null);
        handleBrush(event, brushMDS, "mds_x", "mds_y");
    });
}

function handleBrush(event, brushObj, xKey, yKey) {
    if (!event.sourceEvent) return;
    d3.selectAll(".link-group line").remove(); 

    if (!event.selection) {
        if (event.type === "end") {
            d3.selectAll("circle.dot").attr("opacity", 0.9); 
            resetAllHovers(); 
            updateLiveAnalytics([]); 
        }
        return;
    }

    const [[x0, y0], [x1, y1]] = event.selection;
    let selectedPoints = [];
    dataset.forEach(d => {
        const cx = brushObj.xScale(d[xKey]);
        const cy = brushObj.yScale(d[yKey]);
        const isSelected = x0 <= cx && cx <= x1 && y0 <= cy && cy <= y1;
        if (isSelected) selectedPoints.push(d);
        d3.selectAll(`#dot-${d.id}`).attr("opacity", isSelected ? 0.9 : 0.15);
    });
    
    updateLiveAnalytics(selectedPoints);
}

// --- LOGIC TO UPDATE GAUGES, PLACEHOLDER AND DYNAMIC PANEL ---
function updateLiveAnalytics(selectedPoints) {
    
    if (selectedPoints.length === 0) {
        d3.select("#dynamic-panel-title").text("Live Analytics");
        d3.select("#gauges-container").classed("hidden-panel", true);
        d3.select("#neighbor-graph-container").classed("hidden-panel", true);
        d3.select("#empty-state-placeholder").classed("hidden-panel", false);
        return;
    }
    
    d3.select("#dynamic-panel-title").text("Selection Metrics");
    d3.select("#empty-state-placeholder").classed("hidden-panel", true);
    d3.select("#neighbor-graph-container").classed("hidden-panel", true);
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
    d3.selectAll("circle.dot").attr("r", currentPointSize).attr("stroke", "rgba(0,0,0,0.4)").attr("stroke-width", 0.8);
    hideTooltip();
}

// --- UI HELPERS ---
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

// --- COLOR SCALES & LEGEND ---
function getColor(d) {
    if (colorMode === 'original') return colorOriginal(d.label);
    if (colorMode === 'precision') return colorPrecision(d.precision);
    if (colorMode === 'recall') return colorRecall(d.recall);
    if (colorMode === 'fscore') return colorFScore(d.f_score);
}

function updateColors() {
    d3.selectAll("circle.dot").transition().duration(500).attr("fill", d => getColor(d));
}

function updateLegend() {
    const gradient = d3.select("#legend-gradient");
    const labelsDiv = d3.select("#legend-labels");
    
    gradient.selectAll("*").remove(); 
    labelsDiv.selectAll("*").remove();

    if (colorMode === 'original') {
        gradient.style("border", "none").style("background", "transparent").style("justify-content", "flex-end").style("gap", "15px");
        uniqueClasses.forEach(cls => {
            const item = gradient.append("div").attr("class", "legend-cluster-item");
            item.append("div").attr("class", "legend-cluster-dot").style("background-color", colorOriginal(cls));
            item.append("span").text(`Producer ${cls}`);
        });
    } else {
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

// --- 3 RESPONSIVE GAUGES ---
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

// --- TOOLTIPS ---
function showTooltip(event, d) {
    const tooltip = d3.select("#tooltip");
    
    let fpText = d.precision < 0.9 
        ? `🔴 <strong>False Positive:</strong> Attracts <strong>${((1 - d.precision)*100).toFixed(1)}%</strong> of points from other classes.` 
        : `🟢 <strong>Low FPs:</strong> No class mixing.`;
        
    let fnText = d.recall < 0.9 
        ? `🔴 <strong>False Negative:</strong> Disconnected from <strong>${((1 - d.recall)*100).toFixed(1)}%</strong> of points in its own class.` 
        : `🟢 <strong>Low FNs:</strong> Highly cohesive.`;

    tooltip.html(`
        <strong>ID:</strong> ${d.id} | <strong>Class:</strong> ${d.label}<br>
        <strong>Precision:</strong> ${(d.precision*100).toFixed(1)}%<br>
        <strong>Recall:</strong> ${(d.recall*100).toFixed(1)}%<br>
        <strong>F-Score:</strong> ${(d.f_score*100).toFixed(1)}%
        <div class="tt-diagnosis">
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

    const attributesHtml = Object.entries(d.attributes)
        .map(([key, value]) => `<strong>${key.replace(/_/g, ' ')}:</strong> ${value}`).join('<br>');

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

// --- NEIGHBOR GRAPH (FORCE-DIRECTED) ---
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
        .on("mouseover", showAttributeTooltip) 
        .on("mouseout", hideTooltip)
        .call(drag(simulation, centerNode, size));

    node.append("circle")
        .attr("r", d => d.id === centerNode.id ? 20 : 15)
        .attr("fill", d => colorOriginal(d.label));

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