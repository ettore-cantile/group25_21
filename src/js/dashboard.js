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

// --- DISCRETE COLOR SCALES OTTIMIZZATE ---
// Originali: d3.schemeTableau10, ma con il rosso sostituito da un verde per coerenza tematica.
const customTableau = [...d3.schemeTableau10];
customTableau[2] = '#2ca02c'; // Sostituisce il rosso (#e15759) con un verde standard.
const colorOriginal = d3.scaleOrdinal(customTableau);

// Precision (Blues): Scala invertita. 0 (Poco preciso) = Blu scuro -> 1 (Molto preciso) = Azzurro chiaro.
const bluesDiscrete = ["#08519c", "#3182bd", "#6baed6", "#9ecae1", "#c6dbef"];
const colorPrecision = d3.scaleQuantize().domain([0, 1]).range(bluesDiscrete); 

// Recall (Reds): Scala invertita. 0 (Poco coeso) = Rosso scuro -> 1 (Molto coeso) = Rosa chiaro.
const redsDiscrete = ["#a50f15", "#de2d26", "#fb6a4a", "#fc9272", "#fcbba1"];
const colorRecall = d3.scaleQuantize().domain([0, 1]).range(redsDiscrete);

// F-Score (Red-Yellow-Green): Scala più intuitiva. 0 (Pessimo) = Rosso, 0.5 = Giallo, 1 (Ottimo) = Verde.
const rdYlGnDiscrete = ["#d73027", "#fdae61", "#ffffbf", "#a6d96a", "#1a9641"];
const colorFScore = d3.scaleQuantize().domain([0, 1]).range(rdYlGnDiscrete);

// Tachometri
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
    // Merge attributes from CSV into the main dataset
    data.points.forEach((p, i) => {
        if (wineData[i]) {
            p.attributes = wineData[i];
        }
    });

    dataset = data.points;
    metadata = data.metadata;
    uniqueClasses = Array.from(new Set(dataset.map(d => d.label))).sort();
    pointById = new Map(dataset.map(p => [p.id, p]));


    // 1. Inject Global Assessment
    d3.select("#global-assessment").html(`
        Dataset: <strong>${metadata.dataset}</strong> <br>
        <span style="color: #2980b9;">${metadata.global_assessment.message}</span>
    `);

    // 2. Inject Static Analytics (Trustworthiness & Continuity) into Right Panel
    if(metadata.global_assessment.pca && metadata.global_assessment.mds) {
        const qualityContainer = d3.select(".right-panel").insert("div", "h3")
            .attr("class", "stat-card");

        const pca_trust = (metadata.global_assessment.pca.trustworthiness * 100).toFixed(1);
        const pca_cont = (metadata.global_assessment.pca.continuity * 100).toFixed(1);
        const mds_trust = (metadata.global_assessment.mds.trustworthiness * 100).toFixed(1);
        const mds_cont = (metadata.global_assessment.mds.continuity * 100).toFixed(1);

        qualityContainer.html(`
            <h4 style="margin-bottom: 15px;">PROJECTION QUALITY (GLOBAL)</h4>
            <div class="quality-grid">
                <div class="quality-header"></div>
                <div class="quality-header">Trustworthiness</div>
                <div class="quality-header">Continuity</div>

                <div class="quality-label">PCA</div>
                <div class="quality-value">${pca_trust}%</div>
                <div class="quality-value">${pca_cont}%</div>

                <div class="quality-label">MDS</div>
                <div class="quality-value">${mds_trust}%</div>
                <div class="quality-value">${mds_cont}%</div>
            </div>
        `);
    }

    // 3. Aggiungi contenitore per il grafo dei vicini (inizialmente nascosto)
    d3.select(".right-panel").append("div")
        .attr("id", "neighbor-graph-container")
        .attr("class", "hidden")
        .html(`
            <h3 style="margin-top: 25px; border-top: 1px solid #eee; padding-top: 20px;">Neighbor Graph</h3>
            <p class="help-text">Graph of the selected point and its k-nearest neighbors from the original high-dimensional space.</p>
            <svg id="neighbor-graph-svg"></svg>
        `);

    drawPlot("#pca-plot", "pca_x", "pca_y", "pca", brushPCA);
    drawPlot("#mds-plot", "mds_x", "mds_y", "mds", brushMDS);
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

    // Listener per il cambio dimensione punti
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
    const margin = { top: 35, right: 35, bottom: 35, left: 35 };

    const svgRoot = container.append("svg")
        .attr("viewBox", `0 0 ${width} ${height}`)
        .attr("preserveAspectRatio", "xMidYMid meet")
        .style("width", "100%")
        .style("height", "100%");

    // Aggiungo un listener sull'elemento SVG radice. Questo agisce come un "failsafe"
    // per gli hover "appiccicosi". Se il mouse lascia l'area del grafico rapidamente,
    // l'evento "mouseout" del cerchio potrebbe non attivarsi. Questo garantisce il reset.
    svgRoot.on("mouseleave", resetAllHovers);

    // Aggiungo un click sullo sfondo per resettare tutte le selezioni
    svgRoot.on("click", () => {
        d3.selectAll("circle.dot").attr("opacity", 0.9);
        d3.selectAll(".link-group line").remove();
        d3.select("#neighbor-graph-container").classed("hidden", true);
        d3.select("#brush-stats").classed("hidden", true);
        d3.select("#matrix-container").classed("hidden", true);
        updateLiveAnalytics([]); // Resetta e nasconde i componenti di analisi
    });

    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;

    // Per evitare lo stretching, creiamo un'area di plot quadrata.
    const plotSize = Math.min(innerWidth, innerHeight);
    const xOffset = (innerWidth - plotSize) / 2;
    const yOffset = (innerHeight - plotSize) / 2;

    // Gruppo principale traslato per centrare l'area di plot quadrata.
    const svg = svgRoot.append("g")
        .attr("transform", `translate(${margin.left + xOffset},${margin.top + yOffset})`);

    const xScale = d3.scaleLinear().domain(d3.extent(dataset, d => d[xKey])).nice().range([0, plotSize]);
    const yScale = d3.scaleLinear().domain(d3.extent(dataset, d => d[yKey])).nice().range([plotSize, 0]);

    svg.append("g").attr("transform", `translate(0,${plotSize})`).call(d3.axisBottom(xScale).ticks(5));
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
        // FIX 1: Bordo grigio scuro semitrasparente. Ancora visivamente il punto allo sfondo!
        .attr("stroke", "rgba(0,0,0,0.4)")
        .attr("stroke-width", 0.8)
        // FIX 2: Opacità base alzata a 0.9 per colori più vividi
        .attr("opacity", 0.9)
        .style("cursor", "pointer")
        .on("mouseover", function(event, d) {
            // Chiamata preventiva per pulire qualsiasi stato "appiccicoso".
            // Questo risolve il problema dei punti che rimangono evidenziati
            // durante movimenti veloci del mouse DENTRO al grafico, garantendo
            // che solo il punto corrente sia evidenziato.
            resetAllHovers();

            d3.selectAll(`#dot-${d.id}`)
              .attr("r", currentPointSize * 1.6)
              .attr("stroke", "rgba(0,0,0,0.9)") // Bordo più netto all'hover
              .attr("stroke-width", 2)
              .raise();
            showTooltip(event, d);
        })
        .on("mouseout", function(event, d) {
            // Quando il mouse esce da un punto (verso uno spazio vuoto), resetta lo stato.
            // La combinazione di questo e del reset su mouseover garantisce la massima robustezza.
            resetAllHovers();
        })
        .on("click", function(event, d) {
            event.stopPropagation(); // Impedisce al click sullo sfondo di attivarsi
            updateSelection(d);
        });

    brushObj.extent([[0, 0], [plotSize, plotSize]]);
    brushGroup.call(brushObj);
    
    brushObj.xScale = xScale;
    brushObj.yScale = yScale;
}

// --- LOGICA DI CLICK E DISEGNO LINEE ---
function updateSelection(d) {
    d3.select("#pca-plot .brush-group").call(brushPCA.move, null);
    d3.select("#mds-plot .brush-group").call(brushMDS.move, null);

    // Nascondi i componenti di analisi per selezioni multiple (brush)
    d3.select("#brush-stats").classed("hidden", true);
    d3.select(".gauges-wrapper").classed("hidden", true);

    const neighborIds = d.neighbors || [];
    const activeIds = new Set([d.id, ...neighborIds]);

    // Modifica opacità: isola punto cliccato e vicini (alzato a 0.15)
    d3.selectAll("circle.dot").attr("opacity", p => activeIds.has(p.id) ? 0.9 : 0.15);

    // Disegna le linee di collegamento e il grafo dei vicini
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
    d3.select("#neighbor-graph-container").classed("hidden", true);

    if (!event.selection) {
        if (event.type === "end") {
            d3.selectAll("circle.dot").attr("opacity", 0.9); // Torna vivido
            resetAllHovers(); // Rimuove anche eventuali hover rimasti attivi
            d3.select("#brush-stats").classed("hidden", true);
            d3.select("#matrix-container").classed("hidden", true);
            updateLiveAnalytics([]); // Nasconde i tachimetri
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

// --- HELPERS ---

/**
 * Resetta lo stato di hover (raggio e bordo) di tutti i punti su entrambi i grafici.
 * Nasconde anche il tooltip. Funziona come un failsafe globale.
 */
function resetAllHovers() {
    d3.selectAll("circle.dot")
        .attr("r", currentPointSize)
        .attr("stroke", "rgba(0,0,0,0.4)")
        .attr("stroke-width", 0.8);
    hideTooltip();
}

// --- UI HELPERS ---

/**
 * Trasforma i radio button di base per la selezione del colore in uno switch moderno.
 * Trova il contenitore, lo svuota, e inietta la nuova struttura HTML.
 * Questo approccio non richiede modifiche al file HTML principale.
 */
function enhanceColorModeSwitcher() {
    const options = [
        { value: 'original', label: 'Original' },
        { value: 'precision', label: 'Precision' },
        { value: 'recall', label: 'Recall' },
        { value: 'fscore', label: 'F-Score' }
    ];

    // Assumiamo che i radio button originali siano in un div con classe "control-group"
    const container = d3.select(".control-group");
    if (container.empty()) { return; }
    container.html(""); // Pulisce il contenuto esistente (i vecchi radio button)

    // Trasforma il vecchio div nel nuovo componente
    container.classed("control-group", false).classed("segmented-control", true);

    options.forEach((opt, i) => {
        container.append("input")
            .attr("type", "radio")
            .attr("id", `cm-${opt.value}`)
            .attr("name", "colorMode")
            .attr("value", opt.value)
            .property("checked", i === 0); // Seleziona 'Original' di default

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
            item.append("span").text(`Class ${cls}`);
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

// --- 3 TACHOMETRI (GAUGES) RESPONSIVE ---
function initGauge(selector, gaugeObj) {
    const svg = d3.select(selector);
    const width = 100, height = 60;

    svg.attr("viewBox", `0 0 ${width} ${height}`)
       .style("width", "100%")
       .style("height", "auto");

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

function updateLiveAnalytics(selectedPoints) {
    const statsBox = d3.select("#brush-stats");
    const gaugesWrapper = d3.select(".gauges-wrapper");

    // I tachimetri e le statistiche di brush appaiono solo per selezioni > 1
    const isMultiSelection = selectedPoints.length > 1;

    statsBox.classed("hidden", !isMultiSelection);
    gaugesWrapper.classed("hidden", !isMultiSelection);

    if (selectedPoints.length === 0) {
        // Se la selezione è vuota, resetta i gauge a 0.
        updateGauge(gauges.precision, 0, "#bdc3c7", "#val-precision");
        updateGauge(gauges.recall, 0, "#bdc3c7", "#val-recall");
        updateGauge(gauges.fscore, 0, "#bdc3c7", "#val-fscore");
        return;
    }
    
    if (isMultiSelection) {
        d3.select("#stat-count").text(selectedPoints.length);
        
        const avgPrec = d3.mean(selectedPoints, d => d.precision);
        const avgRecall = d3.mean(selectedPoints, d => d.recall);
        const avgFScore = d3.mean(selectedPoints, d => d.f_score);

        updateGauge(gauges.precision, avgPrec, colorFScore(avgPrec), "#val-precision");
        updateGauge(gauges.recall, avgRecall, colorFScore(avgRecall), "#val-recall");
        updateGauge(gauges.fscore, avgFScore, colorFScore(avgFScore), "#val-fscore");
    }
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

    // --- Logica di posizionamento dinamico per evitare che il tooltip esca dallo schermo ---
    const tooltipNode = tooltip.node();
    const tooltipWidth = tooltipNode.offsetWidth;
    const tooltipHeight = tooltipNode.offsetHeight;
    const margin = 20;

    // Posizione orizzontale: di default a destra, ma a sinistra se non c'è spazio.
    let x = event.pageX + margin;
    if (x + tooltipWidth > window.innerWidth) { x = event.pageX - tooltipWidth - margin; }

    // Posizione verticale: di default sotto, ma sopra se non c'è spazio in basso.
    let y = event.pageY + margin;
    if (y + tooltipHeight > window.innerHeight) { y = event.pageY - tooltipHeight - margin; }

    tooltip.style("left", x + "px").style("top", y + "px").transition().duration(100).style("opacity", 1);
}

function hideTooltip() {
    d3.select("#tooltip").transition().duration(200).style("opacity", 0);
}

function showAttributeTooltip(event, d) {
    const tooltip = d3.select("#tooltip");

    if (!d.attributes) {
        return showTooltip(event, d); // Fallback se gli attributi non sono caricati
    }

    // Costruisce una stringa HTML con tutti gli attributi del vino
    const attributesHtml = Object.entries(d.attributes)
        .map(([key, value]) => `<strong>${key.replace(/_/g, ' ')}:</strong> ${value}`)
        .join('<br>');

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

/**
 * Disegna il grafo dei vicini nel pannello di destra.
 * @param {object} centerNode Il punto selezionato al centro.
 * @param {Array<object>} neighborNodes I suoi vicini.
 */
function drawNeighborGraph(centerNode, neighborNodes) {
    const container = d3.select("#neighbor-graph-container");
    container.classed("hidden", false);

    const svg = d3.select("#neighbor-graph-svg");
    svg.selectAll("*").remove();

    const width = svg.node().getBoundingClientRect().width;
    const height = svg.node().getBoundingClientRect().height;

    // Crea un'area di disegno quadrata al centro per evitare lo stretching del layout
    const size = Math.min(width, height);
    const xOffset = (width - size) / 2;
    const yOffset = (height - size) / 2;
    const g = svg.append("g").attr("transform", `translate(${xOffset}, ${yOffset})`);

    const graphNodes = [centerNode, ...neighborNodes].map(n => ({...n}));
    const graphLinks = neighborNodes.map(n => ({
        source: centerNode.id,
        target: n.id
        // NOTA: la distanza (peso) non è presente nel dataset, quindi non può essere mostrata.
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
        .on("mouseover", showAttributeTooltip) // Usa il nuovo tooltip con gli attributi
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

/**
 * Helper per abilitare il trascinamento dei nodi nel grafo.
 * @param {object} simulation La simulazione D3.
 * @param {object} centerNode Il nodo centrale che deve rimanere fisso.
 * @param {number} size La dimensione dell'area di disegno per i limiti.
 */
function drag(simulation, centerNode, size) {
    function dragstarted(event, d) {
        if (!event.active) simulation.alphaTarget(0.3).restart();
        d.fx = d.x;
        d.fy = d.y;
    }
    function dragged(event, d) {
        // Blocca i nodi all'interno dell'area di disegno
        d.fx = Math.max(0, Math.min(size, event.x));
        d.fy = Math.max(0, Math.min(size, event.y));
    }
    function dragended(event, d) {
        if (!event.active) simulation.alphaTarget(0);
        if (d.id !== centerNode.id) { // Non rilasciare la posizione fissa del nodo centrale
            d.fx = null;
            d.fy = null;
        }
    }
    return d3.drag().on("start", dragstarted).on("drag", dragged).on("end", dragended);
}