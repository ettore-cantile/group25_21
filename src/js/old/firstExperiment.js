// ==========================================
// 1. VARIABILI GLOBALI E SETUP
// ==========================================
let globalData = [];
let globalStats = {};
let currentMode = 'error'; // Modalità di default: mostra gli errori (Falsi Positivi/Negativi)
let colorScaleError;
const colorScaleClass = d3.scaleOrdinal(d3.schemeCategory10);

// Dizionario per salvare le scale (x, y) di entrambi i grafici per poterci disegnare le linee
const chartScales = { pca: null, mds: null };

// Percorso del JSON (Relativo alla posizione dell'HTML in ./src/html/)
const DATA_URL = "../../dataset/firstExperiment.json";

// ==========================================
// 2. CARICAMENTO DATI (D3.js)
// ==========================================
d3.json(DATA_URL).then(function(data) {
    // Il JSON è diviso in data.stats e data.points
    globalData = data.points;
    globalStats = data.stats;
    
    // --- POPOLAMENTO STATISTICHE NELLA DASHBOARD ---
    d3.select("#stat-pca-err").text(globalStats.pca_global);
    d3.select("#stat-pca-match").text(globalStats.pca_matches);
    d3.select("#stat-mds-err").text(globalStats.mds_global);
    d3.select("#stat-mds-match").text(globalStats.mds_matches);
    
    // --- SCALA DISCRETA PER L'ERRORE (ColorBrewer RdBu 7-classi) ---
    // [Rosso Scuro, Rosso, Rosso Chiaro, Neutro(Grigio), Blu Chiaro, Blu, Blu Scuro]
    const colorBrewerRdBu7 = ["#b2182b", "#ef8a62", "#fddbc7", "#f7f7f7", "#d1e5f0", "#67a9cf", "#2166ac"];

    // Trova l'errore massimo assoluto per centrare la scala su 0
    const maxAbsScore = d3.max(globalData, d => Math.max(Math.abs(d.score_pca), Math.abs(d.score_mds)));
    
    // scaleQuantize divide automaticamente il dominio in 7 "fasce" o "scalini" perfetti
    colorScaleError = d3.scaleQuantize()
        .domain([-maxAbsScore, maxAbsScore]) 
        .range(colorBrewerRdBu7);

    // Disegniamo dinamicamente i blocchetti nella legenda HTML per la scala discreta
    const legendBar = d3.select("#discrete-error-legend");
    legendBar.html(""); 
    colorBrewerRdBu7.forEach(color => {
        legendBar.append("div")
            .attr("class", "discrete-step")
            .style("background-color", color);
    });

    // --- SETUP LEGENDA CLASSI (Vini) ---
    const classes = [...new Set(globalData.map(d => d.class_name))].sort();
    const legendClassDiv = d3.select("#legend-class");
    classes.forEach(c => {
        const box = legendClassDiv.append("div").attr("class", "legend-box");
        box.append("div").attr("class", "color-dot").style("background", colorScaleClass(c));
        box.append("span").text(c);
    });

    // Disegna i grafici per la prima volta
    drawCharts();
    
}).catch(err => {
    console.error("Errore caricamento D3:", err);
    alert("Errore nel caricamento del JSON.\nHai avviato il server Python locale? (es. python -m http.server)");
});

// ==========================================
// 3. GESTIONE INTERFACCIA E MODALITÀ
// ==========================================
function setMode(mode) {
    currentMode = mode;
    
    // Aggiorna UI Bottoni
    d3.selectAll("button").classed("active", false);
    d3.select("#btn-" + mode).classed("active", true);
    
    // Mostra/Nascondi le legende corrette
    d3.select("#legend-error").style("display", mode === 'error' ? 'flex' : 'none');
    d3.select("#legend-class").style("display", mode === 'class' ? 'flex' : 'none');
    
    // Anima il cambio di colore dei punti
    d3.select("#chart-pca").selectAll("circle").transition().duration(500)
        .attr("fill", d => currentMode === 'error' ? colorScaleError(d.score_pca) : colorScaleClass(d.class_name));
        
    d3.select("#chart-mds").selectAll("circle").transition().duration(500)
        .attr("fill", d => currentMode === 'error' ? colorScaleError(d.score_mds) : colorScaleClass(d.class_name));
}

// ==========================================
// 4. DISEGNO DEI GRAFICI
// ==========================================
function drawCharts() {
    drawScatter("#chart-pca", "pca", "score_pca", "container-pca");
    drawScatter("#chart-mds", "mds", "score_mds", "container-mds");
}

function drawScatter(selector, type, scoreProp, containerId) {
    const container = document.getElementById(containerId);
    const rect = container.getBoundingClientRect();
    
    // Preveniamo il bug dell'altezza negativa in caso di caricamento CSS ritardato
    const w = Math.max(300, rect.width - 40);
    const h = Math.max(340, rect.height - 60); 
    
    const xProp = type + "_x";
    const yProp = type + "_y";

    // Pulisce l'SVG precedente in caso di ridimensionamento della finestra
    d3.select(selector).html("");

    const svg = d3.select(selector).append("svg")
        .attr("width", w)
        .attr("height", h)
        .style("overflow", "visible"); 
    
    // Scale (adattate sui dati min e max)
    const x = d3.scaleLinear().domain(d3.extent(globalData, d => d[xProp])).nice().range([30, w-20]);
    const y = d3.scaleLinear().domain(d3.extent(globalData, d => d[yProp])).nice().range([h-30, 20]);

    // Salviamo le scale globalmente per usarle poi per tracciare le linee dei vicini
    chartScales[type] = { x: x, y: y };

    // Disegno degli Assi
    svg.append("g").attr("transform", `translate(0,${h-30})`).call(d3.axisBottom(x).ticks(5));
    svg.append("g").attr("transform", `translate(30,0)`).call(d3.axisLeft(y).ticks(5));

    // Creiamo due layer per evitare che le linee coprano i cerchi
    const linksLayer = svg.append("g").attr("class", "links-layer");
    const nodesLayer = svg.append("g").attr("class", "nodes-layer");

    // Disegno dei Punti (Nodi)
    nodesLayer.selectAll("circle")
        .data(globalData)
        .enter().append("circle")
        .attr("cx", d => x(d[xProp]))
        .attr("cy", d => y(d[yProp]))
        .attr("r", 6)
        .attr("fill", d => currentMode === 'error' ? colorScaleError(d[scoreProp]) : colorScaleClass(d.class_name))
        .attr("class", d => "dot-" + d.id)
        
        // --- INTERAZIONI: Brushing, Linking e Tooltip ---
        .on("mouseover", function(event, d) {
            // Evidenzia lo stesso punto su entrambi i grafici
            d3.selectAll(".dot-" + d.id).classed("hovered", true).raise();
            
            // Attenua tutti gli altri punti
            d3.selectAll("circle:not(.dot-" + d.id + ")").classed("dimmed", true);

            // Disegna le linee verso i veri vicini dello spazio a 13D
            drawNeighborLines("pca", d);
            drawNeighborLines("mds", d);

            // Mostra Tooltip
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
            // Ripristina l'opacità dei punti
            d3.selectAll("circle").classed("hovered", false).classed("dimmed", false);
            
            // Nascondi Tooltip
            d3.select("#tooltip").style("opacity", 0);
            
            // Cancella tutte le linee dei vicini disegnate
            d3.selectAll(".links-layer").html("");
        });
}

// ==========================================
// 5. DISEGNO DEI COLLEGAMENTI (VICINI REALI)
// ==========================================
function drawNeighborLines(chartType, sourceData) {
    // Seleziona il livello delle linee del grafico corretto
    const layer = d3.select(`#chart-${chartType} .links-layer`);
    
    // Recupera le scale di quel grafico
    const scaleX = chartScales[chartType].x;
    const scaleY = chartScales[chartType].y;

    // Coordinate del punto su cui abbiamo il mouse
    const sourceX = scaleX(sourceData[`${chartType}_x`]);
    const sourceY = scaleY(sourceData[`${chartType}_y`]);

    // Itera sugli ID dei veri vicini (calcolati dal Python in R^13)
    sourceData.neighbors.forEach(targetId => {
        // Trova i dati del vicino
        const targetData = globalData.find(p => p.id === targetId);
        
        if(targetData) {
            // Calcola la posizione del vicino sul grafico
            const targetX = scaleX(targetData[`${chartType}_x`]);
            const targetY = scaleY(targetData[`${chartType}_y`]);

            // Disegna la linea
            layer.append("line")
                .attr("class", "link-line")
                .attr("x1", sourceX)
                .attr("y1", sourceY)
                .attr("x2", targetX)
                .attr("y2", targetY);
        }
    });
}

// Ridisegna i grafici se la finestra del browser cambia dimensione
window.addEventListener('resize', drawCharts);