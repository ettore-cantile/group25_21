// --- D3 VISUALIZATION DRAWING FUNCTIONS ---

// Function to draw a 2D scatter plot using D3 and attach interactions
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
        d3.selectAll(".dot").style("opacity", 0.9);
        
        d3.selectAll(".pc-line")
          .style("opacity", 0.6)
          .style("stroke-width", d => {
              const showAnon = d3.select("#show-anomalies").property("checked");
              return (showAnon && d.is_anomaly && colorMode === 'original') ? 2.5 : 1.5;
          })
          .style("stroke", d => getLineColor(d));
          
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
        .enter().append("path")
        .attr("class", d => `dot dot-${plotId} pt-${d.id}`)
        .attr("transform", d => `translate(${xScale(d[xKey])},${yScale(d[yKey])})`)
        .attr("d", function(d) {
            return getSymbolPath(`dot dot-${plotId} pt-${d.id}`, d, false);
        })
        .style("fill", d => customColorFn ? customColorFn(d) : getColor(d))
        .style("stroke", "var(--dot-stroke)")
        .style("stroke-width", 0.8)
        .style("opacity", 0.9)
        .style("cursor", "pointer")
        .on("mouseover", function(event, d) {
            resetAllHovers();
            const showAnon = d3.select("#show-anomalies").property("checked");
            let isAnom = false;
            if (plotId === 'pca2d') isAnom = d.pca_is_anomaly;
            else if (plotId === 'mds2d') isAnom = d.mds_is_anomaly;
            else if (plotId === 'kmeans') isAnom = d.is_anomaly;
            
            d3.selectAll(`.dot.pt-${d.id}`)
              .attr("d", function(p) { return getSymbolPath(d3.select(this).attr("class"), p, true); })
              .style("stroke", (showAnon && isAnom) ? "var(--anomaly-hover)" : "var(--hover-stroke)")
              .style("stroke-width", 2).raise();
              
            d3.selectAll(`.pc-line.pt-${d.id}`)
              .style("stroke", (showAnon && d.is_anomaly && colorMode === 'original') ? "var(--anomaly-hover)" : getLineColor(d))
              .style("stroke-width", 3)
              .style("opacity", 1) 
              .raise();
              
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

// Function to draw neighbor connections between selected data points
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

// Function to draw parallel coordinates mapping multi-dimensional data
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
        d3.selectAll(".dot").style("opacity", 0.9);

        d3.selectAll(".pc-line")
          .style("opacity", 0.6)
          .style("stroke-width", d => {
              const showAnon = d3.select("#show-anomalies").property("checked");
              return (showAnon && d.is_anomaly && colorMode === 'original') ? 2.5 : 1.5;
          })
          .style("stroke", d => getLineColor(d));

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
        .style("stroke", d => getLineColor(d))
        .style("stroke-width", d => {
            const showAnon = d3.select("#show-anomalies").property("checked");
            return (showAnon && d.is_anomaly && colorMode === 'original') ? 2.5 : 1.5;
        })
        .style("opacity", d => {
            const showAnon = d3.select("#show-anomalies").property("checked");
            return (showAnon && d.is_anomaly && colorMode === 'original') ? 0.9 : 0.6;
        }) 
        .style("cursor", "pointer")
        .on("mouseover", function(event, d) {
            resetAllHovers();
            const showAnon = d3.select("#show-anomalies").property("checked");

            d3.selectAll(`.dot.pt-${d.id}`)
              .attr("d", function(p) { return getSymbolPath(d3.select(this).attr("class"), p, true); })
              .style("stroke", "var(--hover-stroke)")
              .style("stroke-width", 2).raise();

            d3.selectAll(`.pc-line.pt-${d.id}`)
              .style("stroke", (showAnon && d.is_anomaly && colorMode === 'original') ? "var(--anomaly-hover)" : getLineColor(d))
              .style("stroke-width", 3)
              .style("opacity", 1) 
              .raise();
              
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

// Function to draw Sankey Flow Diagram comparing cluster agreements
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

    const sankeyMode = d3.select("input[name='sankeyMode']:checked").node().value;
    
    let sourceKey, targetKey, sourcePrefix, targetPrefix, isDiscrepancyFn;

    if (sankeyMode === 'pca-mds') {
        sourceKey = 'pca_kmeans_cluster';
        targetKey = 'mds_kmeans_cluster';
        sourcePrefix = 'PCA '; 
        targetPrefix = 'MDS ';
        isDiscrepancyFn = (d) => String(d.pca_kmeans_cluster) !== String(d.mds_kmeans_cluster);
    } else if (sankeyMode === 'gt-pca') {
        sourceKey = 'label';
        targetKey = 'pca_kmeans_cluster';
        sourcePrefix = 'GT ';
        targetPrefix = 'PCA ';
        isDiscrepancyFn = (d) => d.pca_is_anomaly;
    } else if (sankeyMode === 'gt-mds') {
        sourceKey = 'label';
        targetKey = 'mds_kmeans_cluster';
        sourcePrefix = 'GT ';
        targetPrefix = 'MDS ';
        isDiscrepancyFn = (d) => d.mds_is_anomaly;
    }

    const sourceClusters = Array.from(new Set(activeData.map(d => String(d[sourceKey])).filter(c => c !== "undefined"))).sort();
    const targetClusters = Array.from(new Set(activeData.map(d => String(d[targetKey])).filter(c => c !== "undefined"))).sort();

    if (sourceClusters.length === 0 || targetClusters.length === 0) return;

    const nodes = [];
    const nodeMap = new Map();
    let nodeIndex = 0;

    sourceClusters.forEach(c => {
        const name = `${sourcePrefix}${c}`;
        nodes.push({ name: name, type: 'source', cluster: c, isGT: sourceKey === 'label' });
        nodeMap.set(name, nodeIndex++);
    });
    
    targetClusters.forEach(c => {
        const name = `${targetPrefix}${c}`;
        nodes.push({ name: name, type: 'target', cluster: c, isGT: targetKey === 'label' });
        nodeMap.set(name, nodeIndex++);
    });

    const linkMap = new Map();
    activeData.forEach(d => {
        if (d[sourceKey] !== undefined && d[targetKey] !== undefined) {
            const sourceName = `${sourcePrefix}${d[sourceKey]}`;
            const targetName = `${targetPrefix}${d[targetKey]}`;
            const key = `${sourceName}->${targetName}`;
            
            const isDiscrepancy = isDiscrepancyFn(d);
            
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
                .attr("d", function(p) {
                    const plotClass = d3.select(this).attr("class");
                    const overrideR = linkPointIds.has(p.id) ? currentPointSize * 1.5 : currentPointSize;
                    return getSymbolPath(plotClass, p, false, overrideR);
                });
                
            d3.selectAll(".pc-line:not(.filtered-out)")
                .style("stroke", p => getLineColor(p))
                .style("opacity", p => linkPointIds.has(p.id) ? 1 : 0.05)
                .style("stroke-width", p => {
                    const showAnon = d3.select("#show-anomalies").property("checked");
                    return (showAnon && p.is_anomaly && colorMode === 'original') ? 2.5 : 1.5;
                });
                
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
            d3.selectAll(".pc-line:not(.filtered-out)").filter(p => linkPointIds.has(p.id)).raise();
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
        .style("fill", d => colorOriginal(d.cluster)) 
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

// Function to draw the analytical Radar Chart for displaying multi-dimensional point profile
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

    const pcaData = pcaKmeansAvg[String(point.pca_kmeans_cluster)];
    if (pcaData) {
        svg.append("path")
            .datum(getCoordinates(pcaData))
            .attr("d", lineBuilder)
            .style("fill", "none")
            .style("stroke", "var(--radar-kmeans)") 
            .style("stroke-width", "2px");
    }

    const mdsData = mdsKmeansAvg[String(point.mds_kmeans_cluster)];
    if (mdsData) {
        svg.append("path")
            .datum(getCoordinates(mdsData))
            .attr("d", lineBuilder)
            .style("fill", "none")
            .style("stroke", "var(--radar-kmeans-mds)") 
            .style("stroke-width", "2px");
    }

    const pointData = {};
    radarDimensions.forEach(dim => pointData[dim] = +point.attributes[dim]); 
    
    svg.append("path")
        .datum(getCoordinates(pointData))
        .attr("d", lineBuilder)
        .style("fill", "none")
        .style("stroke", "var(--radar-point)") 
        .style("stroke-width", "3px");
        
    svg.selectAll(".radar-point")
        .data(getCoordinates(pointData))
        .enter().append("circle")
        .attr("cx", d => d.x).attr("cy", d => d.y)
        .attr("r", 4)
        .style("fill", "var(--radar-point)");
}

// Function to draw local neighbor graph using D3 force physics simulation
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

// Function handling the drag logic for the neighbor network graph
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

// Function for initial setup of small circular Gauges
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

// Function to transition Gauge arcs based on a selected metric value
function updateGauge(gaugeObj, value, color, textSelector) {
    const safeValue = isNaN(value) ? 0 : value;
    const targetAngle = gaugeAngleScale(safeValue);
    const arcFg = d3.arc().innerRadius(30).outerRadius(45).startAngle(-Math.PI / 2).cornerRadius(3);

    d3.select(textSelector).text((safeValue * 100).toFixed(1) + "%");

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

// Function that updates the dynamically generated confusion matrix tables
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