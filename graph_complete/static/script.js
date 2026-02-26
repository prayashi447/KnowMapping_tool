let network = null;
let mainNetwork = null;  // For main graph
let queryNetwork = null; // For query-specific graph
let graphData = { nodes: [], edges: [] };
let currentQueryGraph = null;

document.addEventListener('DOMContentLoaded', function() {
    document.getElementById('loadDataBtn').addEventListener('click', loadData);
    document.getElementById('queryBtn').addEventListener('click', queryGraph);
    document.getElementById('queryInput').addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            queryGraph();
        }
    });
    
    // Initialize with empty networks
    initializeNetworks();
});

function initializeNetworks() {
    // Main graph container
    const mainContainer = document.getElementById('graph');
    mainNetwork = new vis.Network(mainContainer, {}, {});
    
    // Query graph container
    const queryContainer = document.getElementById('queryGraphContainer');
    queryNetwork = new vis.Network(queryContainer, {}, {});
}

function loadData() {
    showLoading('Loading main graph...');
    
    fetch('/load_data', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        }
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            graphData = data.graph;
            renderMainGraph(graphData);
            hideLoading();
        } else {
            showError('Error loading data: ' + data.error);
        }
    })
    .catch(error => {
        showError('Error loading data: ' + error);
    });
}

function renderMainGraph(data) {
    const container = document.getElementById('graph');
    
    const nodes = new vis.DataSet(data.nodes.map(node => ({
        id: node.id,
        label: node.label,
        title: `Type: ${node.type}\n${node.text || ''}`,
        color: getNodeColor(node),
        shape: getNodeShape(node),
        font: { size: 14 }
    })));
    
    const edges = new vis.DataSet(data.edges.map(edge => ({
        from: edge.from,
        to: edge.to,
        label: edge.relation,
        arrows: 'to',
        font: { align: 'middle', size: 10 },
        title: edge.context || ''
    })));
    
    const options = {
        layout: {
            improvedLayout: true,
            hierarchical: {
                enabled: false
            }
        },
        interaction: {
            hover: true,
            tooltipDelay: 200,
            navigationButtons: true
        },
        physics: {
            stabilization: true,
            barnesHut: {
                gravitationalConstant: -8000,
                springConstant: 0.04,
                springLength: 95
            }
        },
        nodes: {
            borderWidth: 2,
            size: 30,
            shadow: true
        },
        edges: {
            width: 2,
            shadow: true,
            smooth: {
                type: 'continuous'
            }
        }
    };
    
    mainNetwork = new vis.Network(container, { nodes, edges }, options);
    
    mainNetwork.on('click', function(params) {
        if (params.nodes.length > 0) {
            const nodeId = params.nodes[0];
            getNodeInfo(nodeId);
        }
    });
}

function queryGraph() {
    const query = document.getElementById('queryInput').value;
    if (!query.trim()) {
        alert('Please enter a query');
        return;
    }
    
    showLoading('Processing query...');
    
    fetch('/query_graph', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ query: query })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            displayQueryResults(data.interpretation, data.relevant_nodes);
            
            // Render the new query-specific graph
            if (data.query_graph) {
                renderQueryGraph(data.query_graph, query);
                currentQueryGraph = data.query_graph;
            }
            
            // Show query graph section
            document.getElementById('queryGraphSection').style.display = 'block';
        } else {
            showError('Query error: ' + data.error);
        }
        hideLoading();
    })
    .catch(error => {
        showError('Query error: ' + error);
        hideLoading();
    });
}

function renderQueryGraph(graphData, query) {
    const container = document.getElementById('queryGraphContainer');
    
    // Clear previous graph
    while (container.firstChild) {
        container.removeChild(container.firstChild);
    }
    
    const nodes = new vis.DataSet(graphData.nodes.map(node => ({
        id: node.id,
        label: node.label,
        title: `Type: ${node.type}\n${node.text || ''}`,
        color: getNodeColor(node),
        shape: getNodeShape(node),
        font: { size: 12 }
    })));
    
    const edges = new vis.DataSet(graphData.edges.map(edge => ({
        from: edge.from,
        to: edge.to,
        label: edge.relation,
        arrows: 'to',
        font: { align: 'middle', size: 8 },
        title: edge.context || ''
    })));
    
    const options = {
        layout: {
            improvedLayout: true,
            hierarchical: {
                enabled: false
            }
        },
        interaction: {
            hover: true,
            tooltipDelay: 200,
            navigationButtons: true
        },
        physics: {
            stabilization: true,
            barnesHut: {
                gravitationalConstant: -5000,
                springConstant: 0.03,
                springLength: 85
            }
        },
        nodes: {
            borderWidth: 2,
            size: 25,
            shadow: true
        },
        edges: {
            width: 1.5,
            shadow: true,
            smooth: {
                type: 'continuous'
            }
        }
    };
    
    queryNetwork = new vis.Network(container, { nodes, edges }, options);
    
    // Update query title
    document.getElementById('currentQuery').textContent = `Query: ${query}`;
    
    queryNetwork.on('click', function(params) {
        if (params.nodes.length > 0) {
            const nodeId = params.nodes[0];
            // If it's a node from the main graph, get its info
            if (!nodeId.startsWith('Query_') && !nodeId.startsWith('Insight_')) {
                getNodeInfo(nodeId);
            }
        }
    });
}

function getNodeColor(node) {
    switch(node.type) {
        case 'review':
            return node.sentiment === 'positive' ? '#2ecc71' : 
                   node.sentiment === 'negative' ? '#e74c3c' : '#f39c12';
        case 'entity':
            return '#3498db';
        case 'sentiment':
            return '#9b59b6';
        case 'query':
            return '#e67e22';
        case 'insight':
            return '#1abc9c';
        default:
            return '#95a5a6';
    }
}

function getNodeShape(node) {
    switch(node.type) {
        case 'review':
            return 'box';
        case 'entity':
            return 'ellipse';
        case 'sentiment':
            return 'diamond';
        case 'query':
            return 'star';
        case 'insight':
            return 'triangle';
        default:
            return 'dot';
    }
}

function getNodeInfo(nodeId) {
    showLoading('Loading node info...');
    
    fetch('/get_node_info', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ node_id: nodeId })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            displayNodeInfo(data.node_info);
            displayInsights(data.insights);
            if (data.smaller_graph) {
                renderSmallerGraph(data.smaller_graph);
            }
        } else {
            showError('Error getting node info: ' + data.error);
        }
        hideLoading();
    })
    .catch(error => {
        showError('Error getting node info: ' + error);
        hideLoading();
    });
}

function displayQueryResults(interpretation, nodes) {
    const container = document.getElementById('queryResults');
    
    let html = '<div class="query-interpretation">';
    html += '<h4>🔍 Query Interpretation:</h4>';
    html += `<p>${interpretation}</p>`;
    html += '</div>';
    
    if (nodes.length > 0) {
        html += '<h4>📌 Relevant Nodes Found:</h4>';
        html += '<div class="relevant-nodes-list">';
        nodes.forEach(node => {
            html += `<div class="query-result-item" onclick="getNodeInfo('${node.id}')">`;
            html += `<span class="node-type-badge ${node.type}">${node.type}</span>`;
            html += `<strong>${node.text}</strong>`;
            html += `<span class="relevance-score">Relevance: ${(node.relevance * 100).toFixed(0)}%</span>`;
            html += `</div>`;
        });
        html += '</div>';
    }
    
    container.innerHTML = html;
}

function displayNodeInfo(nodeInfo) {
    const container = document.getElementById('nodeInfo');
    
    let html = '<div class="node-details">';
    html += `<h3>📊 Node Details</h3>`;
    html += `<p><strong>Type:</strong> <span class="node-type ${nodeInfo.node_type}">${nodeInfo.node_type}</span></p>`;
    html += `<p><strong>Text:</strong> ${nodeInfo.node_text}</p>`;
    
    if (nodeInfo.full_text) {
        html += `<p><strong>Full Context:</strong> <span class="context-text">${nodeInfo.full_text}</span></p>`;
    }
    
    if (nodeInfo.sentiment) {
        html += `<p><strong>Sentiment:</strong> <span class="sentiment-badge ${nodeInfo.sentiment}">${nodeInfo.sentiment}</span></p>`;
    }
    
    if (nodeInfo.neighbors && nodeInfo.neighbors.length > 0) {
        html += '<h4>🔗 Connected Nodes:</h4>';
        html += '<div class="neighbors-list">';
        nodeInfo.neighbors.forEach(neighbor => {
            html += `<div class="neighbor-item" onclick="getNodeInfo('${neighbor.node_id}')">`;
            html += `<span class="node-type-badge ${neighbor.node_type}">${neighbor.node_type}</span>`;
            html += `<strong>${neighbor.node_text}</strong>`;
            html += `<br><small>Relation: ${neighbor.relation}</small>`;
            if (neighbor.context) {
                html += `<br><small class="context">${neighbor.context.substring(0, 80)}...</small>`;
            }
            html += `</div>`;
        });
        html += '</div>';
    }
    
    html += '</div>';
    
    container.innerHTML = html;
}

function displayInsights(insights) {
    const container = document.getElementById('insightsContent');
    
    let html = '';
    
    if (insights.summary) {
        html += `<div class="insight-card">`;
        html += `<h4>📝 Summary</h4>`;
        html += `<p>${insights.summary}</p>`;
        html += `</div>`;
    }
    
    if (insights.relations_analysis) {
        html += `<div class="insight-card">`;
        html += `<h4>🔄 Relations Analysis</h4>`;
        html += `<p>${insights.relations_analysis}</p>`;
        html += `</div>`;
    }
    
    if (insights.significance) {
        html += `<div class="insight-card">`;
        html += `<h4>⭐ Significance</h4>`;
        html += `<p>${insights.significance}</p>`;
        html += `</div>`;
    }
    
    container.innerHTML = html || '<p class="no-data">No insights available</p>';
}

function renderSmallerGraph(graphData) {
    const container = document.getElementById('smallerGraphContainer');
    
    if (!graphData || !graphData.nodes || graphData.nodes.length === 0) {
        container.innerHTML = '<p class="no-data">No subgraph available</p>';
        return;
    }
    
    // Clear container
    while (container.firstChild) {
        container.removeChild(container.firstChild);
    }
    
    const nodes = new vis.DataSet(graphData.nodes.map(node => ({
        id: node.id,
        label: node.label,
        color: getNodeColor(node),
        shape: getNodeShape(node),
        font: { size: 10 }
    })));
    
    const edges = new vis.DataSet(graphData.edges.map(edge => ({
        from: edge.from,
        to: edge.to,
        label: edge.relation,
        arrows: 'to',
        font: { size: 8 }
    })));
    
    const options = {
        layout: {
            hierarchical: false
        },
        interaction: {
            hover: true,
            tooltipDelay: 200
        },
        physics: {
            enabled: true,
            stabilization: true,
            barnesHut: {
                gravitationalConstant: -3000,
                springConstant: 0.02,
                springLength: 70
            }
        },
        nodes: {
            borderWidth: 1,
            size: 15,
            shadow: true
        },
        edges: {
            width: 1,
            shadow: true
        }
    };
    
    const subNetwork = new vis.Network(container, { nodes, edges }, options);
    
    subNetwork.on('click', function(params) {
        if (params.nodes.length > 0) {
            getNodeInfo(params.nodes[0]);
        }
    });
}

function showLoading(message) {
    const loadingDiv = document.createElement('div');
    loadingDiv.className = 'loading-overlay';
    loadingDiv.id = 'loadingOverlay';
    loadingDiv.innerHTML = `<div class="loading-spinner">${message || 'Loading...'}</div>`;
    document.body.appendChild(loadingDiv);
}

function hideLoading() {
    const loadingDiv = document.getElementById('loadingOverlay');
    if (loadingDiv) {
        loadingDiv.remove();
    }
}

function showError(message) {
    const errorDiv = document.createElement('div');
    errorDiv.className = 'error-toast';
    errorDiv.innerHTML = message;
    errorDiv.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        background-color: #e74c3c;
        color: white;
        padding: 15px;
        border-radius: 5px;
        z-index: 1000;
        animation: slideIn 0.3s ease;
    `;
    document.body.appendChild(errorDiv);
    
    setTimeout(() => {
        errorDiv.remove();
    }, 5000);
}