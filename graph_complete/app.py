import pandas as pd
import networkx as nx
from flask import Flask, render_template, request, jsonify
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import spacy
from collections import defaultdict
import json
import re

app = Flask(__name__)

# Load models globally to avoid reloading
print("Loading models...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_name = "google/flan-t5-base"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)
print("Models loaded successfully!")

# Load spaCy for NLP
try:
    nlp = spacy.load("en_core_web_sm")
except:
    import os
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

class ReviewKnowledgeGraph:
    def __init__(self):
        self.graph = nx.MultiDiGraph()
        self.reviews = []
        self.query_graphs = {}  # Store query-specific graphs
        
    def extract_entities_and_relations(self, text, sentiment, review_id):
        """Extract entities and relations from text using spaCy"""
        doc = nlp(text)
        entities = []
        
        # Extract named entities
        for ent in doc.ents:
            if ent.label_ in ['PERSON', 'ORG', 'PRODUCT', 'GPE', 'DATE', 'EVENT']:
                entities.append({
                    'text': ent.text,
                    'label': ent.label_,
                    'start': ent.start_char,
                    'end': ent.end_char
                })
        
        # Also extract important noun phrases
        for chunk in doc.noun_chunks:
            if len(chunk.text.split()) <= 3 and not chunk.text.lower() in ['the', 'a', 'an']:
                entities.append({
                    'text': chunk.text,
                    'label': 'TOPIC',
                    'start': chunk.start_char,
                    'end': chunk.end_char
                })
        
        return entities
    
    def build_graph_from_reviews(self, df, num_rows=5):
        """Build knowledge graph from first N rows of dataframe"""
        for idx, row in df.head(num_rows).iterrows():
            review_text = row['review'] if 'review' in row else row.iloc[0]
            sentiment = row['sentiment'] if 'sentiment' in row else 'neutral'
            
            review_node = f"Review_{idx}"
            self.graph.add_node(review_node, 
                               type='review',
                               text=review_text[:100] + "...",
                               full_text=review_text,
                               sentiment=sentiment)
            
            # Extract entities
            entities = self.extract_entities_and_relations(review_text, sentiment, idx)
            
            # Add entity nodes and relations
            entity_ids = []
            for entity in entities:
                entity_id = f"Entity_{entity['text'].replace(' ', '_')}_{idx}"
                self.graph.add_node(entity_id,
                                   type='entity',
                                   text=entity['text'],
                                   label=entity['label'])
                self.graph.add_edge(review_node, entity_id, 
                                   relation='mentions',
                                   context=review_text[:200])
                entity_ids.append(entity_id)
            
            # Add sentiment as a node
            sentiment_node = f"Sentiment_{sentiment}_{idx}"
            self.graph.add_node(sentiment_node,
                               type='sentiment',
                               value=sentiment)
            self.graph.add_edge(review_node, sentiment_node, relation='has_sentiment')
            
            # Create relations between entities in the same review
            for i, ent1_id in enumerate(entity_ids):
                for j, ent2_id in enumerate(entity_ids):
                    if i < j:
                        self.graph.add_edge(
                            ent1_id,
                            ent2_id,
                            relation='co-occurs_in_review',
                            review_id=review_node
                        )
        
        return self.graph
    
    def generate_query_graph(self, query, relevant_nodes):
        """Generate a new knowledge graph based on query results"""
        query_graph = nx.MultiDiGraph()
        query_id = f"Query_{len(self.query_graphs)}"
        
        # Add query as central node
        query_graph.add_node(query_id,
                            type='query',
                            text=query,
                            label=f"Query: {query[:50]}...")
        
        # Add relevant nodes and their connections
        for node_info in relevant_nodes:
            node_id = node_info['id']
            if node_id in self.graph:
                # Add the relevant node
                node_data = self.graph.nodes[node_id]
                query_graph.add_node(node_id,
                                    type=node_data.get('type', 'unknown'),
                                    text=node_data.get('text', node_data.get('value', '')),
                                    label=node_data.get('text', node_data.get('value', '')))
                
                # Connect to query
                query_graph.add_edge(query_id, node_id,
                                    relation='relevant_to_query',
                                    relevance_score=node_info.get('relevance', 0.8))
                
                # Add immediate neighbors (1-hop)
                for neighbor in self.graph.neighbors(node_id):
                    if neighbor not in query_graph:
                        neighbor_data = self.graph.nodes[neighbor]
                        query_graph.add_node(neighbor,
                                           type=neighbor_data.get('type', 'unknown'),
                                           text=neighbor_data.get('text', neighbor_data.get('value', '')),
                                           label=neighbor_data.get('text', neighbor_data.get('value', '')))
                    
                    # Add edge
                    edge_data = self.graph.get_edge_data(node_id, neighbor)
                    if edge_data:
                        first_edge = next(iter(edge_data.values()))
                        query_graph.add_edge(node_id, neighbor,
                                           relation=first_edge.get('relation', 'connected'),
                                           context=first_edge.get('context', ''))
        
        # Use FLAN-T5 to identify additional relations
        query_context = f"Query: {query}\n"
        query_context += f"Relevant nodes: {[n['text'] for n in relevant_nodes[:3]]}\n"
        query_context += "Based on this query, what relationships should exist between these nodes?"
        
        try:
            inputs = tokenizer(query_context, return_tensors="pt", max_length=512, truncation=True).to(device)
            outputs = model.generate(inputs.input_ids, max_length=200, num_beams=4)
            insights = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Add insights as a node
            insight_node = f"Insight_{query_id}"
            query_graph.add_node(insight_node,
                                type='insight',
                                text=insights,
                                label='AI Insight')
            
            # Connect insight to query
            query_graph.add_edge(query_id, insight_node,
                               relation='generates_insight')
            
            # Connect insight to relevant nodes based on keywords
            insight_words = set(insights.lower().split())
            for node_id in relevant_nodes[:3]:  # Connect to top 3 nodes
                node_text = self.graph.nodes[node_id].get('text', '').lower()
                if any(word in node_text for word in insight_words):
                    query_graph.add_edge(insight_node, node_id,
                                       relation='analyzes')
        
        except Exception as e:
            print(f"Error generating insights: {e}")
        
        # Store the query graph
        self.query_graphs[query_id] = query_graph
        
        return self.convert_graph_to_json(query_graph), query_id
    
    def convert_graph_to_json(self, graph):
        """Convert networkx graph to JSON format"""
        graph_data = {
            'nodes': [],
            'edges': []
        }
        
        for node in graph.nodes():
            node_data = graph.nodes[node]
            graph_data['nodes'].append({
                'id': node,
                'type': node_data.get('type', 'unknown'),
                'label': node_data.get('label', node_data.get('text', node_data.get('value', ''))),
                'text': node_data.get('text', node_data.get('value', '')),
                'sentiment': node_data.get('sentiment', '')
            })
        
        for u, v, data in graph.edges(data=True):
            graph_data['edges'].append({
                'from': u,
                'to': v,
                'relation': data.get('relation', 'connected'),
                'context': data.get('context', '')
            })
        
        return graph_data
    
    def get_node_info(self, node_id):
        """Get information about a specific node"""
        if node_id not in self.graph:
            return None
        
        node_data = self.graph.nodes[node_id]
        edges = list(self.graph.edges(node_id, data=True))
        
        # Get neighboring nodes
        neighbors = []
        for u, v, data in edges:
            if u == node_id:
                neighbor_id = v
            else:
                neighbor_id = u
                
            neighbor_data = self.graph.nodes[neighbor_id]
            neighbors.append({
                'node_id': neighbor_id,
                'node_type': neighbor_data.get('type', 'unknown'),
                'node_text': neighbor_data.get('text', neighbor_data.get('value', '')),
                'relation': data.get('relation', 'connected'),
                'context': data.get('context', '')
            })
        
        return {
            'node_id': node_id,
            'node_type': node_data.get('type', 'unknown'),
            'node_text': node_data.get('text', node_data.get('value', '')),
            'full_text': node_data.get('full_text', ''),
            'sentiment': node_data.get('sentiment', ''),
            'neighbors': neighbors
        }
    
    def generate_insights_with_flan(self, node_info, neighbors_info):
        """Generate insights using FLAN-T5 model"""
        context = f"Node: {node_info['node_text']} (Type: {node_info['node_type']})\n"
        
        if node_info['full_text']:
            context += f"Full context: {node_info['full_text']}\n"
        
        context += "Connected entities:\n"
        for neighbor in neighbors_info[:5]:
            context += f"- {neighbor['node_text']} (Type: {neighbor['node_type']}, Relation: {neighbor['relation']})\n"
        
        prompts = [
            f"Based on this information: {context}\n\nWhat are the key insights about {node_info['node_text']}?",
            f"Summarize the relationships: {context}\n\nWhat is the main connection pattern?",
            f"Generate a brief analysis: {context}\n\nWhat is significant about this node?",
        ]
        
        insights = []
        for prompt in prompts:
            try:
                inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to(device)
                outputs = model.generate(
                    inputs.input_ids,
                    max_length=150,
                    num_beams=4,
                    temperature=0.7,
                    do_sample=True
                )
                insight = tokenizer.decode(outputs[0], skip_special_tokens=True)
                insights.append(insight)
            except Exception as e:
                insights.append(f"Error generating insight: {str(e)}")
        
        return {
            'summary': insights[0],
            'relations_analysis': insights[1] if len(insights) > 1 else "",
            'significance': insights[2] if len(insights) > 2 else ""
        }
    
    def generate_smaller_graph(self, node_id, depth=2):
        """Generate a smaller subgraph around a selected node"""
        if node_id not in self.graph:
            return None
        
        nodes_to_include = {node_id}
        current_nodes = {node_id}
        
        for _ in range(depth):
            next_nodes = set()
            for node in current_nodes:
                neighbors = list(self.graph.neighbors(node))
                next_nodes.update(neighbors)
                nodes_to_include.update(neighbors)
            current_nodes = next_nodes
        
        subgraph = self.graph.subgraph(nodes_to_include)
        return self.convert_graph_to_json(subgraph)

# Initialize knowledge graph
kg = ReviewKnowledgeGraph()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/load_data', methods=['POST'])
def load_data():
    """Load CSV data and build initial knowledge graph"""
    try:
        # Sample data
        sample_data = {
            'review': [
                "The camera quality is amazing, but battery life is poor. I love the design though!",
                "Customer service was extremely helpful. They resolved my issue quickly.",
                "The product arrived late and was damaged. Very disappointed with the delivery service.",
                "Great value for money! The features are comparable to expensive brands.",
                "Software updates have made this device even better over time. Highly recommended!"
            ],
            'sentiment': ['positive', 'positive', 'negative', 'positive', 'positive']
        }
        df = pd.DataFrame(sample_data)
        
        # Build graph
        graph = kg.build_graph_from_reviews(df, num_rows=5)
        
        # Convert graph to format for frontend
        graph_data = kg.convert_graph_to_json(graph)
        
        return jsonify({'success': True, 'graph': graph_data})
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/get_node_info', methods=['POST'])
def get_node_info():
    """Get information about a specific node"""
    try:
        data = request.json
        node_id = data.get('node_id')
        
        node_info = kg.get_node_info(node_id)
        if not node_info:
            return jsonify({'success': False, 'error': 'Node not found'})
        
        # Generate insights using FLAN-T5
        insights = kg.generate_insights_with_flan(
            node_info, 
            node_info['neighbors']
        )
        
        # Generate smaller subgraph
        smaller_graph = kg.generate_smaller_graph(node_id)
        
        return jsonify({
            'success': True,
            'node_info': node_info,
            'insights': insights,
            'smaller_graph': smaller_graph
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/query_graph', methods=['POST'])
def query_graph():
    """Query the graph and generate a new query-specific graph"""
    try:
        data = request.json
        query = data.get('query')
        
        # Use FLAN-T5 to interpret the query and find relevant nodes
        prompt = f"""Given this knowledge graph about product reviews, find the most relevant nodes for the query: '{query}'
        
        Consider:
        1. Entity nodes (products, features, brands)
        2. Sentiment nodes (positive, negative, neutral)
        3. Review nodes (full reviews)
        
        Return a list of the most relevant node types and specific keywords to search for."""
        
        inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to(device)
        outputs = model.generate(inputs.input_ids, max_length=150, num_beams=4)
        interpretation = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Find relevant nodes in the graph
        relevant_nodes = []
        query_words = set(query.lower().split())
        
        for node in kg.graph.nodes():
            node_data = kg.graph.nodes[node]
            node_text = node_data.get('text', node_data.get('value', '')).lower()
            
            # Calculate relevance score
            relevance = 0
            node_words = set(node_text.split())
            
            # Direct word matches
            matches = query_words.intersection(node_words)
            relevance += len(matches) * 0.3
            
            # Check if node type matches query intent
            if 'sentiment' in query.lower() and node_data.get('type') == 'sentiment':
                relevance += 0.5
            if 'feature' in query.lower() and node_data.get('type') == 'entity':
                relevance += 0.5
            
            if relevance > 0:
                relevant_nodes.append({
                    'id': node,
                    'text': node_data.get('text', node_data.get('value', '')),
                    'type': node_data.get('type', 'unknown'),
                    'relevance': min(relevance, 1.0)
                })
        
        # Sort by relevance and take top results
        relevant_nodes.sort(key=lambda x: x['relevance'], reverse=True)
        relevant_nodes = relevant_nodes[:10]  # Limit to top 10
        
        # Generate a new query-specific graph
        query_graph, query_id = kg.generate_query_graph(query, relevant_nodes)
        
        return jsonify({
            'success': True,
            'interpretation': interpretation,
            'relevant_nodes': relevant_nodes[:5],  # Show top 5 in results
            'query_graph': query_graph,
            'query_id': query_id
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/get_query_graph/<query_id>', methods=['GET'])
def get_query_graph(query_id):
    """Retrieve a previously generated query graph"""
    try:
        if query_id in kg.query_graphs:
            graph_data = kg.convert_graph_to_json(kg.query_graphs[query_id])
            return jsonify({'success': True, 'graph': graph_data})
        else:
            return jsonify({'success': False, 'error': 'Query graph not found'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5000)