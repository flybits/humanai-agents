"""
NetworkX Knowledge Graph for Alice and Bob
Replaces Neo4j with a simple in-memory graph structure.
"""

import networkx as nx
from typing import List, Dict, Tuple
import datetime
import csv

def create_knowledge_graph() -> nx.MultiDiGraph:
    """Create NetworkX knowledge graph with Alice and Bob's data."""
    
    G = nx.MultiDiGraph()
    
    # Alice's knowledge graph
    alice_data = {
        'profession': 'Urban Planner',
        'years_experience': 20,
        'is_politically': 'Conservative',
        'prefers_tech_adoption_style': 'Cautious',
        'prefers_planning_approach': 'Gradualism',
        'values': [
            'Respect for tradition',
            'Environmental sustainability', 
            'Community stability'
        ],
        'believes': [
            'Infrastructure should reflect historical identity',
            'Technology should complement physical infrastructure'
        ],
        'has_experience_in': [
            'Heritage Restoration',
            'Low-rise Zoning',
            'Green Spaces & Recycling'
        ],
        'led_project': [
            'Historic Building Restoration',
            'Low-rise Zoning Policies',
            'Community Recycling Programs'
        ]
    }
    
    # Bob's knowledge graph
    bob_data = {
        'profession': 'Urban Planner',
        'years_experience': 15,
        'is_politically': 'Progressive',
        'prefers_tech_adoption_style': 'Aggressive',
        'prefers_planning_approach': 'Disruption',
        'values': [
            'Efficiency',
            'Economic competitiveness'
        ],
        'believes': [
            'Urban innovation drives progress',
            'Technology should replace outdated systems'
        ],
        'has_experience_in': [
            'Smart Infrastructure',
            'Autonomous Transit',
            'Real-time Data Governance'
        ],
        'led_project': [
            'IoT Urban Networks',
            'Autonomous Transit Systems',
            'Data Governance Platforms'
        ]
    }
    
    # Add Alice to graph
    G.add_node('Alice', node_type='person')
    _add_agent_data_to_graph(G, 'Alice', alice_data)
    
    # Add Bob to graph  
    G.add_node('Bob', node_type='person')
    _add_agent_data_to_graph(G, 'Bob', bob_data)
    
    print(f"✓ Knowledge graph created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    return G

def _add_agent_data_to_graph(G: nx.MultiDiGraph, agent_name: str, data: Dict):
    """Add an agent's data to the graph."""
    
    for relationship, values in data.items():
        if isinstance(values, list):
            # Handle multiple values (e.g., multiple values, beliefs, etc.)
            for value in values:
                value_node = f"{agent_name}_{relationship}_{value.replace(' ', '_')}"
                G.add_node(value_node, node_type=relationship, label=value)
                G.add_edge(agent_name, value_node, relationship=relationship)
        else:
            # Handle single values (e.g., profession, years_experience)
            value_node = f"{agent_name}_{relationship}_{str(values).replace(' ', '_')}"
            G.add_node(value_node, node_type=relationship, label=str(values))
            G.add_edge(agent_name, value_node, relationship=relationship)

def query_graph_for_context(G: nx.MultiDiGraph, agent_name: str, context: str, model) -> List[Tuple[str, str]]:
    """
    Query the NetworkX graph for relevant triplets based on context.
    Returns list of (relationship, value) tuples.
    """
    
    # Generate search strategy using LLM
    search_strategy = generate_search_strategy(context, agent_name, model)
    
    # Execute search on NetworkX graph
    relevant_triplets = execute_networkx_search(G, agent_name, search_strategy)
    
    return relevant_triplets

def generate_search_strategy(context: str, agent_name: str, model) -> Dict:
    """Generate search strategy for NetworkX graph traversal."""
    
    context_summary = context[-1000:]  # Last 1000 chars for recent context
    
    prompt = f"""
You are a strategic identity retrieval system for an AI agent in a political simulation.

CONTEXT:
Agent: {agent_name} (running for mayor)
Current situation: "{context_summary}"

TASK: Determine which identity traits are most relevant for this situation.

AVAILABLE RELATIONSHIP TYPES:
- profession, years_experience, is_politically
- prefers_tech_adoption_style, prefers_planning_approach  
- values, believes, has_experience_in, led_project

REASONING GUIDELINES:
1. If context mentions "environment/pollution" → prioritize: values, believes (environmental)
2. If context mentions "technology/innovation" → prioritize: prefers_tech_adoption_style, believes (tech)
3. If context mentions "community/tradition" → prioritize: values (tradition), has_experience_in
4. If context mentions "economy/development" → prioritize: prefers_planning_approach, led_project
5. If context mentions "campaign/election" → prioritize: is_politically, values
6. Always include: profession, is_politically (for baseline context)

OUTPUT FORMAT:
Return a JSON object with priority-ordered relationship types and optional keywords:
{{
    "high_priority": ["relationship_type1", "relationship_type2"],
    "medium_priority": ["relationship_type3"],
    "keywords": ["keyword1", "keyword2"]
}}

EXAMPLE:
For context about pollution: {{"high_priority": ["values", "believes"], "medium_priority": ["has_experience_in"], "keywords": ["environment", "sustainability"]}}
"""

    try:
        response = model.sample_text(prompt, max_tokens=200, terminators=['\n\n'])
        # Try to parse as JSON, fallback to default if it fails
        import json
        strategy = json.loads(response.strip())
        return strategy
    except:
        # Fallback strategy
        return {
            "high_priority": ["is_politically", "values"],
            "medium_priority": ["believes", "profession"],
            "keywords": []
        }

def execute_networkx_search(G: nx.MultiDiGraph, agent_name: str, search_strategy: Dict) -> List[Tuple[str, str]]:
    """Execute search strategy on NetworkX graph."""
    
    if agent_name not in G:
        return []
    
    relevant_triplets = []
    
    # Get all edges from the agent
    agent_edges = G.edges(agent_name, data=True)
    
    # Priority-based retrieval
    high_priority = search_strategy.get("high_priority", [])
    medium_priority = search_strategy.get("medium_priority", [])
    keywords = search_strategy.get("keywords", [])
    
    # First pass: high priority relationships
    for _, target_node, edge_data in agent_edges:
        relationship = edge_data.get('relationship', '')
        if relationship in high_priority:
            node_label = G.nodes[target_node].get('label', '')
            relevant_triplets.append((relationship, node_label))
    
    # Second pass: medium priority relationships (if we need more)
    if len(relevant_triplets) < 3:
        for _, target_node, edge_data in agent_edges:
            relationship = edge_data.get('relationship', '')
            if relationship in medium_priority:
                node_label = G.nodes[target_node].get('label', '')
                if (relationship, node_label) not in relevant_triplets:
                    relevant_triplets.append((relationship, node_label))
    
    # Third pass: keyword matching (if we still need more)
    if len(relevant_triplets) < 2:
        for _, target_node, edge_data in agent_edges:
            relationship = edge_data.get('relationship', '')
            node_label = G.nodes[target_node].get('label', '').lower()
            
            # Check if any keyword appears in the label
            keyword_match = any(keyword.lower() in node_label for keyword in keywords)
            if keyword_match and (relationship, G.nodes[target_node].get('label', '')) not in relevant_triplets:
                relevant_triplets.append((relationship, G.nodes[target_node].get('label', '')))
    
    # Limit to most relevant (max 6 triplets)
    return relevant_triplets[:6]

def format_triplets_for_identity(agent_name: str, triplets: List[Tuple[str, str]]) -> str:
    """Format triplets into readable identity statements."""
    
    if not triplets:
        # Fallback identity
        return f"{agent_name} is an urban planner and mayoral candidate."
    
    statements = []
    for relationship, value in triplets:
        if relationship == 'profession':
            statements.append(f"{agent_name} is a {value}.")
        elif relationship == 'years_experience':
            statements.append(f"{agent_name} has {value} years of experience.")
        elif relationship == 'is_politically':
            statements.append(f"{agent_name} is politically {value}.")
        elif relationship == 'values':
            statements.append(f"{agent_name} values {value}.")
        elif relationship == 'believes':
            statements.append(f"{agent_name} believes {value}.")
        elif relationship == 'has_experience_in':
            statements.append(f"{agent_name} has experience in {value}.")
        elif relationship == 'led_project':
            statements.append(f"{agent_name} has led projects involving {value}.")
        elif relationship == 'prefers_tech_adoption_style':
            statements.append(f"{agent_name} prefers a {value} approach to technology adoption.")
        elif relationship == 'prefers_planning_approach':
            statements.append(f"{agent_name} prefers {value} in planning approaches.")
        else:
            statements.append(f"{agent_name} {relationship}: {value}.")
    
    return "\n".join(statements)

# Create global graph instance
KNOWLEDGE_GRAPH = None

def get_knowledge_graph() -> nx.MultiDiGraph:
    """Get or create the knowledge graph instance."""
    global KNOWLEDGE_GRAPH
    if KNOWLEDGE_GRAPH is None:
        KNOWLEDGE_GRAPH = create_knowledge_graph()
    return KNOWLEDGE_GRAPH

# Performance logging functions
KG_PERFORMANCE_LOG = []

def log_kg_performance(agent_name: str, query_time: float, retrieval_success: bool, clock):
    """Log KG performance metrics."""
    KG_PERFORMANCE_LOG.append({
        'agent': agent_name,
        'query_time': query_time,
        'success': retrieval_success,
        'timestamp': clock.now().strftime('%Y-%m-%d %H:%M:%S')
    })

def save_kg_performance_results():
    """Save KG performance metrics to CSV for analysis."""
    if KG_PERFORMANCE_LOG:
        filename = f"kg_performance_experiment1_id-rag_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        with open(filename, 'w', newline='') as csvfile:
            fieldnames = ['agent', 'query_time', 'success', 'timestamp']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for entry in KG_PERFORMANCE_LOG:
                writer.writerow(entry)
        
        print(f"✓ KG performance metrics saved to {filename}")
        
        # Print summary statistics
        total_queries = len(KG_PERFORMANCE_LOG)
        successful_queries = sum(1 for entry in KG_PERFORMANCE_LOG if entry['success'])
        avg_query_time = sum(entry['query_time'] for entry in KG_PERFORMANCE_LOG) / total_queries
        success_rate = successful_queries / total_queries * 100
        
        print(f"Summary: {total_queries} total queries, {successful_queries} successful ({success_rate:.1f}%), avg time: {avg_query_time:.2f}s")
    else:
        print("No KG performance data collected")

def save_quiz_graph_performance_results(measurements):
    """Save quiz-specific graph performance metrics to CSV for analysis."""
    quiz_performance_data = []
    
    # Extract quiz graph performance from all agents
    for agent_name in ["Alice", "Bob"]:
        channel_name = f"{agent_name}_IdentityRecallMetric"
        if channel_name in measurements.available_channels():
            channel_data = []
            # Correctly use subscribe/dispose to get all historical data
            subscription = measurements.get_channel(channel_name).subscribe(on_next=channel_data.append)
            subscription.dispose()
            
            for datum in channel_data:
                if 'quiz_graph_performance' in datum and datum['quiz_graph_performance']:
                    for perf_entry in datum['quiz_graph_performance']:
                        quiz_performance_data.append({
                            'agent': agent_name,
                            'quiz_time': datum['time_str'],
                            'question': perf_entry['question'],
                            'query_time': perf_entry['query_time'],
                            'success': perf_entry['success'],
                            'timestamp': perf_entry['timestamp']
                        })
    
    if quiz_performance_data:
        filename = f"quiz_graph_performance_experiment1_id-rag_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        with open(filename, 'w', newline='') as csvfile:
            fieldnames = ['agent', 'quiz_time', 'question', 'query_time', 'success', 'timestamp']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for entry in quiz_performance_data:
                writer.writerow(entry)
        
        print(f"✓ Quiz graph performance metrics saved to {filename}")
        
        # Print summary statistics
        total_quiz_queries = len(quiz_performance_data)
        successful_quiz_queries = sum(1 for entry in quiz_performance_data if entry['success'])
        avg_quiz_query_time = sum(entry['query_time'] for entry in quiz_performance_data) / total_quiz_queries if total_quiz_queries > 0 else 0
        success_rate = successful_quiz_queries / total_quiz_queries * 100 if total_quiz_queries > 0 else 0
        
        print(f"Quiz Graph Performance Summary:")
        print(f"  Total quiz queries: {total_quiz_queries}")
        print(f"  Successful quiz queries: {successful_quiz_queries} ({success_rate:.1f}%)")
        print(f"  Average quiz query time: {avg_quiz_query_time:.2f}s")
        
        # Per-agent breakdown
        for agent_name in ["Alice", "Bob"]:
            agent_data = [entry for entry in quiz_performance_data if entry['agent'] == agent_name]
            if agent_data:
                agent_successful = sum(1 for entry in agent_data if entry['success'])
                agent_avg_time = sum(entry['query_time'] for entry in agent_data) / len(agent_data)
                print(f"  {agent_name}: {len(agent_data)} queries, {agent_successful} successful, {agent_avg_time:.2f}s avg")
    else:
        print("No quiz graph performance data collected")

def augment_working_memory_with_networkx(working_memory: str, knowledge_graph, agent_name: str, model, remove_static_identity_before_kg_search: bool = True) -> str:
    """
    Transform working memory by replacing static identity with NetworkX KG-retrieved information.
    
    Args:
        working_memory: The full context string from ConcatActComponent
        knowledge_graph: NetworkX graph instance
        agent_name: Name of the agent (Alice or Bob only)
        model: LLM model for generating search strategy
        remove_static_identity_before_kg_search: If True, removes static identity before passing 
            to LLM for graph search. If False, passes full working memory to LLM. 
            Regardless of this flag, static identity is ALWAYS removed from final output.
    
    Returns:
        Augmented working memory with KG-retrieved identity (static identity always replaced)
    """
    
    if agent_name not in ["Alice", "Bob"]:
        return working_memory
    
    # Step 1: Parse the static identity section boundaries
    import re
    identity_start_pattern = r"Identity characteristics:"
    identity_end_pattern = r"feeling about recent progress in life:"
    
    # Find the boundaries
    identity_start_match = re.search(identity_start_pattern, working_memory)
    identity_end_match = re.search(identity_end_pattern, working_memory)
    
    if not identity_start_match or not identity_end_match:
        print(f"Warning: Could not find identity section boundaries in working memory for {agent_name}")
        return working_memory
    
    # Extract sections
    before_identity = working_memory[:identity_start_match.start()]
    identity_header = identity_start_match.group()
    static_identity_section = working_memory[identity_start_match.end():identity_end_match.start()]
    after_identity = working_memory[identity_end_match.start():]
    
    print(f"[NetworkX-KG] Found static identity section for {agent_name} (length: {len(static_identity_section)} chars)")
    
    # Step 2: Prepare context for search strategy generation based on flag
    if remove_static_identity_before_kg_search:
        # Remove static identity before passing to LLM for graph search
        context_for_search = before_identity + after_identity
        print(f"[NetworkX-KG] Using working memory WITHOUT static identity for graph search (length: {len(context_for_search)} chars)")
    else:
        # Keep full working memory (including static identity) for LLM graph search
        context_for_search = working_memory
        print(f"[NetworkX-KG] Using FULL working memory (including static identity) for graph search (length: {len(context_for_search)} chars)")
    
    # Step 3: Query NetworkX graph for relevant triplets
    kg_triplets = query_networkx_and_format(knowledge_graph, agent_name, context_for_search, model)
    
    # Step 4: Reconstruct working memory with KG information
    # Note: Static identity is ALWAYS removed from final output, regardless of search flag
    augmented_working_memory = (
        before_identity + 
        after_identity +
        identity_header + 
        kg_triplets
    )
    
    print(f"[NetworkX-KG] Added KG identity for {agent_name} (length: {len(kg_triplets)} chars)")
    return augmented_working_memory

def query_networkx_and_format(knowledge_graph, agent_name: str, context: str, model) -> str:
    """Query NetworkX graph and format results for working memory."""
    
    try:
        # Get relevant triplets from NetworkX graph
        relevant_triplets = query_graph_for_context(knowledge_graph, agent_name, context, model)
        
        print(f"[TRIPLET DEBUG] Retrieved for {agent_name}: {relevant_triplets}")

        # Format triplets into readable identity statements
        formatted_identity = format_triplets_for_identity(agent_name, relevant_triplets)
        
        print(f"[NetworkX-KG] Retrieved {len(relevant_triplets)} relevant triplets for {agent_name}")
        return formatted_identity
                
    except Exception as e:
        print(f"Error querying NetworkX graph for {agent_name}: {e}")
        # Return fallback
        if agent_name == "Bob":
            return f"{agent_name} is a progressive urban planner with 15 years of experience."
        else:
            return f"{agent_name} is a conservative urban planner with 20 years of experience." 