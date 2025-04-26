import json
from pyvis.network import Network

def visualize_subgraph(subgraph_path, output_html="subgraphs/subgraph_visualization.html"):
    # Load the subgraph JSON
    with open(subgraph_path, "r") as f:
        subgraph = json.load(f)

    nodes = subgraph["nodes"]
    relationships = subgraph["relationships"]

    # Create a PyVis network
    net = Network(height="800px", width="100%", notebook=False, directed=True)
    net.barnes_hut()

    # Add nodes
    for node in nodes:
        node_id = node["element_id"]
        labels = node.get("labels", [])
        props = node.get("properties", {})
        name = props.get("name", "Unnamed")

        label = f"{name}\n{','.join(labels)}"
        net.add_node(node_id, label=label, title=json.dumps(props, indent=2))

    # Add relationships (edges)
    for rel in relationships:
        source = rel["start_node_element_id"]
        target = rel["end_node_element_id"]
        rel_type = rel["type"]
        properties = rel.get("properties", {})

        # Add relationship label on the edge, and tooltip with full properties
        net.add_edge(
            source,
            target,
            label=rel_type, 
            title=json.dumps(properties, indent=2),
            arrows="to",
            length=25
        )

    # Generate HTML file
    net.show(output_html, notebook=False)
    print(f"Subgraph visualization saved to {output_html}")

if __name__ == "__main__":
    visualize_subgraph("subgraph_20250426_161843.json", output_html="subgraphs/subgraph_visualization.html")
