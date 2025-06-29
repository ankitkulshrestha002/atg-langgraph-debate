import os
import logging
from dotenv import load_dotenv
from graph import get_graph, format_history

# --- Setup Logging ---
# This ensures all messages are logged to a file as required.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("debate_log.txt", mode='w'),
        logging.StreamHandler()
    ]
)

def main():
    """Main function to run the debate simulation."""
    load_dotenv()
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        logging.error("FATAL: OPENAI_API_KEY environment variable not set.")
        return

    # --- User Input Node (handled at runtime before graph execution) ---
    print("--- Multi-Agent Debate System ---")
    topic = input("Enter the topic for the debate: ")
    logging.info(f"Debate Topic: {topic}")
    
    # Get the compiled graph
    app = get_graph()
    
    # Generate and save the DAG diagram
    try:
        app.get_graph().draw_mermaid_png(output_file_path="debate_dag.png")
        logging.info("DAG diagram saved to debate_dag.png")
    except Exception as e:
        logging.warning(f"Could not generate DAG diagram: {e}. Please ensure graphviz is installed.")


    # --- Execute the Debate ---
    initial_state = {
        "topic": topic,
        "messages": [],
        "round_number": 0,
        "next_speaker": "Scientist", # Scientist starts
    }
    
    print("\nStarting debate between Scientist and Philosopher...")
    logging.info("Starting debate flow...")

    final_state = None
    for event in app.stream(initial_state, {"recursion_limit": 15}):
        # The 'event' is a dictionary with the node name and its output
        node_name = list(event.keys())[0]
        node_output = list(event.values())[0]

        # Log state transitions
        logging.info(f"--- Executing Node: {node_name} ---")
        logging.info(f"Node Output: {node_output}")
        
        # Print agent arguments to the console in a clean format
        if node_name == "agent" and node_output.get("messages"):
            message = node_output["messages"][-1]
            speaker = message.name
            round_num = node_output["round_number"]
            print(f"[Round {round_num}] {speaker}: {message.content}")

        final_state = node_output

    # --- Display Final Judgment ---
    print("\n--- Debate Concluded ---")
    logging.info("--- DEBATE CONCLUDED ---")
    
    if final_state:
        summary = final_state.get("summary")
        winner = final_state.get("winner")
        justification = final_state.get("justification")
        
        print("\n[Judge] Summary of debate:")
        print(summary)
        
        print(f"\n[Judge] Winner: {winner}")
        print(f"[Judge] Reason: {justification}")
        
        logging.info(f"Final Summary: {summary}")
        logging.info(f"Winner: {winner}")
        logging.info(f"Justification: {justification}")
    
    print("\nFull debate log saved to debate_log.txt")

if __name__ == "__main__":
    main()