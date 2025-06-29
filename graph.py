import os
from typing import TypedDict, Annotated, List
from langchain_core.messages import BaseMessage, AnyMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers.str import StrOutputParser
from langgraph.graph import StateGraph, END

# --- State Definition ---
# This class defines the state that will be passed between nodes in the graph.
class GraphState(TypedDict):
    topic: str
    messages: Annotated[List[AnyMessage], lambda x, y: x + y]
    round_number: int
    next_speaker: str
    winner: str
    justification: str
    summary: str

# --- LLM and Persona Setup ---
llm = ChatOpenAI(model="gpt-4o", temperature=0.7)

# Persona prompts
scientist_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a Scientist debating a topic. Your arguments should be evidence-based, logical, and grounded in scientific principles.
            Avoid emotional language and focus on data, research, and established theories.
            You are debating the topic: {topic}.
            The debate history is as follows:
            {history}
            Your opponent just made their argument. Now it is your turn.
            You are the Scientist. Make your next argument concisely (in 1-2 sentences). Do not repeat previous points.
            Directly state your argument without introductory phrases like "As a scientist...".
            """,
        ),
        ("user", "Your turn, Scientist."),
    ]
)

philosopher_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a Philosopher debating a topic. Your arguments should be based on logic, ethics, and philosophical frameworks.
            Explore the abstract, moral, and societal implications of the topic.
            You are debating the topic: {topic}.
            The debate history is as follows:
            {history}
            Your opponent just made their argument. Now it is your turn.
            You are the Philosopher. Make your next argument concisely (in 1-2 sentences). Do not repeat previous points.
            Directly state your argument without introductory phrases like "As a philosopher...".
            """,
        ),
        ("user", "Your turn, Philosopher."),
    ]
)

judge_prompt = ChatPromptTemplate.from_template(
    """You are a neutral Judge tasked with evaluating a debate between a Scientist and a Philosopher on the topic: '{topic}'.
    Below is the full transcript of the debate.

    {debate_history}

    Your task is to perform the following three actions:
    1.  Provide a neutral, one-paragraph summary of the entire debate.
    2.  Declare a winner. The winner must be either "Scientist" or "Philosopher".
    3.  Provide a clear, logical justification for your decision, explaining why the winner's arguments were more persuasive, coherent, or well-supported.

    Structure your output EXACTLY as follows, with each section on a new line:
    SUMMARY: [Your summary here]
    WINNER: [Scientist or Philosopher]
    JUSTIFICATION: [Your justification here]
    """
)

# --- Node Functions ---

def format_history(messages: List[AnyMessage]) -> str:
    """Formats the message history for the LLM prompt."""
    if not messages:
        return "The debate has not started yet."
    
    history_str = ""
    for i, msg in enumerate(messages):
        speaker = "Scientist" if i % 2 == 0 else "Philosopher"
        history_str += f"[Round {i+1}] {speaker}: {msg.content}\n"
    return history_str.strip()

def agent_node(state: GraphState):
    """Represents a single turn for either the Scientist or the Philosopher."""
    speaker = state["next_speaker"]
    prompt = scientist_prompt if speaker == "Scientist" else philosopher_prompt
    
    # Each agent only receives the history, not the full state, as required.
    chain = prompt | llm | StrOutputParser()
    
    response = chain.invoke({
        "topic": state["topic"],
        "history": format_history(state["messages"])
    })
    
    # State Validation: Simple check to discourage repetition
    if response in [msg.content for msg in state["messages"]]:
        response = "I will restate my previous point to emphasize its importance." # Fallback for repeated arg
        
    new_message = AnyMessage(content=response, name=speaker)
    
    next_speaker = "Philosopher" if speaker == "Scientist" else "Scientist"
    
    return {
        "messages": [new_message],
        "round_number": state["round_number"] + 1,
        "next_speaker": next_speaker,
    }

def judge_node(state: GraphState):
    """Evaluates the debate and declares a winner."""
    topic = state["topic"]
    history = format_history(state["messages"])
    
    chain = judge_prompt | llm | StrOutputParser()
    
    result = chain.invoke({"topic": topic, "debate_history": history})
    
    # Parse the structured output from the judge
    try:
        summary = result.split("SUMMARY:")[1].split("WINNER:")[0].strip()
        winner = result.split("WINNER:")[1].split("JUSTIFICATION:")[0].strip()
        justification = result.split("JUSTIFICATION:")[1].strip()
    except IndexError: # Fallback if LLM fails to follow format
        summary = "The judge failed to provide a structured summary."
        winner = "No winner declared"
        justification = "The judge's output was malformed."

    return {
        "summary": summary,
        "winner": winner,
        "justification": justification
    }

# --- Conditional Edge (Router) ---

def router(state: GraphState):
    """Determines the next node to execute based on the round number."""
    if state["round_number"] >= 8: # 8 rounds total (4 for each agent)
        return "judge"
    else:
        return "agent"

# --- Graph Definition ---

def get_graph():
    """Builds and compiles the LangGraph workflow."""
    workflow = StateGraph(GraphState)

    # Add nodes
    workflow.add_node("agent", agent_node)
    workflow.add_node("judge", judge_node)

    # Set entry point
    workflow.set_entry_point("agent")

    # Add conditional edges
    workflow.add_conditional_edges(
        "agent",
        router,
        {"agent": "agent", "judge": "judge"},
    )
    
    # The judge node is the final step
    workflow.add_edge("judge", END)

    # Compile the graph
    return workflow.compile()