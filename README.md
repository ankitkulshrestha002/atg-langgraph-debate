# Multi-Agent Debate DAG using LangGraph

This project is a submission for the ATG Machine Learning Intern technical assignment. It implements a multi-agent debate simulation system where two AI agents, a Scientist and a Philosopher, debate a user-provided topic for a fixed number of rounds. The debate is managed by a LangGraph Directed Acyclic Graph (DAG), and a final Judge node evaluates the arguments to declare a winner.

## Features

-   **Dynamic Topic:** The debate topic is provided by the user at runtime.
-   **Distinct Personas:** Two AI agents (Scientist and Philosopher) with unique perspectives.
-   **Structured Debate:** The debate is strictly controlled for 8 rounds (4 arguments per agent).
-   **State Management:** The graph state tracks the topic, message history, and current round.
-   **Conditional Routing:** A router node directs the flow, looping between agents and transitioning to the Judge node after 8 rounds.
-   **Automated Judging:** A dedicated Judge node reviews the entire debate, provides a summary, and declares a winner with a logical justification.
-   **Comprehensive Logging:** All agent messages, state transitions, and the final verdict are logged to `debate_log.txt`.
-   **CLI Interface:** The entire application runs through a clean command-line interface.

## Project Structure