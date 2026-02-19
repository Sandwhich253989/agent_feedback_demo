from langgraph.types import Command

# from graph_agent_logging import compile_graph
from src.graph_agent_complex import compile_graph
# from graph_agent_selective_section_approval import compile_graph
from langgraph.types import Command
from src.utils.set_logging import logger


def main():
    """Run the progressive refinement agent"""

    # Compile the graph
    graph = compile_graph()

    # Initial state with configuration
    initial_state = {
        "prompt": "Write a comprehensive technical guide about Docker containerization for beginners",
        "messages": [],
        "sections": [],
        "high_confidence_sections": [],
        "review_req_sections": [],
        "approved_sections": [],
        "rejected_sections": [],
        "section_feedback": {},
        "section_rules": {},
        "mistakes": [],
        "revision_count": 0,
        "auto_approval_count": 0,
        "human_review_count": 0,
        "confidence_threshold": 0.8,  # Sections with confidence >= 0.8 are auto-approved
        "max_regen_attempts": 3,  # Maximum regeneration cycles
        "feedback": "",
        "output": ""
    }

    # Configuration for the graph
    config = {
        "configurable": {
            "thread_id": "test_1"
        }
    }

    logger.info("=" * 70)
    logger.info("[INFO] STARTING AGENT")
    logger.info("=" * 70)
    logger.info(f"[INFO] Task: {initial_state['prompt']}")
    logger.info(f"[INFO]  Confidence Threshold: {initial_state['confidence_threshold']}")
    logger.info(f"[INFO] Max Regeneration Attempts: {initial_state['max_regen_attempts']}")
    logger.info("=" * 70)

    # Run the graph with interrupts
    try:
        state = initial_state
        for event in graph.stream(state, config, stream_mode="updates"):
            logger.debug(f"[STREAM EVENT] {event}")

            # Handle human interrupts
            if "__interrupt__" in event:
                interrupt_data = event["__interrupt__"]

                for item in interrupt_data:
                    # Get the interrupt value (question + details)
                    interrupt_value = item.get("value", {})
                    question = interrupt_value.get("question", "")
                    details = interrupt_value.get("details", "")

                    # Display to user
                    if details:
                        print("\n" + details)

                    # Get user input
                    user_response = input(f"\n{question}").strip()

                    # Resume with user's response
                    state = graph.invoke(Command(resume=user_response), config)

        logger.info("\n[INFO] Agent execution completed successfully!")

    except KeyboardInterrupt:
        logger.warning("\n[INFO] Execution interrupted by user")
    except Exception as e:
        logger.error(f"\n[INFO] ❌ Error during execution: {e}", exc_info=True)


# if __name__ == "__main__":
#     main()

graph = compile_graph()

