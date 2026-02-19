from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.constants import END
from langgraph.graph import StateGraph
from langgraph.types import interrupt, Command
from src.utils.set_logging import logger
from src.utils.state import AgentState
from src.utils.tools import write_sections_to_doc
import json

# ---------------------------LOADING ENV----------------------------
load_dotenv(".env")


# ------------------------------------------------------------------

# ------------------------------------NODES---------------------------------------------------

def ai_generate_with_confidence(state: AgentState):
    """
    Node that generates content with sections and confidence scores.
    LLM self-assesses confidence for each section.
    """
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

    logger.info(f"[AGENT] Generating content with confidence assessment...")

    prompt = state["prompt"]
    learned_rules = ""
    section_specific_rules = state.get("section_rules", {})

    # Build learned rules
    if state.get("mistakes", []):
        learned_rules = "\n\nGlobal rules to follow:\n- " + "\n- ".join(state["mistakes"])

    # Build section-specific rules
    section_rules_text = ""
    if section_specific_rules:
        section_rules_text = "\n\nSection-specific rules:"
        for section_type, rules in section_specific_rules.items():
            section_rules_text += f"\n\nFor '{section_type}' sections:\n- " + "\n- ".join(rules)

    system_prompt = f"""You are an expert content generator for technical documentation.

Your task:
1. Analyze the user's request and identify logical sections for the document
2. Generate content for each section
3. Self-assess your confidence for each section (0.0 to 1.0 scale)

Confidence scoring guidelines:
- 0.9-1.0: Very clear requirements, straightforward content, high certainty
- 0.8-0.89: Clear requirements, minor ambiguity
- 0.7-0.79: Some ambiguity in requirements or complexity in content
- 0.5-0.69: Significant ambiguity or missing context
- Below 0.5: High uncertainty, major gaps in understanding

Respond ONLY with valid JSON in this exact format:
{{
    "sections": [
        {{
            "name": "Section Name",
            "content": "The actual content here...",
            "confidence": 0.85,
            "reasoning": "Why this confidence score"
        }}
    ]
}}

{learned_rules}
{section_rules_text}
"""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"User request: {prompt}")
    ]

    logger.debug(f"[DEBUG] Generating sections with confidence...")
    response = model.invoke(messages)

    try:
        # Parse JSON response
        response_text = response.content.strip()

        # Handle markdown code blocks
        if response_text.startswith("```"):
            response_text = response_text.split("```")[1]
            if response_text.startswith("json"):
                response_text = response_text[4:]

        result = json.loads(response_text)
        sections = result.get("sections", [])

        # Set threshold
        threshold = state.get("confidence_threshold", 0.8)

        # Categorize sections
        high_confidence = []
        review_required = []

        for section in sections:
            section_name = section["name"]
            confidence = section["confidence"]

            if confidence >= threshold:
                high_confidence.append(section_name)
                section["status"] = "auto_approved"
                logger.info(f"[AGENT] Section '{section_name}' auto-approved (confidence: {confidence:.2f})")
            else:
                review_required.append(section_name)
                section["status"] = "pending_review"
                logger.warning(f"[AGENT] Section '{section_name}' needs review (confidence: {confidence:.2f})")

        auto_count = len(high_confidence)
        total_count = len(sections)

        logger.info(
            f"[AGENT] Generated {total_count} sections: {auto_count} auto-approved, {total_count - auto_count} need review")

        return {
            "sections": sections,
            "high_confidence_sections": high_confidence,
            "review_req_sections": review_required,
            "auto_approval_count": state.get("auto_approval_count", 0) + auto_count,
            "messages": state.get("messages", []) + [AIMessage(content=f"Generated {total_count} sections")]
        }

    except json.JSONDecodeError as e:
        logger.error(f"[ERROR] Failed to parse JSON response: {e}")
        logger.error(f"[ERROR] Response was: {response.content}")
        # Fallback: treat as single section with low confidence
        return {
            "sections": [{
                "name": "Content",
                "content": response.content,
                "confidence": 0.5,
                "reasoning": "Failed to parse structured response",
                "status": "pending_review"
            }],
            "high_confidence_sections": [],
            "review_req_sections": ["Content"],
            "messages": state.get("messages", []) + [AIMessage(content=response.content)]
        }


def evaluate_sections(state: AgentState):
    """
    Routing node: Decides whether to proceed to finalization or human review.
    """
    review_required = state.get("review_req_sections", [])

    if not review_required:
        logger.info("[AGENT] All sections confident -> Proceeding to finalization")
        return Command(goto="finalize")
    else:
        logger.info(f"[AGENT] {len(review_required)} section(s) require human review")
        return Command(goto="human_selective_review")


def human_selective_review(state: AgentState):
    """
    Node that presents ONLY uncertain sections for human review.
    Shows auto-approved sections for context but doesn't require approval.
    """

    sections = state.get("sections", [])
    high_confidence = state.get("high_confidence_sections", [])
    review_required = state.get("review_req_sections", [])

    logger.info("[HUMAN] Showcasing sections for selective review...")

    # Build review prompt
    review_output = "\n" + "=" * 60 + "\n"
    review_output += "DOCUMENT SECTIONS GENERATED\n"
    review_output += "=" * 60 + "\n\n"

    # Show auto-approved sections (for context)
    if high_confidence:
        review_output += "AUTO-APPROVED SECTIONS (no review needed):\n"
        review_output += "-" * 60 + "\n"
        for section in sections:
            if section["name"] in high_confidence:
                review_output += f"\n▪ {section['name']} (Confidence: {section['confidence']:.2f})\n"
                review_output += f"  {section['content'][:150]}...\n"
        review_output += "\n"

    # Show sections needing review
    review_output += "SECTIONS REQUIRING YOUR REVIEW:\n"
    review_output += "-" * 60 + "\n"

    for section in sections:
        if section["name"] in review_required:
            review_output += f"\n Section: {section['name']}\n"
            review_output += f"   Confidence: {section['confidence']:.2f}\n"
            review_output += f"   Reason: {section.get('reasoning', 'N/A')}\n"
            review_output += f"\n   Content:\n   {section['content']}\n"
            review_output += "-" * 60 + "\n"

    # Ask for approval
    prompt = {
        "question": "Review complete. Approve all reviewed sections? (y/n): ",
        "details": review_output
    }

    response = interrupt(prompt)

    logger.error(f"[ERROR] Response was: {response}")
    if response.lower() == "y":
        logger.info("[HUMAN] All reviewed sections approved")
        # Mark all as approved
        for section in sections:
            if section["name"] in review_required:
                section["status"] = "human_reviewed"

        return Command(
            goto="finalize",
            update={
                "sections": sections,
                "approved_sections": state.get("approved_sections", []) + review_required,
                "human_review_count": state.get("human_review_count", 0) + len(review_required)
            }
        )

    # Collect section-specific feedback
    logger.warning("[HUMAN] Feedback requested")

    feedback_prompt = {
        "question": "Enter section name(s) to reject (comma-separated) or 'all': ",
        "details": review_output
    }
    rejected_sections_input = interrupt(feedback_prompt)

    if rejected_sections_input.lower() == "all":
        rejected = review_required
    else:
        rejected = [s.strip() for s in rejected_sections_input.split(",")]

    # Collect feedback for each rejected section
    section_feedback = state.get("section_feedback", {})

    for section_name in rejected:
        if section_name in review_required:
            feedback = interrupt({
                "question": f"What's wrong with section '{section_name}'? Provide feedback: ",
                "details": f""
            })
            section_feedback[section_name] = feedback

    approved = [s for s in review_required if s not in rejected]

    # Update status
    for section in sections:
        if section["name"] in approved:
            section["status"] = "human_reviewed"

    logger.info(f"[HUMAN] Approved: {len(approved)}, Rejected: {len(rejected)}")

    return Command(
        goto="reflect_and_learn",
        update={
            "sections": sections,
            "approved_sections": state.get("approved_sections", []) + approved,
            "rejected_sections": rejected,
            "section_feedback": section_feedback,
            "human_review_count": state.get("human_review_count", 0) + len(review_required)
        }
    )


def reflect_and_learn(state: AgentState):
    """
    Enhanced reflection node that extracts section-specific rules.
    """
    model = ChatOpenAI(model="gpt-4o-mini")

    logger.info("[AGENT] Reflecting on section-specific feedback...")

    section_feedback = state.get("section_feedback", {})
    rejected_sections = state.get("rejected_sections", [])
    sections = state.get("sections", [])

    new_global_mistakes = []
    section_rules = state.get("section_rules", {})

    for section_name in rejected_sections:
        if section_name in section_feedback:
            feedback = section_feedback[section_name]

            # Find section content
            section_content = ""
            for s in sections:
                if s["name"] == section_name:
                    section_content = s["content"]
                    break

            reflection_prompt = f"""
Analyze this feedback and extract rules.

Section: {section_name}
Generated content: {section_content}
Human feedback: {feedback}

Provide two types of rules:
1. GLOBAL RULE: A general rule applicable to all sections (if applicable)
2. SECTION-SPECIFIC RULE: A rule specific to '{section_name}' type sections

Format your response as:
GLOBAL: <rule or "NONE">
SPECIFIC: <rule or "NONE">

Rules should be imperative (e.g., "Use simpler language", "Include code examples").
"""

            response = model.invoke(reflection_prompt).content.strip()

            # Parse response
            lines = response.split("\n")
            for line in lines:
                if line.startswith("GLOBAL:"):
                    rule = line.replace("GLOBAL:", "").strip()
                    if rule and rule.upper() != "NONE":
                        new_global_mistakes.append(rule)
                        logger.info(f"[AGENT] New global rule learned: {rule}")

                elif line.startswith("SPECIFIC:"):
                    rule = line.replace("SPECIFIC:", "").strip()
                    if rule and rule.upper() != "NONE":
                        if section_name not in section_rules:
                            section_rules[section_name] = []
                        section_rules[section_name].append(rule)
                        logger.info(f"[AGENT] New rule for '{section_name}': {rule}")

    return Command(
        goto="regenerate_sections",
        update={
            "mistakes": state.get("mistakes", []) + new_global_mistakes,
            "section_rules": section_rules,
            "revision_count": state.get("revision_count", 0) + 1
        }
    )


def regenerate_sections(state: AgentState):
    """
    Regenerates ONLY rejected sections while preserving approved ones.
    """
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

    logger.info("[AGENT] Regenerating rejected sections...")

    rejected = state.get("rejected_sections", [])
    sections = state.get("sections", [])
    section_feedback = state.get("section_feedback", {})
    learned_rules = state.get("mistakes", [])
    section_rules = state.get("section_rules", {})

    max_attempts = state.get("max_regen_attempts", 3)
    current_revision = state.get("revision_count", 0)

    if current_revision >= max_attempts:
        logger.warning(f"[AGENT] Max regeneration attempts ({max_attempts}) reached. Escalating to human.")
        return Command(goto="human_selective_review")

    # Regenerate each rejected section
    for section_name in rejected:
        feedback = section_feedback.get(section_name, "")

        # Get section-specific rules
        specific_rules = section_rules.get(section_name, [])

        # Find original section
        original_section = None
        for s in sections:
            if s["name"] == section_name:
                original_section = s
                break

        if not original_section:
            continue

        regeneration_prompt = f"""
Regenerate the content for section: {section_name}

Original content:
{original_section['content']}

Human feedback:
{feedback}

Global rules to follow:
{chr(10).join('- ' + rule for rule in learned_rules) if learned_rules else '- None'}

Section-specific rules for '{section_name}':
{chr(10).join('- ' + rule for rule in specific_rules) if specific_rules else '- None'}

Respond with JSON:
{{
    "content": "regenerated content here",
    "confidence": 0.XX,
    "reasoning": "why this confidence"
}}
"""

        response = model.invoke(regeneration_prompt)

        try:
            response_text = response.content.strip()
            if response_text.startswith("```"):
                response_text = response_text.split("```")[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:]

            result = json.loads(response_text)

            # Update section
            original_section["content"] = result["content"]
            original_section["confidence"] = result["confidence"]
            original_section["reasoning"] = result.get("reasoning", "Regenerated based on feedback")

            # Re-evaluate confidence
            threshold = state.get("confidence_threshold", 0.8)
            if result["confidence"] >= threshold:
                original_section["status"] = "auto_approved"
                logger.info(f"[AGENT]  Regenerated '{section_name}' now confident ({result['confidence']:.2f})")
            else:
                original_section["status"] = "pending_review"
                logger.warning(
                    f"[AGENT] Regenerated '{section_name}' still needs review ({result['confidence']:.2f})")

        except json.JSONDecodeError as e:
            logger.error(f"[ERROR] Failed to parse regeneration response for '{section_name}'")
            original_section["status"] = "pending_review"

    # Re-categorize sections
    high_confidence = []
    review_required = []
    threshold = state.get("confidence_threshold", 0.8)

    for section in sections:
        if section["confidence"] >= threshold and section["status"] != "human_reviewed":
            high_confidence.append(section["name"])
            if section["status"] != "human_reviewed":
                section["status"] = "auto_approved"
        elif section["status"] == "pending_review":
            review_required.append(section["name"])

    logger.info(f"[AGENT] After regeneration: {len(high_confidence)} confident, {len(review_required)} need review")

    return Command(
        goto="evaluate_sections",
        update={
            "sections": sections,
            "high_confidence_sections": high_confidence,
            "review_req_sections": review_required,
            "rejected_sections": [],  # Clear rejected list
            "section_feedback": {}  # Clear feedback
        }
    )


def finalize(state: AgentState):
    """
    Final node that writes the approved document.
    """
    logger.info("[AGENT] Writing to document...")

    sections = state.get("sections", [])
    prompt = state.get("prompt", "Generated Document")

    # Extract title from prompt or use default
    title = prompt[:50] if len(prompt) > 50 else prompt

    # Write document using enhanced tool
    result = write_sections_to_doc.invoke({
        "title": title,
        "sections": sections
    })

    # Log statistics
    total = len(sections)
    auto_approved = state.get("auto_approval_count", 0)
    human_reviewed = state.get("human_review_count", 0)
    revisions = state.get("revision_count", 0)

    logger.info("="*60)
    logger.info("[AGENT] Document Generation Complete!")
    logger.info(f"[STATS] Total Sections: {total}")
    logger.info(f"[STATS] Auto-approved: {auto_approved}")
    logger.info(f"[STATS] Human-reviewed: {human_reviewed}")
    logger.info(f"[STATS] Revision cycles: {revisions}")
    logger.info(f"[STATS] Automation rate: {(auto_approved / total * 100):.1f}%")
    logger.info(f"[RESULT] {result}")
    logger.info("=" * 60)

    return {
        "output": f"Document completed with {total} sections. {result}"
    }


# --------------------------------------------------------------------------------------------

def compile_graph():
    """Method to compile the progressive refinement graph"""
    checkpointer = InMemorySaver()
    builder = StateGraph(AgentState)

    # Add nodes
    builder.add_node("ai_generate_with_confidence", ai_generate_with_confidence)
    builder.add_node("evaluate_sections", evaluate_sections)
    builder.add_node("human_selective_review", human_selective_review)
    builder.add_node("reflect_and_learn", reflect_and_learn)
    builder.add_node("regenerate_sections", regenerate_sections)
    builder.add_node("finalize", finalize)

    # Add edges
    builder.add_edge("ai_generate_with_confidence", "evaluate_sections")
    builder.add_edge("regenerate_sections", "evaluate_sections")
    builder.add_edge("finalize",END)

    # Set entry point
    builder.set_entry_point("ai_generate_with_confidence")

    # Compile
    graph = builder.compile()

    return graph
