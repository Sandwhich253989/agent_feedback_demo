from typing import TypedDict, List, Annotated, Optional

from langgraph.graph import add_messages


#--------------------STATE----------------------
class AgentState(TypedDict):
    # logs: List[dict]
    prompt: str
    output: str
    messages: Annotated[List,add_messages]

    feedback: str
    mistakes: List[str]
    revision_count: int
    # doc_path: str
    #
    # #document inspection
    # doc_outline: List[str]
    # selected_heading: str
    # selected_content: str
    #
    # # pending edits
    # pending_heading: str
    # pending_content: str
    #
    # action: Optional[str]

    sections: List[dict]
    high_confidence_sections: List[str]
    review_req_sections: List[str]
    review_phase: str
    approved_sections: List[str]
    rejected_sections: List[str]
    section_feedback: dict
    section_rules: dict



    auto_approval_count: int
    human_review_count: int
    confidence_threshold: float
    max_regen_attempts: int

#-----------------------------------------------
