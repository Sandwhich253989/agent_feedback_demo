import asyncio
from fastapi import FastAPI, APIRouter, HTTPException
from fastapi.responses import StreamingResponse
# from langfuse import Langfuse, get_client
# from langfuse.langchain import CallbackHandler
from pydantic import BaseModel
from typing import Dict, Any
import uuid
import json

from langgraph.types import Command
from graph_agent_complex import compile_graph
from utils.set_logging import logger

from dotenv import load_dotenv

# ----------------------------------CONFIGS-------------------------------------
load_dotenv("../.env")

# Example Secret keys (Not Valid)
# Langfuse(
#     public_key="pk-lf-bae3dcd6-1356-4f30-87c7-0c104c50d596",
#     secret_key="sk-lf-74281145-630f-49fd-868f-b66d7923a729",
#     host="http://localhost:3000/"  # Optional: defaults to https://cloud.langfuse.com
# )
# # Get the configured client instance
# langfuse = get_client()
#
# # Initialize the Langfuse handler
# langfuse_handler = CallbackHandler()
app = FastAPI(title="LangGraph Agent API")
router = APIRouter(prefix="/agent", tags=["agent"])

graph = compile_graph()
# ------------------------------------------------------------------------------


# ----------------------------THREADING/PERSISTENCE-----------------------------
# In-memory thread registry
THREADS: Dict[str, Dict[str, Any]] = {}


# Replace with Database
# ------------------------------------------------------------------------------


# ---------------------------------SCHEMAS--------------------------------------
class StartAgentRequest(BaseModel):
    prompt: str
    confidence_threshold: float = 0.8
    max_regen_attempts: int = 3


class RespondRequest(BaseModel):
    response: str


# ------------------------------------------------------------------------------


# -------------------------HELPER FUNCTIONS-------------------------------------
def default_initial_state(prompt: str, confidence_threshold: float, max_regen_attempts: int):
    return {
        "prompt": prompt,
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
        "confidence_threshold": confidence_threshold,
        "max_regen_attempts": max_regen_attempts,
        "feedback": "",
        "output": ""
    }


def graph_config(thread_id: str):
    return {
        "configurable": {
            "thread_id": thread_id
        }
        # "callbacks":[langfuse_handler]
    }


# ------------------------------------------------------------------------------


# ----------------------------------APIs-----------------------------------------


# ------------------------------------------------------------------------------
# 1. POST /agent/start
# ------------------------------------------------------------------------------

@router.post("/start")
async def start_agent(req: StartAgentRequest):
    thread_id = str(uuid.uuid4())

    logger.info("=" * 70)
    logger.info(f"[API] Starting agent for thread: {thread_id}")
    logger.info(f"[API] Task: {req.prompt}")
    logger.info(f"[API] Confidence Threshold: {req.confidence_threshold}")
    logger.info(f"[API] Max Regeneration Attempts: {req.max_regen_attempts}")
    logger.info("=" * 70)

    initial_state = default_initial_state(
        req.prompt,
        req.confidence_threshold,
        req.max_regen_attempts
    )

    THREADS[thread_id] = {
        "status": "created",
        "initial_state": initial_state,
        "pending_resume": None,
    }

    logger.info(f"[AGENT] Created thread {thread_id}")

    return {
        "thread_id": thread_id,
        "status": "created",
        "message": "Agent thread initialized"
    }


# ------------------------------------------------------------------------------
# 2. GET /agent/stream/{thread_id}
#    - Starts OR resumes execution
#    - Stops on interrupt
# ------------------------------------------------------------------------------

@router.get("/stream/{thread_id}")
async def stream_agent(thread_id: str):
    if thread_id not in THREADS:
        raise HTTPException(status_code=404, detail="Invalid thread_id")

    thread = THREADS[thread_id]
    config = graph_config(thread_id)

    if thread["status"] == "completed":
        return {"status": "completed", "details": "Agent has already completed its execution!"}

    async def event_stream():
        try:
            logger.info(f"[AGENT] Streaming execution for {thread_id}")

            if thread["status"] == "created":
                input_state = thread["initial_state"]
                thread["started"] = True

            elif thread["status"] == "ready_to_resume":
                input_state = Command(resume=thread["pending_resume"])
                thread["pending_resume"] = None
                thread["status"] = "running"

            else:
                input_state = None  # normal resume
            # IMPORTANT:
            # graph.stream() will automatically resume from checkpoint
            async for event in graph.astream(input_state, config, stream_mode="updates"):

                # Interrupt detected → return + pause execution
                if "__interrupt__" in event:
                    interrupt_payload = []

                    interrupt_payload.append({
                        "question": event["__interrupt__"][0].value["question"],
                        "details": event["__interrupt__"][0].value["details"],
                    })
                    # for item in event["__interrupt__"]:
                    #     value = item.get("value", {})
                    #     interrupt_payload.append({
                    #         "question": value.get("question"),
                    #         "details": value.get("details")
                    #     })

                    yield json.dumps({
                        "type": "interrupt",
                        "data": interrupt_payload
                    }) + "\n"

                    THREADS[thread_id]["status"] = "waiting_for_user"
                    return

                # Normal update
                yield json.dumps({
                    "type": "update",
                    "data": [e[0] for e in event.items()],
                }) + "\n"

                await asyncio.sleep(0.1)  # Prevent overwhelming the client

            # Completed
            THREADS[thread_id]["status"] = "completed"
            yield json.dumps({
                "type": "done",
                "message": "Agent execution completed"
            }) + "\n"

        except Exception as e:
            logger.error(f"[AGENT] Error in stream: {e}", exc_info=True)
            yield json.dumps({
                "type": "error",
                "message": str(e)
            }) + "\n"

    return StreamingResponse(event_stream(), media_type="application/json", headers={
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
    })


# ------------------------------------------------------------------------------
# 4. POST /agent/respond/{thread_id}
#    - Resume from interrupt (NO restart)
# ------------------------------------------------------------------------------

@router.post("/respond/{thread_id}")
async def respond_to_agent(thread_id: str, req: RespondRequest):
    if thread_id not in THREADS:
        raise HTTPException(status_code=404, detail="Invalid thread_id")

    logger.info(f"[AGENT] Resuming thread {thread_id} with user input")

    try:

        THREADS[thread_id]["pending_resume"] = req.response
        THREADS[thread_id]["status"] = "ready_to_resume"

        return {
            "status": "ready_to_resume",
            "details": "response sent to agent! use /stream/{thread_id} to continue execution",
            "thread_id": thread_id
        }

    except Exception as e:
        logger.error(f"[AGENT] Resume failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
# Mount router
# ------------------------------------------------------------------------------

app.include_router(router)


@app.get('/agent/state/{thread_id}')
async def get_thread_state(thread_id: str):
    """
    Get the full state of a thread.
    """
    if thread_id not in THREADS:
        raise HTTPException(status_code=404, detail="Thread not found")

    try:
        config = {
            "configurable": {
                "thread_id": thread_id
            }
        }

        state_snapshot = graph.get_state(config)

        return {
            "thread_id": thread_id,
            "values": state_snapshot.values,
            "next_nodes": state_snapshot.next,
            "metadata": state_snapshot.metadata
        }

    except Exception as e:
        logger.error(f"[API] Error getting state: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get('/')
async def check():
    """
    Health check endpoint.
    """
    return {"status": "healthy", "active_threads": len(THREADS)}


if __name__ == "__main__":
    import uvicorn

    app.include_router(router)
    uvicorn.run(app, host="127.0.0.1", port=8000)
