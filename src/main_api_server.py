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


# from typing import Dict, Any, Optional
# from uuid import uuid4
# import asyncio
#
# from fastapi import FastAPI, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import StreamingResponse
# from pydantic import BaseModel, Field
#
# from utils.set_logging import logger
# from graph_agent_complex import compile_graph
# from dotenv import load_dotenv
#
# load_dotenv("../.env")
# app = FastAPI(title="LangGraph Agent API", version="1.0.0")
#
# # CORS middleware for frontend integration
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Adjust for production
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )
#
# # In-memory storage for active sessions (use Redis/DB in production)
# active_threads = {}
#
# # Compiled graph
# graph = compile_graph()
#
# # ==================== REQUEST/RESPONSE MODELS ====================
#
# class StartAgentRequest(BaseModel):
#     prompt: str = Field(..., description="User's task description")
#     confidence_threshold: float = Field(0.8, ge=0.0, le=1.0, description="Confidence threshold for auto-approval")
#     max_regen_attempts: int = Field(3, ge=1, le=10, description="Maximum regeneration attempts")
#     thread_id: Optional[str] = Field(None, description="Optional thread ID for resuming")
#
#
# class InterruptResponse(BaseModel):
#     response: str = Field(..., description="User's response to the interrupt")
#
#
# class ThreadStatus(BaseModel):
#     thread_id: str
#     status: str
#     current_node: Optional[str] = None
#     interrupt_data: Optional[Dict[str, Any]] = None
#
#
# # ==================== ENDPOINTS ====================
#
# @app.post('/agent/start')
# async def start_agent(request: StartAgentRequest):
#     """
#     Start a new agent execution or resume an existing one.
#     Returns thread_id for tracking.
#     """
#     thread_id = request.thread_id or f"thread_{uuid4()}"
#
#     logger.info("=" * 70)
#     logger.info(f"[API] Starting agent for thread: {thread_id}")
#     logger.info(f"[API] Task: {request.prompt}")
#     logger.info(f"[API] Confidence Threshold: {request.confidence_threshold}")
#     logger.info(f"[API] Max Regeneration Attempts: {request.max_regen_attempts}")
#     logger.info("=" * 70)
#
#     # Initial state
#     initial_state = {
#         "prompt": request.prompt,
#         "confidence_threshold": request.confidence_threshold,
#         "max_regen_attempts": request.max_regen_attempts,
#         "messages": [],
#         "mistakes": [],
#         "revision_count": 0,
#         "sections": [],
#         "high_confidence_sections": [],
#         "review_req_sections": [],
#         "approved_sections": [],
#         "rejected_sections": [],
#         "section_feedback": {},
#         "section_rules": {},
#         "auto_approval_count": 0,
#         "human_review_count": 0,
#         "output": "",
#     }
#
#     # Store thread info
#     active_threads[thread_id] = {
#         "status": "running",
#         "initial_state": initial_state
#     }
#
#     return {
#         "thread_id": thread_id,
#         "message": "Agent started successfully",
#         "status": "running"
#     }
#
#
# @app.get('/agent/stream/{thread_id}')
# async def stream_agent_events(thread_id: str):
#     """
#     Stream agent execution events using Server-Sent Events (SSE).
#     Handles interrupts by pausing the stream and waiting for user input.
#     """
#     if thread_id not in active_threads:
#         raise HTTPException(status_code=404, detail="Thread not found")
#
#     async def event_generator():
#         try:
#             config = {
#                 "configurable": {
#                     "thread_id": thread_id
#                 }
#             }
#
#
#             # Check if we need to resume from checkpoint or start fresh
#             state_snapshot = graph.get_state(config)
#             has_checkpoint = bool(state_snapshot.values)
#
#             if has_checkpoint and state_snapshot.next:
#                 # Resume from checkpoint - there's existing state and pending nodes
#                 logger.info(f"[API] Resuming thread {thread_id} from checkpoint")
#                 stream_input = None  # Pass None to resume from checkpoint
#             else:
#                 # Fresh start - no checkpoint exists
#                 logger.info(f"[API] Starting fresh execution for thread {thread_id}")
#                 stream_input = active_threads[thread_id]["initial_state"]
#
#             # Stream events
#             for event in graph.stream(stream_input, config=config, stream_mode="values"):
#                 # Check if there's an interrupt
#                 state_snapshot = graph.get_state(config)
#
#                 if state_snapshot.next:  # Has pending nodes
#                     # Send current state
#                     yield f"data: {{'type': 'state_update', 'data': {event}}}\n\n"
#
#                     # Check for interrupt
#                     if hasattr(state_snapshot, 'tasks') and state_snapshot.tasks:
#                         for task in state_snapshot.tasks:
#                             if task.interrupts:
#                                 # Store interrupt info
#                                 interrupt_data = task.interrupts[0].value
#                                 active_threads[thread_id]["status"] = "awaiting_input"
#                                 active_threads[thread_id]["interrupt_data"] = interrupt_data
#
#                                 # Notify frontend about interrupt
#                                 yield f"data: {{'type': 'interrupt', 'data': {interrupt_data}}}\n\n"
#                                 return  # Pause stream, wait for user input
#                 else:
#                     # Final state
#                     yield f"data: {{'type': 'final', 'data': {event}}}\n\n"
#                     active_threads[thread_id]["status"] = "completed"
#
#                 await asyncio.sleep(0.1)  # Prevent overwhelming the client
#
#             yield f"data: {{'type': 'done'}}\n\n"
#
#         except Exception as e:
#             logger.error(f"[API] Error in stream: {str(e)}")
#             active_threads[thread_id]["status"] = "error"
#             yield f"data: {{'type': 'error', 'message': '{str(e)}'}}\n\n"
#
#     return StreamingResponse(
#         event_generator(),
#         media_type="text/event-stream",
#         headers={
#             "Cache-Control": "no-cache",
#             "Connection": "keep-alive",
#         }
#     )
#
#
# @app.post('/agent/respond/{thread_id}')
# async def respond_to_interrupt(thread_id: str, response: InterruptResponse):
#     """
#     Send user's response to an interrupt and resume execution.
#     """
#     if thread_id not in active_threads:
#         raise HTTPException(status_code=404, detail="Thread not found")
#
#     thread_info = active_threads[thread_id]
#
#     if thread_info["status"] != "awaiting_input":
#         raise HTTPException(status_code=400, detail="Thread is not awaiting input")
#
#     logger.info(f"[API] Received interrupt response for thread {thread_id}: {response.response}")
#
#     try:
#         config = {
#             "configurable": {
#                 "thread_id": thread_id
#             }
#         }
#
#         # Resume with user's response - update_state resumes the interrupt
#         graph.update_state(config, response.response, as_node="human_selective_review")
#
#         # Update thread status
#         active_threads[thread_id]["status"] = "running"
#         active_threads[thread_id].pop("interrupt_data", None)
#
#         return {
#             "message": "Response received, agent resuming",
#             "thread_id": thread_id,
#             "status": "running"
#         }
#
#     except Exception as e:
#         logger.error(f"[API] Error responding to interrupt: {str(e)}")
#         raise HTTPException(status_code=500, detail=str(e))
#
#
# @app.get('/agent/status/{thread_id}')
# async def get_thread_status(thread_id: str):
#     """
#     Get the current status of a thread.
#     """
#     if thread_id not in active_threads:
#         raise HTTPException(status_code=404, detail="Thread not found")
#
#     thread_info = active_threads[thread_id]
#
#     return ThreadStatus(
#         thread_id=thread_id,
#         status=thread_info["status"],
#         interrupt_data=thread_info.get("interrupt_data")
#     )
#
#
