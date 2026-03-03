"""Coordinator and worker agent logic for the iLiv travel planner."""

from __future__ import annotations

import asyncio
import copy
import json
import os
from dataclasses import dataclass, field
from typing import Any

from google import genai
from google.genai import types
from pydantic import BaseModel

from config import (
    COORDINATOR_MODEL,
    COORDINATOR_PARSE_SYSTEM,
    COORDINATOR_SYNTHESIZE_SYSTEM,
    CURRENT_YEAR,
    DEFAULT_ACTIVITIES_DAILY_USD,
    DEFAULT_FOOD_DAILY_USD,
    WORKER_MODEL,
    WORKER_PROMPTS,
)
from models import (
    AccommodationOptions,
    CoordinatorParseResponse,
    DestinationInfo,
    SafetyReport,
    TransportOptions,
    TravelPlan,
    TravelRequest,
    WorkerManifest,
    WorkerResult,
    WorkerTask,
    WorkerTaskType,
)
from tools import batch_search, format_search_results


# ── Event System ───────────────────────────────────────────────


@dataclass
class PlanEvent:
    """Structured event emitted during planning for live UI updates."""
    phase: str          # "parse", "worker", "synthesis", "done", "error"
    event_type: str     # "start", "complete", "fail", "status"
    agent_name: str = ""
    description: str = ""
    data: dict = field(default_factory=dict)


AGENT_LABELS: dict[str, str] = {
    "destination_research": "Destination Research",
    "accommodation": "Accommodation",
    "transport": "Transport",
    "safety": "Safety & Accessibility",
}


# ── Schema Cleaning (Gemini compatibility) ─────────────────────


def _clean_schema(schema: dict, defs: dict | None = None) -> dict:
    """Recursively strip properties unsupported by Gemini's structured output."""
    if defs is None:
        defs = schema.pop("$defs", {})

    if "$ref" in schema:
        ref_name = schema["$ref"].rsplit("/", 1)[-1]
        resolved = copy.deepcopy(defs.get(ref_name, {}))
        resolved.update({k: v for k, v in schema.items() if k != "$ref"})
        schema = resolved

    schema.pop("additionalProperties", None)
    schema.pop("title", None)
    schema.pop("default", None)

    if "anyOf" in schema:
        non_null = [s for s in schema["anyOf"] if s != {"type": "null"}]
        if len(non_null) == 1:
            resolved = _clean_schema(non_null[0], defs)
            resolved["nullable"] = True
            schema = resolved
        else:
            schema["anyOf"] = [_clean_schema(s, defs) for s in schema["anyOf"]]

    if "properties" in schema:
        schema["properties"] = {
            k: _clean_schema(copy.deepcopy(v), defs)
            for k, v in schema["properties"].items()
        }

    if "items" in schema:
        schema["items"] = _clean_schema(copy.deepcopy(schema["items"]), defs)

    return schema


def gemini_schema(model_class: type[BaseModel]) -> dict:
    raw = model_class.model_json_schema()
    return _clean_schema(raw)


# ── Client ─────────────────────────────────────────────────────

_client: genai.Client | None = None


def get_client() -> genai.Client:
    global _client
    if _client is None:
        api_key = os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY environment variable not set")
        _client = genai.Client(api_key=api_key)
    return _client


# ── Worker Dispatch ────────────────────────────────────────────

_HANDLERS: dict[WorkerTaskType, Any] = {}


def _handler(task_type: WorkerTaskType):
    def decorator(fn):
        _HANDLERS[task_type] = fn
        return fn
    return decorator


def _task_params(task: WorkerTask) -> dict[str, str]:
    return {p.key: p.value for p in task.parameters}


async def run_worker(
    task: WorkerTask,
    index: int = 0,
    event_queue: asyncio.Queue[PlanEvent] | None = None,
) -> WorkerResult:
    """Run a single worker, pushing events to the queue if provided."""
    agent_name = task.task_type.value
    label = AGENT_LABELS.get(agent_name, agent_name)

    if event_queue:
        await event_queue.put(PlanEvent(
            phase="worker",
            event_type="start",
            agent_name=agent_name,
            description=task.description,
            data={"index": index},
        ))

    handler = _HANDLERS.get(task.task_type)
    if not handler:
        result = WorkerResult(
            task_type=task.task_type, success=False, data={},
            error=f"Unknown task type: {task.task_type}",
        )
        if event_queue:
            await event_queue.put(PlanEvent(
                phase="worker", event_type="fail", agent_name=agent_name,
                description=f"{label}: unknown task type",
                data={"index": index},
            ))
        return result

    try:
        result = await handler(task)
        if event_queue:
            # Extract a short summary from the result data
            summary = _summarize_result(result)
            await event_queue.put(PlanEvent(
                phase="worker", event_type="complete", agent_name=agent_name,
                description=summary,
                data={"index": index},
            ))
        return result
    except Exception as e:
        result = WorkerResult(
            task_type=task.task_type, success=False, data={},
            error=f"{type(e).__name__}: {e}",
        )
        if event_queue:
            await event_queue.put(PlanEvent(
                phase="worker", event_type="fail", agent_name=agent_name,
                description=f"{type(e).__name__}",
                data={"index": index, "error": str(e)},
            ))
        return result


def _summarize_result(result: WorkerResult) -> str:
    """Create a short human-readable summary of a worker's findings."""
    d = result.data
    if result.task_type == WorkerTaskType.DESTINATION_RESEARCH:
        attractions = d.get("top_attractions", [])
        city = d.get("city", "?")
        return f"{city}: {len(attractions)} attractions found"
    elif result.task_type == WorkerTaskType.ACCOMMODATION:
        options = d.get("options", [])
        city = d.get("city", "?")
        if options:
            prices = [o.get("price_per_night_usd", 0) for o in options]
            return f"{city}: {len(options)} options (${min(prices):.0f}-${max(prices):.0f}/night)"
        return f"{city}: {len(options)} options found"
    elif result.task_type == WorkerTaskType.TRANSPORT:
        legs = d.get("legs", [])
        if legs:
            leg = legs[0]
            return f"{leg.get('from_city','?')} -> {leg.get('to_city','?')}: {leg.get('mode','?')}"
        return "Transport options found"
    elif result.task_type == WorkerTaskType.SAFETY:
        city = d.get("city", "?")
        rating = d.get("overall_safety_rating", "?")
        return f"{city}: safety rating {rating}/10"
    return "Complete"


# ── Workers ────────────────────────────────────────────────────


@_handler(WorkerTaskType.DESTINATION_RESEARCH)
async def _worker_destination(task: WorkerTask) -> WorkerResult:
    p = _task_params(task)
    city = p.get("city", "unknown")
    travel_period = p.get("travel_period", str(CURRENT_YEAR))
    origin = p.get("origin", "the US")
    interests_raw = p.get("interests", "")
    interests = [i.strip() for i in interests_raw.split(",") if i.strip()]

    queries = [
        f"{city} top attractions things to do {CURRENT_YEAR}",
        f"{city} travel guide tips best neighborhoods to stay",
        f"{city} weather climate {travel_period}",
    ]
    if interests:
        queries.append(f"{city} {' '.join(interests[:3])} {CURRENT_YEAR}")

    results = await batch_search(queries, max_results=3)
    context = format_search_results(results)

    prompt = WORKER_PROMPTS["destination_research"].format(
        city=city, travel_period=travel_period, origin=origin, context=context,
    )

    client = get_client()
    response = await client.aio.models.generate_content(
        model=WORKER_MODEL, contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=gemini_schema(DestinationInfo),
        ),
    )

    data = DestinationInfo.model_validate_json(response.text)
    return WorkerResult(task_type=WorkerTaskType.DESTINATION_RESEARCH, success=True, data=data.model_dump())


@_handler(WorkerTaskType.ACCOMMODATION)
async def _worker_accommodation(task: WorkerTask) -> WorkerResult:
    p = _task_params(task)
    city = p.get("city", "unknown")
    adults = int(p.get("adults", "2"))
    children = int(p.get("children", "0"))
    nights = int(p.get("nights", "3"))
    budget_per_night = p.get("budget_per_night", "flexible")
    currency = p.get("currency", "USD")
    preferences = p.get("preferences", "none specified")
    num_travelers = adults + children

    budget_str = f"${budget_per_night}" if budget_per_night.replace(".", "").isdigit() else budget_per_night
    queries = [
        f"best hotels {city} {budget_str} per night {CURRENT_YEAR}",
        f"{city} accommodation apartments family {CURRENT_YEAR} reviews",
    ]
    if children > 0:
        queries.append(f"{city} family friendly hotels kids {CURRENT_YEAR}")

    results = await batch_search(queries, max_results=3)
    context = format_search_results(results)

    prompt = WORKER_PROMPTS["accommodation"].format(
        city=city, num_travelers=num_travelers, adults=adults, children=children,
        nights=nights, budget_per_night=budget_str, currency=currency,
        preferences=preferences, context=context,
    )

    client = get_client()
    response = await client.aio.models.generate_content(
        model=WORKER_MODEL, contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=gemini_schema(AccommodationOptions),
        ),
    )

    data = AccommodationOptions.model_validate_json(response.text)
    return WorkerResult(task_type=WorkerTaskType.ACCOMMODATION, success=True, data=data.model_dump())


@_handler(WorkerTaskType.TRANSPORT)
async def _worker_transport(task: WorkerTask) -> WorkerResult:
    p = _task_params(task)
    from_city = p.get("from_city", "unknown")
    to_city = p.get("to_city", "unknown")
    travel_date = p.get("travel_date", str(CURRENT_YEAR))
    num_travelers = int(p.get("num_travelers", "2"))

    queries = [
        f"flights {from_city} to {to_city} {travel_date} price",
        f"{from_city} to {to_city} train bus transport options {CURRENT_YEAR}",
        f"getting around {to_city} public transport tips {CURRENT_YEAR}",
    ]

    results = await batch_search(queries, max_results=3)
    context = format_search_results(results)

    prompt = WORKER_PROMPTS["transport"].format(
        from_city=from_city, to_city=to_city,
        travel_date=travel_date, num_travelers=num_travelers, context=context,
    )

    client = get_client()
    response = await client.aio.models.generate_content(
        model=WORKER_MODEL, contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=gemini_schema(TransportOptions),
        ),
    )

    data = TransportOptions.model_validate_json(response.text)
    return WorkerResult(task_type=WorkerTaskType.TRANSPORT, success=True, data=data.model_dump())


@_handler(WorkerTaskType.SAFETY)
async def _worker_safety(task: WorkerTask) -> WorkerResult:
    p = _task_params(task)
    city = p.get("city", "unknown")
    country = p.get("country", "unknown")
    traveler_profile = p.get("traveler_profile", "general travelers")

    queries = [
        f"{city} {country} travel safety advisory {CURRENT_YEAR}",
        f"{city} areas to avoid tourist safety tips",
        f"{city} emergency numbers health requirements visitors",
    ]

    results = await batch_search(queries, max_results=3)
    context = format_search_results(results)

    prompt = WORKER_PROMPTS["safety"].format(
        city=city, country=country, traveler_profile=traveler_profile, context=context,
    )

    client = get_client()
    response = await client.aio.models.generate_content(
        model=WORKER_MODEL, contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=gemini_schema(SafetyReport),
        ),
    )

    data = SafetyReport.model_validate_json(response.text)
    return WorkerResult(task_type=WorkerTaskType.SAFETY, success=True, data=data.model_dump())


# ── Coordinator ────────────────────────────────────────────────


def _build_synthesis_context(
    request: TravelRequest,
    results: list[WorkerResult],
    user_feedback: str = "",
) -> str:
    sections = [f"## Original Travel Request\n{request.model_dump_json(indent=2)}"]

    for result in results:
        status = "SUCCESS" if result.success else f"FAILED: {result.error}"
        data_str = json.dumps(result.data, indent=2) if result.success else "No data available"
        sections.append(f"## Worker: {result.task_type.value} [{status}]\n{data_str}")

    sections.append(
        f"\n## Default Assumptions (use if not derivable from worker data)\n"
        f"- Estimated food cost per day: ${DEFAULT_FOOD_DAILY_USD}\n"
        f"- Estimated activities cost per day: ${DEFAULT_ACTIVITIES_DAILY_USD}\n"
    )

    if user_feedback.strip():
        sections.append(
            f"## User Feedback (IMPORTANT — prioritize these adjustments)\n"
            f"{user_feedback.strip()}"
        )

    return "\n\n".join(sections)


async def research_trip(
    user_query: str,
    event_queue: asyncio.Queue[PlanEvent],
) -> tuple[TravelRequest, list[WorkerTask], list[WorkerResult]]:
    """Phase 1+2: Parse query and run all research workers.

    Emits PlanEvents to event_queue as work progresses.
    Returns (request, tasks, results) for later synthesis.
    """
    client = get_client()

    # Phase 1: Parse
    await event_queue.put(PlanEvent(
        phase="parse", event_type="start",
        description="Analyzing your travel request...",
    ))

    parse_response = await client.aio.models.generate_content(
        model=COORDINATOR_MODEL,
        contents=user_query,
        config=types.GenerateContentConfig(
            system_instruction=COORDINATOR_PARSE_SYSTEM,
            response_mime_type="application/json",
            response_schema=gemini_schema(CoordinatorParseResponse),
        ),
    )

    parsed = CoordinatorParseResponse.model_validate_json(parse_response.text)
    request = parsed.travel_request
    manifest = parsed.worker_manifest

    await event_queue.put(PlanEvent(
        phase="parse", event_type="complete",
        description=f"Found {len(request.destinations)} destination(s): {', '.join(request.destinations)}",
        data={
            "destinations": request.destinations,
            "duration": request.duration_days,
            "budget": request.budget_total,
            "num_tasks": len(manifest.tasks),
        },
    ))

    # Phase 2: Workers
    await event_queue.put(PlanEvent(
        phase="worker", event_type="status",
        description=f"Dispatching {len(manifest.tasks)} research agents...",
        data={"total_tasks": len(manifest.tasks)},
    ))

    worker_coros = [
        run_worker(task, index=i, event_queue=event_queue)
        for i, task in enumerate(manifest.tasks)
    ]
    raw_results = await asyncio.gather(*worker_coros, return_exceptions=True)

    results: list[WorkerResult] = []
    for i, r in enumerate(raw_results):
        if isinstance(r, Exception):
            results.append(WorkerResult(
                task_type=manifest.tasks[i].task_type,
                success=False, data={}, error=f"{type(r).__name__}: {r}",
            ))
        else:
            results.append(r)

    succeeded = sum(1 for r in results if r.success)
    await event_queue.put(PlanEvent(
        phase="worker", event_type="status",
        description=f"Research complete: {succeeded}/{len(results)} agents succeeded",
        data={"succeeded": succeeded, "total": len(results)},
    ))

    return request, manifest.tasks, results


async def synthesize_trip(
    request: TravelRequest,
    results: list[WorkerResult],
    user_feedback: str = "",
    event_queue: asyncio.Queue[PlanEvent] | None = None,
) -> TravelPlan:
    """Phase 3: Synthesize research into a TravelPlan."""
    client = get_client()

    if event_queue:
        msg = "Building your personalized itinerary"
        if user_feedback.strip():
            msg += " (incorporating your feedback)"
        msg += "..."
        await event_queue.put(PlanEvent(
            phase="synthesis", event_type="start", description=msg,
        ))

    synthesis_context = _build_synthesis_context(request, results, user_feedback)

    last_error = None
    for attempt in range(3):
        try:
            synthesis_response = await client.aio.models.generate_content(
                model=COORDINATOR_MODEL,
                contents=synthesis_context,
                config=types.GenerateContentConfig(
                    system_instruction=COORDINATOR_SYNTHESIZE_SYSTEM,
                    response_mime_type="application/json",
                    response_schema=gemini_schema(TravelPlan),
                ),
            )
            plan = TravelPlan.model_validate_json(synthesis_response.text)
            if event_queue:
                await event_queue.put(PlanEvent(
                    phase="done", event_type="complete",
                    description="Itinerary ready!",
                ))
            return plan
        except Exception as e:
            last_error = e
            if attempt < 2:
                wait = (attempt + 1) * 5
                if event_queue:
                    await event_queue.put(PlanEvent(
                        phase="synthesis", event_type="fail",
                        description=f"Attempt {attempt+1} failed, retrying in {wait}s...",
                    ))
                await asyncio.sleep(wait)

    if event_queue:
        await event_queue.put(PlanEvent(
            phase="error", event_type="fail",
            description=f"Synthesis failed after 3 attempts: {last_error}",
        ))
    raise RuntimeError(f"Synthesis failed after 3 attempts: {last_error}")


async def plan_trip(
    user_query: str,
    on_status: Any | None = None,
) -> TravelPlan:
    """CLI entry point: runs the full pipeline (research + synthesize)."""
    event_queue: asyncio.Queue[PlanEvent] = asyncio.Queue()

    async def _drain_events():
        while True:
            try:
                event = event_queue.get_nowait()
                msg = event.description
                print(f"  {msg}")
                if on_status:
                    await on_status(msg)
            except asyncio.QueueEmpty:
                break

    request, tasks, results = await research_trip(user_query, event_queue)
    await _drain_events()

    plan = await synthesize_trip(request, results, event_queue=event_queue)
    await _drain_events()

    return plan
