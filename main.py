"""iLiv Travel Planner — Multi-Agent Travel Planning System.

Usage:
    python main.py --web                    # Launch Gradio web UI
    python main.py "Plan a trip to Tokyo"   # CLI mode
    python main.py                          # Interactive CLI mode
    python main.py --json "..."             # Output raw JSON
"""

from __future__ import annotations

import asyncio
import os
import sys

from dotenv import load_dotenv

load_dotenv()

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

from agents import (
    AGENT_LABELS,
    PlanEvent,
    plan_trip,
    research_trip,
    synthesize_trip,
)
from models import TravelPlan


def format_plan_markdown(plan: TravelPlan) -> str:
    lines: list[str] = []

    lines.append(f"# {plan.title}")
    lines.append("")
    dest_str = ", ".join(plan.destinations)
    t = plan.travelers
    travelers_str = f"{t.adults} adult(s)"
    if t.children:
        ages = ", ".join(str(a) for a in t.child_ages) if t.child_ages else "?"
        travelers_str += f", {t.children} child(ren) (ages {ages})"
    lines.append(f"**Destinations:** {dest_str}")
    lines.append(f"**Duration:** {plan.duration_days} days")
    lines.append(f"**Travelers:** {travelers_str}")
    lines.append(f"**Estimated Total Cost:** ${plan.cost_breakdown.grand_total:,.0f} {plan.cost_breakdown.currency}")
    lines.append("")
    lines.append(plan.summary)
    lines.append("")
    lines.append("---")
    lines.append("")

    lines.append("## Day-by-Day Itinerary")
    lines.append("")
    for day in plan.itinerary:
        date_str = f" - {day.date}" if day.date else ""
        lines.append(f"### Day {day.day_number}{date_str} | {day.city}")
        lines.append(f"**Morning:** {day.morning}")
        lines.append(f"**Afternoon:** {day.afternoon}")
        lines.append(f"**Evening:** {day.evening}")
        if day.meals:
            lines.append(f"**Meals:** {', '.join(day.meals)}")
        lines.append(f"**Estimated daily cost:** ${day.estimated_cost:,.0f}")
        if day.notes:
            for note in day.notes:
                lines.append(f"- {note}")
        lines.append("")

    lines.append("---")
    lines.append("")

    if plan.accommodation_plan:
        lines.append("## Accommodation Plan")
        lines.append("")
        lines.append("| Property | Type | City/Area | $/Night | Rating |")
        lines.append("|----------|------|-----------|---------|--------|")
        for acc in plan.accommodation_plan:
            rating = f"{acc.rating}/5" if acc.rating else "N/A"
            lines.append(
                f"| {acc.name} | {acc.type} | {acc.neighborhood} "
                f"| ${acc.price_per_night_usd:,.0f} | {rating} |"
            )
        lines.append("")

    if plan.transport_plan:
        lines.append("## Transportation Plan")
        lines.append("")
        lines.append("| Route | Mode | Duration | Est. Cost | Provider |")
        lines.append("|-------|------|----------|-----------|----------|")
        for leg in plan.transport_plan:
            provider = leg.provider or "Various"
            lines.append(
                f"| {leg.from_city} -> {leg.to_city} | {leg.mode}"
                f"| {leg.duration_hours:.1f}h | ${leg.price_usd:,.0f} | {provider} |"
            )
        lines.append("")

    cb = plan.cost_breakdown
    lines.append("## Cost Breakdown")
    lines.append("")
    lines.append(f"| Category | Amount ({cb.currency}) |")
    lines.append("|----------|--------|")
    lines.append(f"| Accommodation | ${cb.accommodation_total:,.0f} |")
    lines.append(f"| Transportation | ${cb.transport_total:,.0f} |")
    lines.append(f"| Food (~${cb.food_estimated_daily:,.0f}/day) | ${cb.food_total:,.0f} |")
    lines.append(f"| Activities (~${cb.activities_estimated_daily:,.0f}/day) | ${cb.activities_total:,.0f} |")
    lines.append(f"| Miscellaneous | ${cb.miscellaneous:,.0f} |")
    lines.append(f"| **Total** | **${cb.grand_total:,.0f}** |")
    if cb.budget_remaining is not None:
        label = "Remaining" if cb.budget_remaining >= 0 else "Over budget"
        lines.append(f"| {label} | ${abs(cb.budget_remaining):,.0f} |")
    lines.append("")

    if cb.cost_saving_tips:
        lines.append("**Cost-Saving Tips:**")
        for tip in cb.cost_saving_tips:
            lines.append(f"- {tip}")
        lines.append("")

    if plan.safety_notes:
        lines.append("## Safety Notes")
        lines.append("")
        for note in plan.safety_notes:
            lines.append(f"- {note}")
        lines.append("")

    if plan.packing_suggestions:
        lines.append("## Packing Suggestions")
        lines.append("")
        for item in plan.packing_suggestions:
            lines.append(f"- {item}")
        lines.append("")

    if plan.important_links:
        lines.append("## Useful Links")
        lines.append("")
        for link in plan.important_links:
            lines.append(f"- {link}")
        lines.append("")

    lines.append("---")
    lines.append("")
    if plan.disclaimers:
        for d in plan.disclaimers:
            lines.append(f"*{d}*")
    lines.append(
        "*This plan was generated using real-time web search data. "
        "Prices are estimates and may vary. Always verify bookings directly with providers.*"
    )
    lines.append("")

    return "\n".join(lines)


async def async_main(query: str, output_json: bool = False) -> None:
    for var in ("GEMINI_API_KEY", "TAVILY_API_KEY"):
        if not os.environ.get(var):
            print(f"Error: {var} environment variable is not set.")
            sys.exit(1)

    print("iLiv Travel Planner")
    print("=" * 50)
    print(f"\nQuery: {query}\n")
    print("Planning your trip... This typically takes 20-40 seconds.\n")

    plan = await plan_trip(query)

    if output_json:
        print(plan.model_dump_json(indent=2))
    else:
        print(format_plan_markdown(plan))


EXAMPLE_QUERIES = [
    "Plan a 5-day trip to Tokyo for 2 adults, budget $3000, interested in food and culture",
    "Solo female traveler visiting Paris and Barcelona for 7 days in June. Budget $4000. Love art, wine, and safe walkable neighborhoods.",
    "Family trip with 2 kids (ages 5 and 9) to Orlando and Miami for 10 days. Budget $6000. Need kid-friendly activities.",
    "Backpacking Southeast Asia for 3 weeks: Bangkok, Hanoi, and Bali. Budget $2500. Vegetarian.",
]


# ── Activity Log Renderer ──────────────────────────────────────


class ActivityLog:
    """Accumulates PlanEvents and renders a live Markdown activity log."""

    def __init__(self) -> None:
        self.parse_status: str = ""
        self.parse_detail: str = ""
        self.worker_rows: list[dict] = []  # {agent, description, status, summary}
        self.total_tasks: int = 0
        self.research_summary: str = ""
        self.synthesis_status: str = ""
        self.show_details: bool = False

    def update(self, event: PlanEvent) -> None:
        if event.phase == "parse":
            if event.event_type == "start":
                self.parse_status = "Analyzing your travel request..."
            elif event.event_type == "complete":
                dests = event.data.get("destinations", [])
                budget = event.data.get("budget")
                duration = event.data.get("duration")
                self.total_tasks = event.data.get("num_tasks", 0)
                parts = [f"**Destinations:** {', '.join(dests)}"]
                if duration:
                    parts.append(f"**Duration:** {duration} days")
                if budget:
                    parts.append(f"**Budget:** ${budget:,.0f}")
                self.parse_detail = " | ".join(parts)
                self.parse_status = "Query understood"

        elif event.phase == "worker":
            if event.event_type == "start":
                label = AGENT_LABELS.get(event.agent_name, event.agent_name)
                self.worker_rows.append({
                    "index": event.data.get("index", 0),
                    "agent": label,
                    "description": event.description,
                    "status": "running",
                    "summary": "",
                })
            elif event.event_type in ("complete", "fail"):
                idx = event.data.get("index", -1)
                for row in self.worker_rows:
                    if row["index"] == idx:
                        row["status"] = "done" if event.event_type == "complete" else "failed"
                        row["summary"] = event.description
                        break
            elif event.event_type == "status":
                self.research_summary = event.description

        elif event.phase == "synthesis":
            self.synthesis_status = event.description

        elif event.phase == "done":
            self.synthesis_status = "Itinerary ready!"

        elif event.phase == "error":
            self.synthesis_status = f"Error: {event.description}"

    def render(self) -> str:
        """Render the current state as Markdown."""
        lines: list[str] = []

        # Phase 1: Parse
        if self.parse_status:
            icon = "..." if not self.parse_detail else "OK"
            lines.append(f"**Step 1 - Understanding Request** [{icon}]")
            lines.append("")
            if self.parse_detail:
                lines.append(self.parse_detail)
                lines.append("")

        # Phase 2: Workers
        if self.worker_rows or self.total_tasks:
            done_count = sum(1 for r in self.worker_rows if r["status"] in ("done", "failed"))
            started = len(self.worker_rows)
            lines.append(f"**Step 2 - Researching** [{done_count}/{self.total_tasks} agents complete]")
            lines.append("")

            if self.worker_rows:
                # Compact view
                for row in self.worker_rows:
                    if row["status"] == "running":
                        icon = "[ ]"
                    elif row["status"] == "done":
                        icon = "[x]"
                    else:
                        icon = "[!]"
                    summary = f" -- {row['summary']}" if row["summary"] else ""
                    lines.append(f"- {icon} **{row['agent']}**: {row['description']}{summary}")

                lines.append("")

                # Detailed table (collapsible)
                if self.show_details and any(r["status"] == "done" for r in self.worker_rows):
                    lines.append("<details><summary>Detailed agent results</summary>")
                    lines.append("")
                    lines.append("| Agent | Task | Status | Findings |")
                    lines.append("|-------|------|--------|----------|")
                    for row in self.worker_rows:
                        status_icon = {"running": "...", "done": "OK", "failed": "FAIL"}[row["status"]]
                        lines.append(
                            f"| {row['agent']} | {row['description'][:40]} "
                            f"| {status_icon} | {row['summary']} |"
                        )
                    lines.append("")
                    lines.append("</details>")
                    lines.append("")

            if self.research_summary:
                lines.append(f"*{self.research_summary}*")
                lines.append("")

        # Phase 3: Synthesis
        if self.synthesis_status:
            lines.append(f"**Step 3 - Building Itinerary** [{self.synthesis_status}]")
            lines.append("")

        return "\n".join(lines)


# ── Gradio Web UI ──────────────────────────────────────────────


def launch_web() -> None:
    import gradio as gr

    async def run_pipeline(query: str, feedback_text: str):
        """Async generator: runs full pipeline, yields (activity_log, plan, state) at each event."""
        if not query or not query.strip():
            yield ("*Please describe your trip above.*", "", None, gr.update(visible=False))
            return

        for var in ("GEMINI_API_KEY", "TAVILY_API_KEY"):
            if not os.environ.get(var):
                yield (f"**Error:** `{var}` not set in `.env`", "", None, gr.update(visible=False))
                return

        event_queue: asyncio.Queue[PlanEvent] = asyncio.Queue()
        log = ActivityLog()

        # Run research in a background task
        research_data = {}

        async def do_research():
            request, tasks, results = await research_trip(query.strip(), event_queue)
            research_data["request"] = request
            research_data["tasks"] = tasks
            research_data["results"] = results

        research_task = asyncio.create_task(do_research())

        # Stream events from research phase
        while not research_task.done():
            try:
                event = await asyncio.wait_for(event_queue.get(), timeout=0.3)
                log.update(event)
                yield (log.render(), "", None, gr.update(visible=True))
            except asyncio.TimeoutError:
                pass
            except Exception:
                break

        # Drain remaining events
        while not event_queue.empty():
            event = event_queue.get_nowait()
            log.update(event)

        # Check for research errors
        if research_task.exception():
            err = research_task.exception()
            yield (log.render() + f"\n\n**Error:** {err}", "", None, gr.update(visible=False))
            return

        # Yield research-complete state (user can see activity, feedback box is visible)
        yield (log.render(), "", research_data, gr.update(visible=True))

        # Give user a moment to type feedback if they want
        # (the feedback_text param captures whatever they've typed by now)
        await asyncio.sleep(0.5)

        # Phase 3: Synthesize (incorporating any feedback the user typed)
        actual_feedback = feedback_text.strip() if feedback_text else ""
        try:
            plan = await synthesize_trip(
                research_data["request"],
                research_data["results"],
                user_feedback=actual_feedback,
                event_queue=event_queue,
            )
        except Exception as e:
            yield (log.render() + f"\n\n**Synthesis error:** {e}", "", None, gr.update(visible=False))
            return

        # Drain synthesis events
        while not event_queue.empty():
            event = event_queue.get_nowait()
            log.update(event)

        result_md = format_plan_markdown(plan)
        yield (log.render(), result_md, None, gr.update(visible=False))

    with gr.Blocks(title="iLiv Travel Planner") as app:
        gr.Markdown(
            "# iLiv Travel Planner\n"
            "AI-powered multi-agent trip planning. Describe your trip below "
            "and our agents will research destinations, accommodations, transport, "
            "and safety to build you a complete itinerary."
        )

        query_input = gr.Textbox(
            label="Describe your trip",
            placeholder="e.g. Plan a 5-day trip to Tokyo for 2 adults, budget $3000...",
            lines=4,
        )

        submit_btn = gr.Button("Plan My Trip", variant="primary")

        gr.Examples(
            examples=[[q] for q in EXAMPLE_QUERIES],
            inputs=[query_input],
            label="Example queries",
        )

        activity_display = gr.Markdown(
            value="",
            label="Agent Activity",
        )

        feedback_input = gr.Textbox(
            label="Optional: Add feedback while agents work (e.g. 'prefer boutique hotels', 'skip hostels')",
            placeholder="Type here if you want to steer the plan before it's generated...",
            lines=2,
            visible=False,
        )

        # State to hold research data between steps
        research_state = gr.State(value=None)

        result_display = gr.Markdown(
            value="",
            label="Travel Plan",
        )

        submit_btn.click(
            fn=run_pipeline,
            inputs=[query_input, feedback_input],
            outputs=[activity_display, result_display, research_state, feedback_input],
        )

    app.launch(theme=gr.themes.Soft())


def main() -> None:
    args = sys.argv[1:]

    if "--web" in args:
        launch_web()
        return

    output_json = False
    if "--json" in args:
        output_json = True
        args.remove("--json")

    query = " ".join(args) if args else None

    if not query:
        print("iLiv Travel Planner")
        print("=" * 50)
        print("\nDescribe your trip (press Enter when done):\n")
        query = input("> ").strip()
        if not query:
            print("No query provided. Exiting.")
            sys.exit(0)

    asyncio.run(async_main(query, output_json))


if __name__ == "__main__":
    main()
