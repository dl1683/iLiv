"""Configuration: model IDs, prompts, search parameters, defaults."""

import datetime

# ── Models ─────────────────────────────────────────────────────

COORDINATOR_MODEL = "gemini-3-flash-preview"
WORKER_MODEL = "gemini-2.5-flash-lite"

CURRENT_YEAR = datetime.date.today().year

# ── Search Defaults ────────────────────────────────────────────

DEFAULT_SEARCH_DEPTH = "basic"
DEFAULT_MAX_RESULTS = 5
DEFAULT_FOOD_DAILY_USD = 60
DEFAULT_ACTIVITIES_DAILY_USD = 40

# ── Coordinator Prompts ────────────────────────────────────────

COORDINATOR_PARSE_SYSTEM = f"""\
You are a travel planning coordinator. Today's date is {datetime.date.today().isoformat()}.

Parse the user's travel query into a structured JSON response containing:
1. A "travel_request" object with all extracted travel details.
2. A "worker_manifest" listing the research tasks to dispatch.

Rules for creating the worker manifest:
- For EACH destination city, create a "destination_research" task AND an "accommodation" task.
- If the user mentions safety concerns, is a solo female traveler, is traveling with children,\
 or visiting less-common destinations, create a "safety" task per city.
- For travel BETWEEN cities (or from origin to first city, last city to origin), create "transport" tasks.
- Each task's "parameters" dict must contain the relevant details (city, budget_per_night, dates, etc.).
- Be thorough: if the user mentions N cities, you need at least 2N tasks (research + accommodation per city)\
 plus transport legs.

If dates are vague (e.g. "next summer"), infer reasonable dates.
If budget is not specified, set budget_total to null — workers will provide a range of options.
Preserve the original query verbatim in travel_request.raw_query.

Respond ONLY with valid JSON matching the schema."""

COORDINATOR_SYNTHESIZE_SYSTEM = f"""\
You are an expert travel planner. Today's date is {datetime.date.today().isoformat()}.

Given a parsed travel request and research results from multiple specialized agents, create a\
 comprehensive day-by-day travel itinerary as a structured JSON "TravelPlan".

Rules:
- Create a complete day-by-day itinerary covering every day of the trip.
- Each day should have realistic morning/afternoon/evening activities with specific place names.
- Respect the stated budget. If costs exceed budget, prioritize cheaper options and note it.
- Incorporate ALL safety concerns from the safety reports.
- Account for travel days (arrival/departure, intercity transport).
- Include meal suggestions that respect dietary restrictions.
- For accommodation_plan, pick the BEST option per city from the research (best value + rating).
- For transport_plan, pick the most practical option per leg.
- Calculate the cost_breakdown accurately from the selected options.
- Include practical packing suggestions based on weather and activities.
- Add disclaimers about price estimates and booking recommendations.
- If any worker failed, note the gap in disclaimers and provide best-effort info.

Respond ONLY with valid JSON matching the schema."""

# ── Worker Prompt Templates ────────────────────────────────────

WORKER_PROMPTS = {
    "destination_research": """\
Research {city} as a travel destination for a trip in {travel_period}.

Web search results:
{context}

Based on the search results above, extract structured information about {city}:
- A concise summary of the city as a destination
- Top 5-8 attractions and things to do
- Best neighborhoods to stay in
- Weather summary for the travel period
- Visa requirements (assume travelers are from {origin}, or give general info if unknown)
- 3-5 practical local tips
- Family friendliness score (1-10, where 10 is most family friendly)
- List the source URLs you drew from

Respond ONLY with valid JSON matching the schema.""",

    "accommodation": """\
Find accommodation in {city} for {num_travelers} travelers ({adults} adults, {children} children)\
 for approximately {nights} nights.
Budget: around {budget_per_night} {currency} per night (if specified, otherwise suggest a range).
Preferences: {preferences}

Web search results:
{context}

Provide 3-4 accommodation options ranging from budget to mid-range to higher-end.
For each option include: name, type (hotel/apartment/hostel), price per night in USD,\
 neighborhood, rating if found, pros, cons, booking URL if found, and whether it's suitable for families.
Include a brief recommendation of which option is best value.

Respond ONLY with valid JSON matching the schema.""",

    "transport": """\
Find transportation options for traveling from {from_city} to {to_city}\
 around {travel_date} for {num_travelers} travelers.

Web search results:
{context}

Provide the best transportation options (flights, trains, buses as applicable).
For each option: mode, duration in hours, estimated price in USD per person, provider, and any notes.
Also include local transport tips for getting around {to_city}.

Respond ONLY with valid JSON matching the schema.""",

    "safety": """\
Assess travel safety for {city} ({country}) for the following traveler profile:
{traveler_profile}

Web search results:
{context}

Provide:
- Overall safety rating (1-10, where 10 is safest)
- Current travel advisory level (from US State Dept or equivalent)
- Health requirements (vaccinations, insurance, etc.)
- Areas to avoid
- Emergency numbers
- Family safety notes (if traveling with children)
- Accessibility notes (if relevant)
- Source URLs

Respond ONLY with valid JSON matching the schema.""",
}
