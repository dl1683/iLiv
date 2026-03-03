# iLiv Travel Planner

Multi-agent AI travel planner that researches destinations, accommodations, transport, and safety in parallel to build complete day-by-day itineraries.

## How It Works

A coordinator agent (Gemini) parses your travel query and dispatches specialized worker agents that search the web in real time:

- **Destination Research** — attractions, neighborhoods, weather, visa info, local tips
- **Accommodation** — hotels/apartments across budget tiers with ratings and pros/cons
- **Transport** — flights, trains, buses between cities + local transit tips
- **Safety** — advisories, areas to avoid, emergency numbers, health requirements

Results are synthesized into a structured itinerary with cost breakdowns, packing suggestions, and useful links.

## Prerequisites

- Python 3.10+
- A [Google Gemini API key](https://aistudio.google.com/apikey)
- A [Tavily API key](https://tavily.com/)

## Setup

1. **Clone the repo**

   ```bash
   git clone https://github.com/dl1683/iLiv.git
   cd iLiv
   ```

2. **Create a virtual environment** (recommended)

   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Linux/macOS
   .venv\Scripts\activate      # Windows
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Configure API keys**

   Create a `.env` file in the project root:

   ```
   GEMINI_API_KEY=your_gemini_api_key_here
   TAVILY_API_KEY=your_tavily_api_key_here
   ```

## Usage

### Web UI (Gradio)

```bash
python main.py --web
```

Opens a browser interface where you can type a query, watch agents work in real time, and optionally steer the plan with feedback while research runs.

### CLI

```bash
# Direct query
python main.py "Plan a 5-day trip to Tokyo for 2 adults, budget $3000"

# Interactive prompt
python main.py

# JSON output
python main.py --json "Solo trip to Barcelona for 4 days"
```

## Example Queries

- *"Plan a 5-day trip to Tokyo for 2 adults, budget $3000, interested in food and culture"*
- *"Solo female traveler visiting Paris and Barcelona for 7 days in June. Budget $4000. Love art, wine, and safe walkable neighborhoods."*
- *"Family trip with 2 kids (ages 5 and 9) to Orlando and Miami for 10 days. Budget $6000. Need kid-friendly activities."*
- *"Backpacking Southeast Asia for 3 weeks: Bangkok, Hanoi, and Bali. Budget $2500. Vegetarian."*

## Project Structure

```
main.py          Entry point — CLI, Gradio web UI, markdown formatter
agents.py        Coordinator + worker agent logic, event system
models.py        Pydantic data contracts (requests, results, final plan)
config.py        Model IDs, prompts, search defaults
tools.py         Async Tavily search wrappers
```
