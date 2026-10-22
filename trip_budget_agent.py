"""
Trip Budget Agent - base starter code
Estimates transportation, accommodation, food and activity costs for a trip.

Run:  python trip_budget_agent.py
Currency: INR (change CURRENCY / rate tables as needed)

Next steps to make it a real "AI agent":
  1. Replace the static tables with live APIs (flights, hotels, maps).
  2. Use an LLM (see parse_request_with_llm) to turn free text into a TripRequest.
  3. Let the LLM call the estimate_* functions as tools and explain the result.
"""

from dataclasses import dataclass, field
from typing import Dict, List

CURRENCY = "INR"

# ----------------------------------------------------------------------------
# 1. Sample knowledge base (replace with real data / APIs later)
# ----------------------------------------------------------------------------

# Approx. one-way cost per km per person
TRANSPORT_RATE_PER_KM = {
    "bus": 2.0,
    "train": 1.5,
    "car": 8.0,      # fuel + tolls, shared across the car (not per person)
    "flight": 6.0,
}

# Known route distances in km (fallback: pass distance_km yourself)
ROUTE_DISTANCE_KM = {
    ("bhopal", "goa"): 1500,
    ("bhopal", "delhi"): 750,
    ("bhopal", "mumbai"): 780,
    ("bhopal", "jaipur"): 780,
}

# Price per room per night by hotel class
HOTEL_PER_NIGHT = {"budget": 1500, "standard": 3500, "luxury": 9000}

# Food cost per person per day
FOOD_PER_DAY = {"street": 400, "casual": 900, "fine": 2200}

# City price multiplier (1.0 = average Indian city)
DESTINATION_MULTIPLIER = {"goa": 1.3, "delhi": 1.1, "mumbai": 1.4, "jaipur": 1.0}

# Activity price per person
ACTIVITY_COST = {
    "museum": 300,
    "water sports": 1800,
    "city tour": 1000,
    "trekking": 1200,
    "boat ride": 600,
}


# ----------------------------------------------------------------------------
# 2. Data models
# ----------------------------------------------------------------------------
@dataclass
class TripRequest:
    origin: str
    destination: str
    days: int
    travelers: int = 1
    transport_mode: str = "train"     # bus | train | car | flight
    hotel_class: str = "standard"     # budget | standard | luxury
    food_style: str = "casual"        # street | casual | fine
    activities: List[str] = field(default_factory=list)
    distance_km: float | None = None  # optional override
    rooms: int | None = None          # default: 1 room per 2 travelers


@dataclass
class BudgetReport:
    breakdown: Dict[str, float]
    buffer_pct: float
    buffer_amount: float
    total: float
    per_person: float
    notes: List[str]

    def __str__(self) -> str:
        lines = ["=" * 44, "TRIP BUDGET ESTIMATE", "=" * 44]
        for k, v in self.breakdown.items():
            lines.append(f"{k.capitalize():<16}{CURRENCY} {v:>12,.0f}")
        lines.append("-" * 44)
        lines.append(f"{'Buffer (' + str(int(self.buffer_pct * 100)) + '%)':<16}{CURRENCY} {self.buffer_amount:>12,.0f}")
        lines.append(f"{'TOTAL':<16}{CURRENCY} {self.total:>12,.0f}")
        lines.append(f"{'Per person':<16}{CURRENCY} {self.per_person:>12,.0f}")
        if self.notes:
            lines.append("\nNotes:")
            lines += [f"  - {n}" for n in self.notes]
        return "\n".join(lines)


# ----------------------------------------------------------------------------
# 3. Estimator "tools" (an LLM agent can call these as functions)
# ----------------------------------------------------------------------------
def _multiplier(city: str) -> float:
    return DESTINATION_MULTIPLIER.get(city.lower(), 1.0)


def estimate_transport(req: TripRequest, notes: List[str]) -> float:
    distance = req.distance_km or ROUTE_DISTANCE_KM.get(
        (req.origin.lower(), req.destination.lower())
    )
    if distance is None:
        distance = 500
        notes.append("Route distance unknown; assumed 500 km. Pass distance_km for accuracy.")

    rate = TRANSPORT_RATE_PER_KM[req.transport_mode]
    round_trip = distance * 2
    if req.transport_mode == "car":
        cost = round_trip * rate                 # one car, cost shared
    else:
        cost = round_trip * rate * req.travelers
    # local transport at destination (auto/cab), ~300 per day per group
    local = 300 * req.days * _multiplier(req.destination)
    return cost + local


def estimate_accommodation(req: TripRequest, notes: List[str]) -> float:
    rooms = req.rooms or -(-req.travelers // 2)  # ceil division
    nights = max(req.days - 1, 1)
    return HOTEL_PER_NIGHT[req.hotel_class] * rooms * nights * _multiplier(req.destination)


def estimate_food(req: TripRequest, notes: List[str]) -> float:
    return FOOD_PER_DAY[req.food_style] * req.travelers * req.days * _multiplier(req.destination)


def estimate_activities(req: TripRequest, notes: List[str]) -> float:
    total = 0.0
    for act in req.activities:
        price = ACTIVITY_COST.get(act.lower())
        if price is None:
            price = 700
            notes.append(f"No price for activity '{act}'; assumed {CURRENCY} 700/person.")
        total += price * req.travelers
    return total * _multiplier(req.destination)


# ----------------------------------------------------------------------------
# 4. The agent
# ----------------------------------------------------------------------------
class TripBudgetAgent:
    def __init__(self, buffer_pct: float = 0.10):
        self.buffer_pct = buffer_pct

    def estimate(self, req: TripRequest) -> BudgetReport:
        notes: List[str] = []
        breakdown = {
            "transportation": estimate_transport(req, notes),
            "accommodation": estimate_accommodation(req, notes),
            "food": estimate_food(req, notes),
            "activities": estimate_activities(req, notes),
        }
        subtotal = sum(breakdown.values())
        buffer_amount = subtotal * self.buffer_pct
        total = subtotal + buffer_amount
        return BudgetReport(
            breakdown=breakdown,
            buffer_pct=self.buffer_pct,
            buffer_amount=buffer_amount,
            total=total,
            per_person=total / req.travelers,
            notes=notes,
        )


# ----------------------------------------------------------------------------
# 5. Optional: LLM front-end (natural language -> TripRequest)
# ----------------------------------------------------------------------------
def parse_request_with_llm(user_text: str) -> TripRequest:
    """
    Sketch using the Anthropic SDK (pip install anthropic, set ANTHROPIC_API_KEY).
    Ask the model to return JSON matching TripRequest, then build the object.
    """
    import json
    import anthropic

    client = anthropic.Anthropic()
    system = (
        "Extract trip details as JSON only, with keys: origin, destination, days, "
        "travelers, transport_mode (bus|train|car|flight), hotel_class "
        "(budget|standard|luxury), food_style (street|casual|fine), activities (list)."
    )
    msg = client.messages.create(
        model="claude-sonnet-4-5",  # use a current model name from the docs
        max_tokens=500,
        system=system,
        messages=[{"role": "user", "content": user_text}],
    )
    data = json.loads(msg.content[0].text)
    return TripRequest(**data)


# ----------------------------------------------------------------------------
# Demo
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    request = TripRequest(
        origin="Bhopal",
        destination="Goa",
        days=5,
        travelers=3,
        transport_mode="train",
        hotel_class="standard",
        food_style="casual",
        activities=["water sports", "boat ride", "city tour"],
    )
    print(TripBudgetAgent().estimate(request))
