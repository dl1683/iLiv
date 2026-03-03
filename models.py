"""Data contracts for all agents in the iLiv travel planning system."""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ── Input Parsing ──────────────────────────────────────────────


class TravelerProfile(BaseModel):
    adults: int = 2
    children: int = 0
    child_ages: list[int] = Field(default_factory=list)
    dietary_restrictions: list[str] = Field(default_factory=list)
    accessibility_needs: list[str] = Field(default_factory=list)
    interests: list[str] = Field(default_factory=list)


class TravelRequest(BaseModel):
    destinations: list[str]
    origin: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    duration_days: Optional[int] = None
    budget_total: Optional[float] = None
    budget_currency: str = "USD"
    travelers: TravelerProfile = Field(default_factory=TravelerProfile)
    priorities: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    raw_query: str


# ── Worker Orchestration ───────────────────────────────────────


class WorkerTaskType(str, Enum):
    DESTINATION_RESEARCH = "destination_research"
    ACCOMMODATION = "accommodation"
    TRANSPORT = "transport"
    SAFETY = "safety"


class WorkerTaskParam(BaseModel):
    key: str
    value: str


class WorkerTask(BaseModel):
    task_type: WorkerTaskType
    description: str
    parameters: list[WorkerTaskParam]


class WorkerManifest(BaseModel):
    tasks: list[WorkerTask]
    reasoning: str


class CoordinatorParseResponse(BaseModel):
    travel_request: TravelRequest
    worker_manifest: WorkerManifest


# ── Worker Results ─────────────────────────────────────────────


class DestinationInfo(BaseModel):
    city: str
    country: str
    summary: str
    top_attractions: list[str]
    best_neighborhoods: list[str]
    weather_summary: str
    visa_requirements: str
    local_tips: list[str]
    family_friendliness_score: Optional[int] = None
    sources: list[str] = Field(default_factory=list)


class AccommodationOption(BaseModel):
    name: str
    type: str
    price_per_night_usd: float
    neighborhood: str
    rating: Optional[float] = None
    pros: list[str]
    cons: list[str]
    url: Optional[str] = None
    family_suitable: bool = True


class AccommodationOptions(BaseModel):
    city: str
    options: list[AccommodationOption]
    recommendation: str
    sources: list[str] = Field(default_factory=list)


class TransportLeg(BaseModel):
    from_city: str
    to_city: str
    mode: str
    duration_hours: float
    price_usd: float
    provider: Optional[str] = None
    notes: str = ""


class LocalTransportTip(BaseModel):
    city: str
    tip: str


class TransportOptions(BaseModel):
    legs: list[TransportLeg]
    local_transport_tips: list[LocalTransportTip] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)


class EmergencyNumber(BaseModel):
    service: str
    number: str


class SafetyReport(BaseModel):
    city: str
    overall_safety_rating: int
    travel_advisory_level: str
    health_requirements: list[str]
    areas_to_avoid: list[str]
    emergency_numbers: list[EmergencyNumber] = Field(default_factory=list)
    family_safety_notes: list[str] = Field(default_factory=list)
    accessibility_notes: list[str] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)


class WorkerResult(BaseModel):
    """Internal-only model. Not sent to Gemini as a schema."""
    task_type: WorkerTaskType
    success: bool
    data: dict  # not used as a Gemini response_schema
    error: Optional[str] = None


# ── Final Output ───────────────────────────────────────────────


class DayPlan(BaseModel):
    day_number: int
    date: Optional[str] = None
    city: str
    morning: str
    afternoon: str
    evening: str
    meals: list[str] = Field(default_factory=list)
    estimated_cost: float = 0.0
    notes: list[str] = Field(default_factory=list)


class CostBreakdown(BaseModel):
    accommodation_total: float
    transport_total: float
    food_estimated_daily: float
    food_total: float
    activities_estimated_daily: float
    activities_total: float
    miscellaneous: float
    grand_total: float
    budget_remaining: Optional[float] = None
    cost_saving_tips: list[str] = Field(default_factory=list)
    currency: str = "USD"


class TravelPlan(BaseModel):
    title: str
    summary: str
    destinations: list[str]
    duration_days: int
    travelers: TravelerProfile
    itinerary: list[DayPlan]
    accommodation_plan: list[AccommodationOption]
    transport_plan: list[TransportLeg]
    cost_breakdown: CostBreakdown
    safety_notes: list[str]
    packing_suggestions: list[str]
    important_links: list[str] = Field(default_factory=list)
    disclaimers: list[str] = Field(default_factory=list)
