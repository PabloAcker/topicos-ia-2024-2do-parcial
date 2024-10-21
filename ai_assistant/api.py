from fastapi import FastAPI, Depends, Query
from llama_index.core.agent import ReActAgent
from ai_assistant.agent import TravelAgent
from ai_assistant.models import AgentAPIResponse
from ai_assistant.tools import reserve_flight, reserve_bus, reserve_hotel, reserve_restaurant
from typing import List
from datetime import datetime
from ai_assistant.prompts import agent_prompt_tpl


def get_agent() -> ReActAgent:
    return TravelAgent(agent_prompt_tpl).get_agent()


app = FastAPI(title="AI Agent")


@app.get("/recommendations/cities")
def recommend_cities(
    notes: list[str] = Query(...), agent: ReActAgent = Depends(get_agent)
):
    prompt = f"recommend cities in bolivia with the following notes: {notes}"
    return AgentAPIResponse(status="OK", agent_response=str(agent.chat(prompt)))


# Recomendaciones de lugares en una ciudad
@app.get("/recommendations/places")
def recommend_places(city: str, notes: List[str] = Query(None), agent: ReActAgent = Depends(get_agent)):
    notes_str = ", ".join(notes) if notes else "No notes provided"
    prompt = f"Recommend places to visit in {city} with the following notes: {notes_str}"
    return AgentAPIResponse(status="OK", agent_response=str(agent.chat(prompt)))


# Recomendaciones de hoteles en una ciudad
@app.get("/recommendations/hotels")
def recommend_hotels(city: str, notes: List[str] = Query(None), agent: ReActAgent = Depends(get_agent)):
    notes_str = ", ".join(notes) if notes else "No notes provided"
    prompt = f"Recommend hotels in {city} with the following notes: {notes_str}"
    return AgentAPIResponse(status="OK", agent_response=str(agent.chat(prompt)))


# Recomendaciones de actividades en una ciudad
@app.get("/recommendations/activities")
def recommend_activities(city: str, notes: List[str] = Query(None), agent: ReActAgent = Depends(get_agent)):
    notes_str = ", ".join(notes) if notes else "No notes provided"
    prompt = f"Recommend activities to do in {city} with the following notes: {notes_str}"
    return AgentAPIResponse(status="OK", agent_response=str(agent.chat(prompt)))


# Reservar vuelo
@app.post("/reserve/flight")
def reserve_flight_endpoint(departure: str, destination: str, date: str):
    reservation = reserve_flight(date_str=date, departure=departure, destination=destination)
    return {"status": "OK", "reservation": reservation.dict()}


# Reservar bus
@app.post("/reserve/bus")
def reserve_bus_endpoint(departure: str, destination: str, date: str):
    reservation = reserve_bus(date_str=date, departure=departure, destination=destination)
    return {"status": "OK", "reservation": reservation.dict()}


# Reservar hotel
@app.post("/reserve/hotel")
def reserve_hotel_endpoint(checkin_date: str, checkout_date: str, hotel: str, city: str):
    reservation = reserve_hotel(checkin_date=checkin_date, checkout_date=checkout_date, hotel_name=hotel, city=city)
    return {"status": "OK", "reservation": reservation.dict()}


# Reservar restaurante
@app.post("/reserve/restaurant")
def reserve_restaurant_endpoint(reservation_time: str, restaurant: str, city: str, dish: str):
    reservation = reserve_restaurant(reservation_time=reservation_time, restaurant=restaurant, city=city, dish=dish)
    return {"status": "OK", "reservation": reservation.dict()}


# Generar reporte de viaje basado en actividades
@app.get("/report/trip")
def generate_trip_report(agent: ReActAgent = Depends(get_agent)):
    prompt = "Generate a detailed trip report based on the log of activities, including all places visited, dates, total budget, and comments on the activities."
    response = agent.chat(prompt)
    return AgentAPIResponse(status="OK", agent_response=str(response))