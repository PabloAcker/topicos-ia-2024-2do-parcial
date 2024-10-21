from random import randint
from datetime import date, datetime, time
from llama_index.core.tools import QueryEngineTool, FunctionTool, ToolMetadata
from ai_assistant.rags import TravelGuideRAG
from ai_assistant.prompts import travel_guide_qa_tpl, travel_guide_description
from ai_assistant.config import get_agent_settings
from ai_assistant.models import (
    TripReservation,
    TripType,
    HotelReservation,
    RestaurantReservation,
)
from ai_assistant.utils import save_reservation
import json
from ai_assistant.config import get_agent_settings
from ai_assistant.utils import custom_serializer

SETTINGS = get_agent_settings()

travel_guide_tool = QueryEngineTool(
    query_engine=TravelGuideRAG(
        store_path=SETTINGS.travel_guide_store_path,
        data_dir=SETTINGS.travel_guide_data_path,
        qa_prompt_tpl=travel_guide_qa_tpl,
    ).get_query_engine(),
    metadata=ToolMetadata(
        name="travel_guide", description=travel_guide_description, return_direct=False
    ),
)


# Helper function to create a reservation
def create_reservation(reservation_type: str, cost_range: tuple, **kwargs):
    """
    General function to create different types of reservations with random costs.

    Args:
        reservation_type (str): Type of reservation ('flight', 'bus', 'hotel', or 'restaurant').
        cost_range (tuple): A tuple representing the min and max cost.
        kwargs (dict): Additional reservation details (varies by type).

    Returns:
        Reservation object depending on type.
    """
    cost = randint(*cost_range)
    reservation_data = kwargs.copy()
    reservation_data['cost'] = cost

    if reservation_type == "flight" or reservation_type == "bus":
        return TripReservation(trip_type=TripType[reservation_type], **reservation_data)
    elif reservation_type == "hotel":
        return HotelReservation(**reservation_data)
    elif reservation_type == "restaurant":
        return RestaurantReservation(**reservation_data)

# Flight reservation function
def reserve_flight(date_str: str, departure: str, destination: str) -> TripReservation:
    print(f"Making flight reservation from {departure} to {destination} on date: {date_str}")
    reservation = create_reservation(
        reservation_type="flight",
        cost_range=(200, 700),
        departure=departure,
        destination=destination,
        date=date.fromisoformat(date_str)
    )
    save_reservation(reservation)
    return reservation

flight_tool = FunctionTool.from_defaults(fn=reserve_flight, return_direct=False)

# Bus reservation function
def reserve_bus(date_str: str, departure: str, destination: str) -> TripReservation:
    print(f"Making bus reservation from {departure} to {destination} on date: {date_str}")
    reservation = create_reservation(
        reservation_type="bus",
        cost_range=(50, 200),
        departure=departure,
        destination=destination,
        date=date.fromisoformat(date_str)
    )
    save_reservation(reservation)
    return reservation

bus_tool = FunctionTool.from_defaults(fn=reserve_bus, return_direct=False)

# Hotel reservation function
def reserve_hotel(checkin_date_str: str, checkout_date_str: str, hotel_name: str, city: str) -> HotelReservation:
    print(f"Making hotel reservation at {hotel_name} in {city} from {checkin_date_str} to {checkout_date_str}")
    checkin_date = date.fromisoformat(checkin_date_str)
    checkout_date = date.fromisoformat(checkout_date_str)
    num_nights = (checkout_date - checkin_date).days
    total_cost = num_nights * randint(100, 300)
    
    reservation = create_reservation(
        reservation_type="hotel",
        cost_range=(100 * num_nights, 300 * num_nights),
        checkin_date=checkin_date,
        checkout_date=checkout_date,
        hotel_name=hotel_name,
        city=city,
        cost=total_cost
    )
    save_reservation(reservation)
    return reservation

hotel_tool = FunctionTool.from_defaults(fn=reserve_hotel, return_direct=False)

# Restaurant reservation function
def reserve_restaurant(reservation_datetime_str: str, restaurant: str, city: str, dish: str = "not specified") -> RestaurantReservation:
    print(f"Making restaurant reservation at {restaurant} in {city} on {reservation_datetime_str}")
    reservation_time = datetime.fromisoformat(reservation_datetime_str)
    
    reservation = create_reservation(
        reservation_type="restaurant",
        cost_range=(20, 100),
        reservation_time=reservation_time,
        restaurant=restaurant,
        city=city,
        dish=dish
    )
    save_reservation(reservation)
    return reservation

restaurant_tool = FunctionTool.from_defaults(fn=reserve_restaurant, return_direct=False)

def generate_trip_report() -> str:
    """
    Generates a detailed report of the trip based on the trip log (trip.json).
    The report includes all activities organized by place and date, a total budget summary,
    and comments about the places and activities.

    Returns:
        str: A detailed trip report summarizing the activities, places, dates, and costs.
    """
    log_file = SETTINGS.log_file
    report = []
    total_cost = 0
    places_visited = {}
    
    try:
        # Leer el archivo de registro de actividades
        with open(log_file, 'r') as file:
            trip_data = json.load(file)
        
        # Organizar las actividades por lugar y fecha
        for entry in trip_data:
            city = entry.get('city', entry.get('destination', 'Unknown'))
            date = entry.get('date', entry.get('checkin_date', entry.get('reservation_time', 'Unknown')))
            cost = entry.get('cost', 0)
            total_cost += cost

            if city not in places_visited:
                places_visited[city] = []

            activity = f"{entry['reservation_type']} on {date} - Cost: ${cost}"
            places_visited[city].append(activity)
        
        # Crear el reporte detallado
        for city, activities in places_visited.items():
            report.append(f"City: {city}")
            report.extend(activities)
            report.append("\n")
        
        report.append(f"Total budget for the trip: ${total_cost}")
    
    except FileNotFoundError:
        return "Error: trip log file not found."
    except json.JSONDecodeError:
        return "Error: Could not decode the trip log file."

    return "\n".join(report)

# Register this tool as a FunctionTool
trip_report_tool = FunctionTool.from_defaults(
    fn=generate_trip_report,
    return_direct=False
)


