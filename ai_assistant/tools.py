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


# Tool functions
def reserve_flight(date_str: str, departure: str, destination: str) -> TripReservation:
    """
    Book a flight between two cities for a specific date.

    Args:
        date_str (str): Flight date in ISO format (YYYY-MM-DD).
        departure (str): Departure city.
        destination (str): Destination city.

    Returns:
        TripReservation: A TripReservation instance that contains the reservation details, such as trip type,
         date, departure location, destination location, and cost.

    Ejemplo:
        reserve_flight('2024-12-01', 'La Paz', 'Santa Cruz')
    """
    print(
        f"Making flight reservation from {departure} to {destination} on date: {date}"
    )
    reservation = TripReservation(
        trip_type=TripType.flight,
        departure=departure,
        destination=destination,
        date=date.fromisoformat(date_str),
        cost=randint(200, 700),
    )

    save_reservation(reservation)
    return reservation


flight_tool = FunctionTool.from_defaults(fn=reserve_flight, return_direct=False)

def reserve_hotel(checkin_date: str, checkout_date: str, hotel_name: str, city: str) -> HotelReservation:
    """
    Book a hotel in a specific city between two dates.

    Args:
        checkin_date (str): Check-in date in ISO format (YYYY-MM-DD).
        checkout_date (str): Check-out date in ISO format (YYYY-MM-DD).
        hotel_name (str): Hotel name.
        city (str): City where the reservation is made.

    Returns:
        HotelReservation: Hotel reservation details.
    """
    print(f"Making hotel reservation at {hotel_name} in {city} from {checkin_date} to {checkout_date}")
    reservation = HotelReservation(
        checkin_date=date.fromisoformat(checkin_date),
        checkout_date=date.fromisoformat(checkout_date),
        hotel_name=hotel_name,
        city=city,
        cost=randint(50, 300),
    )

    save_reservation(reservation)
    return reservation

hotel_tool = FunctionTool.from_defaults(fn=reserve_hotel, return_direct=False)


def reserve_bus(date_str: str, departure: str, destination: str) -> TripReservation:
    """
    Book a bus trip between two cities on a specific date.

    Args:
        date_str (str): Trip date in ISO format.
        departure (str): Departure city.
        destination (str): Destination city.

    Returns:
        TripReservation: A TripReservation instance with the reservation details.
    """
    print(f"Making bus reservation from {departure} to {destination} on date: {date_str}")
    reservation = TripReservation(
        trip_type=TripType.bus,
        departure=departure,
        destination=destination,
        date=date.fromisoformat(date_str),
        cost=randint(20, 100),
    )

    save_reservation(reservation)
    return reservation

bus_tool = FunctionTool.from_defaults(fn=reserve_bus, return_direct=False)


def reserve_restaurant(reservation_time: str, restaurant: str, city: str, dish: str) -> RestaurantReservation:
    """
    Make a reservation at a restaurant in a specific city for a given time.

    Args:
        reservation_time (str): Reservation time in ISO format.
        restaurant (str): Restaurant name.
        city (str): City where the restaurant is located.
        dish (str): Dish to be ordered.

    Returns:
        RestaurantReservation: Restaurant reservation details.
    """
    print(f"Reserving a table at {restaurant} in {city} for {dish} at {reservation_time}")
    reservation = RestaurantReservation(
        reservation_time=datetime.fromisoformat(reservation_time),
        restaurant=restaurant,
        city=city,
        dish=dish,
        cost=randint(10, 50),
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


