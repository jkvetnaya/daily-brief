from openai import OpenAI
import json
from dotenv import load_dotenv
import os
import requests

load_dotenv()

client = OpenAI()


def get_weather(city):
    """Get the weather for a given city"""
    base_url = "https://api.openweathermap.org/data/2.5/weather"
    params = {
        "q": city,
        "appid": os.getenv('WEATHER_API_KEY'),
        "units": "metric"  # Use metric units for temperature
    }
    try:
        response = requests.get(base_url, params=params)
        weather_data = response.json()
    except requests.exceptions.RequestException as e:
        raise Exception(f"Error fetching weather data: {e}")
        return None
    if response.status_code != 200:
        raise Exception(f"Error fetching weather data: {response.status_code}")
    data = response.json()

    if "main" not in weather_data:
        raise Exception("Invalid response from weather API")
    temperature = weather_data["main"]["temp"]
    weather_description = weather_data["weather"][0]["description"]
    # Format the weather information
    return json.dumps({
        "city": city,
        "weather": weather_description,
        "temperature": f"{temperature}°C"
    })
    

def get_top_headlines():
    """Get the top headlines"""
    api_key = os.getenv('NEWS_API_KEY')
    url = f"https://newsapi.org/v2/top-headlines?country=us&apiKey={api_key}"

    try:
        response = requests.get(url)
        data = response.json()
    except requests.exceptions.RequestException as e:
        raise Exception(f"Error fetching news data: {e}")
        return None

    return json.dumps({
        "headlines": [
            article["title"] for article in data["articles"]
            if "title" in article   
        ]
    })


def synthesize_briefing():
    # Step 1: send the conversation and available functions to the model
    user = """Create a daily briefing for me that includes:
    1. Today's weather in San Francisco
    2. Top news headlines
    Organize this information in a creative story not more than 200 words."""
    messages = [{"role": "user", "content": user}]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the weather for a given city.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                        },
                    },
                    "required": ["city"],
                },
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_top_headlines",
                "description": "Get the top news headlines.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            }
        }
       ]
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        messages=messages,
        tools=tools,
        tool_choice="auto",  # auto is default, but we'll be explicit
    )
    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls
    # Step 2: check if the model wanted to call a function
    if tool_calls:
        # Step 3: call the function
        # Note: the JSON response may not always be valid; be sure to handle errors
        available_functions = {
            "get_weather": get_weather,
            "get_top_headlines": get_top_headlines,
        }  # only one function in this example, but you can have multiple
        messages.append(response_message)  # extend conversation with assistant's reply
        # Step 4: send the info for each function call and function response to the model
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_to_call = available_functions[function_name]
            function_args = json.loads(tool_call.function.arguments)
            if function_name == "get_weather":
                function_response = function_to_call(
                    city=function_args.get("city"),
                )
            else:
                function_response = function_to_call()
            messages.append(
                {
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": function_response,
                }
            )  # extend conversation with function response
        second_response = client.chat.completions.create(
            model="gpt-3.5-turbo-1106",
            messages=messages,
        )  # get a new response from the model where it can see the function response
        return second_response.choices[0].message.content
        return second_response.choices[0].message.content

print(synthesize_briefing())