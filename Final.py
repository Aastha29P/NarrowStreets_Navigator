from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
import os
from dotenv import load_dotenv
import logging
import pandas as pd
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize OpenAI client
api_key = os.getenv("OPENAI_API_KEY")
logger.info("Checking API key...")
if not api_key:
    logger.error("OpenAI API key not found!")
    raise ValueError("OpenAI API key not found")

client = OpenAI(api_key=api_key)
logger.info("OpenAI client initialized")

# Load and process the tourist places data
try:
    df = pd.read_csv('Top_Indian_Places_to_Visit.csv')
    places_data = {}
    for _, row in df.iterrows():
        place_name = row['Name'].lower()
        places_data[place_name] = {
            'name': row['Name'],
            'city': row['City'],
            'state': row['State'],
            'entry_fee': row['Entrance Fee in INR'],
            'weekly_off': row['Weekly Off'],
            'dslr_allowed': row['DSLR Allowed'],
            'best_time': row['Best Time to visit'],
            'time_needed': row['time needed to visit in hrs'],
            'type': row['Type'],
            'rating': row['Google review rating']
        }
    logger.info("Tourist places data loaded successfully")
except Exception as e:
    logger.error(f"Error loading tourist places data: {str(e)}")
    places_data = {}

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:5174", "http://localhost:5175"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatMessage(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str

# Global context to store the last discussed place
last_place_context = {"name": None}

def handle_follow_up_question(message: str) -> str:
    """Handle follow-up questions about previously mentioned places"""
    follow_up_indicators = [
        "tell me more",
        "what about",
        "how about",
        "more details",
        "what else",
        "and",
        "about it"
    ]
    
    if any(indicator in message.lower() for indicator in follow_up_indicators) and last_place_context["name"]:
        return f"tell me about {last_place_context['name']}"
    return message

def get_place_info(message):
    """Extract place information if mentioned in the message"""
    place_info = []
    message_lower = message.lower()
    
    for place_name, data in places_data.items():
        if place_name in message_lower:
            info = []
            if data['entry_fee'] > 0:
                info.append(f"🎟️ Ticket price: ₹{data['entry_fee']} (totally worth the experience!) 💫")
            else:
                info.append("🎟️ Amazing news - Entry is absolutely free! 🎉 💝")
                
            if data['weekly_off'] != 'None':
                info.append(f"📅 Quick tip: Plan around their weekly off on {data['weekly_off']} 📌")
            
            info.append(f"📸 Photography lovers: DSLR cameras are {'📷 welcome here! 🤳' if data['dslr_allowed'] == 'Yes' else '📱 not allowed, but your phone camera works fine! 🤳'}")
            info.append(f"⏰ Pro tip: Best time to visit is {data['best_time']} 🌟")
            info.append(f"⌛ Suggested duration: {data['time_needed']} hours to soak in all the beauty ✨")
            info.append(f"⭐ Visitor rating: {data['rating']}/5 - {get_rating_comment(data['rating'])} 🏆")
            
            place_info.append({
                'name': data['name'],
                'info': info
            })
    
    return place_info

def get_rating_comment(rating):
    """Get enthusiastic comment based on rating"""
    if rating >= 4.7:
        return "Absolutely fantastic! 🌟 🎯 💯"
    elif rating >= 4.5:
        return "Visitors love it! 🎉 ⭐"
    elif rating >= 4.0:
        return "Really great place! 🌟 👍"
    else:
        return "Worth checking out! ✨ 👌"

def format_itinerary_prompt(current_day, max_days):
    """Format enthusiastic prompts for itinerary continuation"""
    if not current_day:
        return "Let me craft an amazing itinerary for you! 🎨 I'll include local insights 🗺️, hidden gems 💎, and practical tips to make your trip unforgettable! 🌟"
    
    prompts = [
        f"Now, let me tell you about the exciting activities for Day {current_day}! 🎯 ✨",
        f"Let's continue with Day {current_day} of your adventure! 🚀 🌟",
        f"You'll love what I have planned for Day {current_day}! 🎉 💫",
        "Here's what's next on your incredible journey! 🗺️ 🌈",
        "Wait until you hear about the next amazing experiences! 🎨 ✨"
    ]
    return prompts[current_day % len(prompts)]

def format_response(content, is_ai=True):
    """Format response with appropriate icons and styling"""
    prefix = "✈️ " if is_ai else "🌎 "
    return prefix + content

# Define the system prompt with a more enthusiastic and friendly tone
SYSTEM_PROMPT = """Hi there! 🌟 I'm your super enthusiastic travel buddy, and I'm absolutely thrilled to help you discover the magic of India! 🎉 

I LOVE sharing exciting travel tips and creating personalized experiences! 💝 Here's what makes our chat special:

✨ When you ask about places:
• I'll share fascinating details with genuine excitement! 🎯
• Give you insider tips that locals know and love 🗺️
• Tell you fun stories and interesting facts 📚
• Help you avoid tourist traps and find hidden gems 💎

🎨 Feel free to ask me anything like:
• "Tell me about [any place]!" 🏛️
• "What's the best time to visit [place]?" ⏰
• "Help me plan a trip to [destination]!" 🗺️
• "What should I not miss in [city]?" 🎯

💫 What makes me special:
• I keep our chat fun and friendly - like talking to a friend who knows all the cool spots! 🤝
• I'm always excited to share the latest info about timings, fees, and special events 📅
• I love suggesting unique experiences based on your interests 🎨
• I'll help you make the most of your time with smart tips and tricks 💡

🌈 My Style:
• Super friendly and always excited to help 🤗
• Clear and organized information with a fun twist 📝
• Interactive - I love when you ask follow-up questions! 💭
• Practical tips mixed with fun facts 🎯

Remember: No question is too small - I'm here to make your travel planning fun and exciting! Let's explore together! 🚀 ✈️ 🌏"""

@app.post("/api/chat", response_model=ChatResponse)
async def chat(chat_message: ChatMessage):
    try:
        logger.info(f"Received message: {chat_message.message}")
        
        # Handle follow-up questions
        processed_message = handle_follow_up_question(chat_message.message)
        logger.info(f"Processed message: {processed_message}")
        
        # Check if it's an itinerary request
        is_itinerary = any(keyword in processed_message.lower() 
                         for keyword in ["itinerary", "trip", "plan", "visit", "days", "curate"])
        
        # Get place-specific information
        place_info = get_place_info(processed_message)
        
        # Update context if a specific place is mentioned
        if place_info and len(place_info) > 0:
            last_place_context["name"] = place_info[0]["name"].lower()
            logger.info(f"Updated context to: {last_place_context['name']}")
        
        # Combine all information with enthusiasm
        enhanced_message = format_response(processed_message, is_ai=False)
        if place_info:
            enhanced_message += "\n\n✨ Fantastic choice! Let me share some exciting details about these places! 🎉\n"
            for place in place_info:
                enhanced_message += f"\n🌟 {place['name']} - You're going to love this! 💫\n" + "\n".join(place['info']) + "\n"
            enhanced_message += "\n💭 Curious about anything else? Ask me about:\n"
            enhanced_message += "• 🚗 Best ways to get there\n"
            enhanced_message += "• 💎 Hidden gems nearby\n"
            enhanced_message += "• 🍽️ Local food recommendations\n"
            enhanced_message += "• 🎯 Best photo spots\n"
            enhanced_message += "• 🛍️ Shopping opportunities\n"
            enhanced_message += "• 🎨 Cultural experiences\n"
            enhanced_message += "Or anything else you'd like to know! 🌟 ✨"

        try:
            if is_itinerary:
                # For itineraries, generate in chunks to ensure completion
                full_response = ""
                current_day = 1
                max_days = 7  # Maximum number of days to generate
                
                while current_day <= max_days:
                    prompt = format_itinerary_prompt(current_day if full_response else None, max_days)
                    if full_response:
                        prompt += f" Continue from Day {current_day}, making sure to include all the exciting details! 🎯 ✨"
                    else:
                        prompt = enhanced_message
                    
                    messages = [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "assistant", "content": full_response} if full_response else {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt}
                    ]
                    
                    response = client.chat.completions.create(
                        model="ft:gpt-4o-mini-2024-07-18:personal::Ado77Ab6",
                        messages=messages,
                        temperature=0.8,
                        max_tokens=2000,
                        presence_penalty=0.6,
                        frequency_penalty=0.2
                    )
                    
                    chunk_content = response.choices[0].message.content
                    full_response += ("\n" if full_response else "") + chunk_content
                    
                    days_found = re.findall(r"Day (\d+)", chunk_content)
                    if days_found:
                        current_day = max(map(int, days_found)) + 1
                    else:
                        break
                    
                    if any(marker in chunk_content.lower() for marker in [
                        "end of itinerary",
                        "enjoy your trip",
                        "safe travels",
                        "final day",
                        f"day {max_days}"
                    ]):
                        full_response += "\n\n✨ I'm so excited for your upcoming adventure! 🎉 Feel free to ask me any questions about:\n"
                        full_response += "• 🏛️ Specific places in the itinerary\n"
                        full_response += "• 🎭 Local customs and traditions\n"
                        full_response += "• 🚗 Transportation options\n"
                        full_response += "• 🍽️ Food recommendations\n"
                        full_response += "• 📸 Photography spots\n"
                        full_response += "• 🛍️ Shopping areas\n"
                        full_response += "Or anything else to make your trip amazing! 🌟 ✨ 🚀"
                        break
                
                response_content = format_response(full_response, is_ai=True)  # AI response with plane
                
            else:
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": enhanced_message}
                ]
                
                response = client.chat.completions.create(
                    model="ft:gpt-4o-mini-2024-07-18:personal::Ado77Ab6",
                    messages=messages,
                    temperature=0.9,
                    max_tokens=800
                )
                response_content = response.choices[0].message.content
                
                if not response_content.endswith(("!", "?", ".")):
                    response_content += "! ✨"
                response_content += "\n\n💫 Would you like to know more? Feel free to ask about:\n"
                response_content += "• 🎯 Specific details\n"
                response_content += "• 📸 Photo opportunities\n"
                response_content += "• 🍽️ Food recommendations\n"
                response_content += "• 🎨 Cultural experiences\n"
                response_content += "I'm here to help! 🌟 🚀"
                
                response_content = format_response(response_content, is_ai=True)
            
            return ChatResponse(response=response_content)
            
        except Exception as openai_error:
            logger.error(f"OpenAI API error: {str(openai_error)}")
            raise HTTPException(
                status_code=500,
                detail=f"OpenAI API error: {str(openai_error)}"
            )
            
    except Exception as e:
        logger.error(f"General error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)


