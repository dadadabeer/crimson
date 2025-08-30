from flask import Flask, request, send_file, jsonify
from openai import OpenAI
import os
import re
import json
from dotenv import load_dotenv

# Load environment variables from config.env file
load_dotenv('config.env')

# Initialize OpenAI client with API key from environment
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Initialize conversation history
conversation = [{"role": "system", "content": "You are a helpful financial assistant. When users share financial information, ALWAYS acknowledge it in this exact format: 'I see your monthly income is $X, monthly expenses are $Y, monthly savings are $Z, and total assets are $W.' Then provide brief, actionable advice. Keep responses concise and focused."}]

# Financial data storage
financial_data = {
    "total_assets": 0,
    "monthly_income": 0,
    "monthly_expenses": 0,
    "monthly_savings": 0
}

def extract_financial_data(text):
    """Extract financial numbers from text using enhanced regex patterns"""
    extracted_data = {}
    
    # Enhanced patterns for different financial metrics
    patterns = {
        'total_assets': [
            r'total assets?[:\s]*\$?([\d,]+(?:\.\d{2})?)',
            r'assets?[:\s]*\$?([\d,]+(?:\.\d{2})?)',
            r'\$?([\d,]+(?:\.\d{2})?)[\s]*in assets?',
            r'net worth[:\s]*\$?([\d,]+(?:\.\d{2})?)',
            r'have[:\s]*\$?([\d,]+(?:\.\d{2})?)[\s]*(?:in assets?|total)',
            r'worth[:\s]*\$?([\d,]+(?:\.\d{2})?)'
        ],
        'monthly_income': [
            r'monthly income[:\s]*\$?([\d,]+(?:\.\d{2})?)',
            r'income[:\s]*\$?([\d,]+(?:\.\d{2})?)',
            r'\$?([\d,]+(?:\.\d{2})?)[\s]*per month',
            r'\$?([\d,]+(?:\.\d{2})?)[\s]*monthly',
            r'(?:make|earn)[:\s]*\$?([\d,]+(?:\.\d{2})?)[\s]*(?:per month|monthly|a month)',
            r'salary[:\s]*\$?([\d,]+(?:\.\d{2})?)'
        ],
        'monthly_expenses': [
            r'monthly expenses?[:\s]*\$?([\d,]+(?:\.\d{2})?)',
            r'expenses?[:\s]*\$?([\d,]+(?:\.\d{2})?)',
            r'\$?([\d,]+(?:\.\d{2})?)[\s]*in expenses?',
            r'\$?([\d,]+(?:\.\d{2})?)[\s]*monthly expenses?',
            r'spend[:\s]*\$?([\d,]+(?:\.\d{2})?)[\s]*(?:per month|monthly|a month)',
            r'\$?([\d,]+(?:\.\d{2})?)[\s]*(?:on expenses?|spending)'
        ],
        'monthly_savings': [
            r'monthly savings?[:\s]*\$?([\d,]+(?:\.\d{2})?)',
            r'savings?[:\s]*\$?([\d,]+(?:\.\d{2})?)',
            r'\$?([\d,]+(?:\.\d{2})?)[\s]*in savings?',
            r'\$?([\d,]+(?:\.\d{2})?)[\s]*monthly savings?',
            r'save[:\s]*\$?([\d,]+(?:\.\d{2})?)[\s]*(?:per month|monthly|a month)',
            r'\$?([\d,]+(?:\.\d{2})?)[\s]*saved'
        ]
    }
    
    for metric, pattern_list in patterns.items():
        for pattern in pattern_list:
            match = re.search(pattern, text.lower())
            if match:
                # Convert string to number, removing commas
                value_str = match.group(1).replace(',', '')
                try:
                    value = float(value_str)
                    extracted_data[metric] = value
                    print(f"Backend extracted {metric}: {value}")  # Debug logging
                    break
                except ValueError:
                    continue
    
    return extracted_data

def update_financial_data(new_data):
    """Update financial data with new extracted values"""
    global financial_data
    for key, value in new_data.items():
        if key in financial_data and value > 0:
            financial_data[key] = value
    
    # Update AI context with new financial information
    update_ai_financial_context()

def update_ai_financial_context():
    """Update the AI's context with current financial data"""
    global conversation, financial_data
    
    # Create a financial summary for the AI
    if any(value > 0 for value in financial_data.values()):
        financial_summary = f"User's current financial situation: "
        if financial_data['total_assets'] > 0:
            financial_summary += f"Total Assets: ${financial_data['total_assets']:,.0f}, "
        if financial_data['monthly_income'] > 0:
            financial_summary += f"Monthly Income: ${financial_data['monthly_income']:,.0f}, "
        if financial_data['monthly_expenses'] > 0:
            financial_summary += f"Monthly Expenses: ${financial_data['monthly_expenses']:,.0f}, "
        if financial_data['monthly_savings'] > 0:
            financial_summary += f"Monthly Savings: ${financial_data['monthly_savings']:,.0f}"
        
        # Remove trailing comma and space
        financial_summary = financial_summary.rstrip(', ')
        
        # Update or add financial context to conversation
        context_message = {"role": "system", "content": financial_summary}
        
        # Check if we already have a financial context message
        context_updated = False
        for i, msg in enumerate(conversation):
            if msg["role"] == "system" and "User's current financial situation" in msg["content"]:
                conversation[i] = context_message
                context_updated = True
                break
        
        # If no existing context message, add it
        if not context_updated:
            conversation.append(context_message)

def get_simple_response(message):
    """Fallback simple chatbot responses when OpenAI API is unavailable"""
    message = message.lower().strip()
    
    if any(word in message for word in ['hello', 'hi', 'hey']):
        return "Hello! How can I help you today?"
    elif any(word in message for word in ['how are you', 'how r u']):
        return "I'm doing well, thank you for asking! How can I assist you?"
    elif any(word in message for word in ['bye', 'goodbye', 'see you']):
        return "Goodbye! Have a great day!"
    elif any(word in message for word in ['help', 'what can you do']):
        return "I'm a simple chatbot. I can help with basic conversations, answer questions, and assist with various topics. What would you like to know?"
    elif any(word in message for word in ['weather', 'temperature']):
        return "I don't have access to real-time weather data, but I'd be happy to help with other questions!"
    elif any(word in message for word in ['name', 'who are you']):
        return "I'm an AI assistant chatbot. I'm here to help answer your questions and have conversations!"
    elif '?' in message:
        return "That's an interesting question! I'm currently running in fallback mode, but I'd be happy to help with what I can."
    else:
        return "I understand you said: '" + message + "'. I'm currently running in simple mode. How can I help you?"


def get_response(message):
    try:
        conversation.append({"role": "user", "content": message})
        response = client.chat.completions.create(
            model=os.getenv('OPENAI_MODEL', 'gpt-3.5-turbo'),
            messages=conversation,
            max_tokens=int(os.getenv('MAX_TOKENS', 150)),
            temperature=float(os.getenv('TEMPERATURE', 0.7))
        )
        reply = response.choices[0].message.content
        conversation.append({"role": "assistant", "content": reply})
        
        # Extract financial data from the response
        extracted_data = extract_financial_data(reply)
        if extracted_data:
            update_financial_data(extracted_data)
        
        return reply
    except Exception as e:
        # Handle different types of errors and use fallback
        if "insufficient_quota" in str(e) or "quota" in str(e).lower():
            fallback_response = "I'm currently using my fallback mode due to API limits. " + get_simple_response(message)
            # Extract financial data from fallback response too
            extracted_data = extract_financial_data(fallback_response)
            if extracted_data:
                update_financial_data(extracted_data)
            return fallback_response
        elif "rate_limit" in str(e).lower():
            fallback_response = "I'm receiving too many requests right now. Using fallback mode: " + get_simple_response(message)
            extracted_data = extract_financial_data(fallback_response)
            if extracted_data:
                update_financial_data(extracted_data)
            return fallback_response
        elif "authentication" in str(e).lower() or "invalid" in str(e).lower():
            fallback_response = "I'm using fallback mode due to authentication issues. " + get_simple_response(message)
            extracted_data = extract_financial_data(fallback_response)
            if extracted_data:
                update_financial_data(extracted_data)
            return fallback_response
        else:
            fallback_response = "I'm using fallback mode due to an error. " + get_simple_response(message)
            extracted_data = extract_financial_data(fallback_response)
            if extracted_data:
                update_financial_data(extracted_data)
            return fallback_response


app = Flask(__name__)


@app.route('/')
def home():
    return send_file('chatbot.html')

@app.route('/ask', methods=['POST'])
def ask():
    try:
        message = request.form['message']
        if not message or message.strip() == "":
            return "Please enter a valid message."
        
        # Extract financial data from user's message BEFORE processing
        user_financial_data = extract_financial_data(message)
        if user_financial_data:
            print(f"Extracted from user message: {user_financial_data}")
            update_financial_data(user_financial_data)
        
        response = get_response(message)
        
        # Return both the response and current financial data
        return jsonify({
            'response': response,
            'financial_data': financial_data
        })
    except Exception as e:
        return jsonify({
            'response': f"Sorry, something went wrong: {str(e)[:100]}...",
            'financial_data': financial_data
        })

@app.route('/financial-data', methods=['GET'])
def get_financial_data():
    """Endpoint to get current financial data"""
    return jsonify(financial_data)

@app.route('/update-financial-data', methods=['POST'])
def update_financial_data_endpoint():
    """Endpoint to manually update financial data"""
    try:
        data = request.get_json()
        if data:
            update_financial_data(data)
            # AI context is automatically updated in update_financial_data function
        return jsonify({'success': True, 'financial_data': financial_data})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == "__main__":
    app.run(
        debug=os.getenv('FLASK_DEBUG', 'True').lower() == 'true',
        host=os.getenv('FLASK_HOST', '0.0.0.0'),
        port=int(os.getenv('FLASK_PORT', 5001))
    )