# backend/app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_bcrypt import Bcrypt
from datetime import datetime, timedelta
import speech_recognition as sr
from pydub import AudioSegment   #mp3
import psycopg2
import uuid
import os  
import requests
import json  
from logging_config import setup_logging, log_action, log_error
import openai
# from openai.error import APIConnectionError, APIError, RateLimitError, InvalidRequestError
import spacy
import jwt  # Import the PyJWT library
import datetime
from datetime import datetime, timedelta
import requests


# from transformers import pipeline


# Setup logging configuration
# Create 'logs' directory if it doesn't exist
setup_logging()


app = Flask(__name__)
CORS(app)
bcrypt = Bcrypt(app)

# Load the trained model and tokenizer
# model = AutoModelForSeq2SeqLM.from_pretrained("./llm_model")
# tokenizer = AutoTokenizer.from_pretrained("./llm_model")

# Load spaCy model
# nlp = spacy.load('en_core_web_sm')

# Configure OpenAI API key
# openai.api_key = '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'

# Load API key from environment variables
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise ValueError("OpenAI API key not found in environment variables.")

# Define the OpenAI API URL
API_URL = "https://api.openai.com/v1/chat/completions"


# PostgreSQL connection
def get_db_connection():

    conn = psycopg2.connect(
        host='localhost',  # Your database host
        database='Voice_Recognition',  # Your database name
        user='postgres',  # Your database user
        password='Varan@404'  # Your database password
    )
    return conn


# Register a user
@app.route('/api/register', methods=['POST'])
def register():
    data = request.get_json()
    first_name = data['firstName']
    last_name = data['lastName']
    email = data['email']
    mobile = data['mobile']
    password = data['password']
    
    # Hash the password
    password_hash = bcrypt.generate_password_hash(password).decode('utf-8')
    
    # Insert into database
    conn = get_db_connection()
    cur = conn.cursor()
    
    try:
        cur.execute(
            'INSERT INTO users (first_name, last_name, email, mobile, password_hash) VALUES (%s, %s, %s, %s, %s)',
            (first_name, last_name, email, mobile, password_hash)
        )
        conn.commit()
        log_action(f"User registered: {email}")
    except psycopg2.IntegrityError:
        conn.rollback()
        log_error(f"Registration failed: {email} already exists.")
        return jsonify({'error': 'Email already exists'}), 400
    finally:
        cur.close()
        conn.close()

    return jsonify({'message': 'User registered successfully'}), 201

# Login a user
@app.route('/api/login', methods=['POST'])
def login():
    data = request.get_json()
    email = data['email']
    password = data['password']

    # Get user from database
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute('SELECT password_hash FROM users WHERE email = %s', (email,))
    user = cur.fetchone()
    
    if user and bcrypt.check_password_hash(user[0], password):
        log_action(f"Login successful for {email}")
        return jsonify({'message': 'Login successful'}), 200
    else:
        log_error(f"Login failed for {email}")
        return jsonify({'error': 'Invalid credentials'}), 401


@app.route('/api/forgot-password', methods=['POST'])
def forgot_password():
    data = request.get_json()
    email = data['email']

    # Check if user exists in the database
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute('SELECT email FROM users WHERE email = %s', (email,))
    user = cur.fetchone()

    if user:
        # Generate a reset token (in this case, a UUID)
        reset_token = str(uuid.uuid4())
        reset_token_expires = datetime.now() + timedelta(hours=1)  # Token valid for 1 hour
        cur.execute('UPDATE users SET reset_token = %s, reset_token_expires = %s WHERE email = %s', 
            (reset_token, reset_token_expires, email))

        conn.commit()

        # Here, you would send an email with the reset link (mock response for now)
        reset_link = f"http://127.0.0.1:3000/reset-password?token={reset_token}"
        print(f"Send email with reset link: {reset_link}")
        log_action(f"Password reset link sent to {email}")
        return jsonify({'message': 'Password reset link sent', 'reset_link': reset_link}), 200
    else:
        log_error(f"Password reset failed for {email}: user not found")
        return jsonify({'error': 'User not found'}), 404


# Reset password (where the user can submit a new password using the reset token)
@app.route('/api/reset-password', methods=['POST'])
def reset_password():
    data = request.get_json()

    # Input validation
    if not data or 'token' not in data or 'newPassword' not in data:
        log_error("Reset password request missing token or new password")
        return jsonify({'error': 'Missing token or new password'}), 400

    reset_token = data['token']
    new_password = data['newPassword']

    # Validate password strength (example: length, complexity, etc.)
    if len(new_password) < 8:
        return jsonify({'error': 'Password must be at least 8 characters long.'}), 400

    # Check if the reset token is valid
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute('SELECT email FROM users WHERE reset_token = %s', (reset_token,))
        user = cur.fetchone()

        if user:
            # Hash the new password
            password_hash = bcrypt.generate_password_hash(new_password).decode('utf-8')

            # Update the user's password and clear the reset token
            cur.execute('UPDATE users SET password_hash = %s, reset_token = NULL WHERE reset_token = %s', 
                        (password_hash, reset_token))
            conn.commit()
            log_action(f"Password reset successful for {user[0]}")
            return jsonify({'message': 'Password has been reset successfully'}), 200
        else:
            log_error(f"Password reset failed for token {reset_token}: invalid/expired")
            return jsonify({'error': 'Invalid or expired token'}), 400

    except Exception as e:
        # Handle exceptions (e.g., database errors)
        conn.rollback()  # Rollback in case of error
        log_error(f"Database error during password reset: {e}")
        return jsonify({'error': str(e)}), 500

    finally:
        cur.close()
        conn.close()


# Ensure the temp directory exists
def ensure_temp_directory():
    if not os.path.exists('temp'):
        os.makedirs('temp')

def convert_to_wav(audio_file):
    # Ensure temp directory exists
    ensure_temp_directory()

    # Save the file temporarily
    file_path = os.path.join('temp', audio_file.filename)
    audio_file.save(file_path)

    # Convert file to wav
    wav_file_path = file_path.rsplit('.', 1)[0] + '.wav'
    audio = AudioSegment.from_file(file_path)
    audio.export(wav_file_path, format="wav")
    return wav_file_path

@app.route('/api/voice-to-claim', methods=['POST'])
def voice_to_claim():
    if 'audio' not in request.files:
        log_error("No audio file provided for voice-to-claim")
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files['audio']
    recognizer = sr.Recognizer()

    try:
        # Convert to WAV format
        wav_file_path = convert_to_wav(audio_file)

        with sr.AudioFile(wav_file_path) as source:
            audio_data = recognizer.record(source)
            transcription = recognizer.recognize_google(audio_data)
            print("Transcription:", transcription)
            
            claim_payload = generate_claim_payload(transcription)
            print("Claim Payload:", claim_payload)

            # Save audio file locally
            audio_file_path = save_audio_file(audio_file)

            # Store in database (implement as needed)
            store_claim_data(audio_file_path, transcription, claim_payload)

            log_action("Claim created from voice input")
            return jsonify({"transcription": transcription, "claimPayload": claim_payload})

    except sr.UnknownValueError:
        log_error("Audio could not be understood")
        return jsonify({"error": "Could not understand the audio"}), 400

    except sr.RequestError as e:
        log_error(f"Speech recognition API error: {e}")
        return jsonify({"error": f"API error: {e}"}), 500

    except Exception as e:
        log_error(f"Error processing audio: {e}")
        return jsonify({"error": "Could not process audio", "details": str(e)}), 500



# def generate_claim_payload(transcription):
#     return {
#         "claimant_name": "John Doe",
#         "policy_number": "123456789",
#         "transcription": transcription
#     }

# # Open Ai to generate Playload 

# def extract_entities(text):
#     doc = nlp(text)
#     entities = {ent.label_: ent.text for ent in doc.ents}
#     return entities



# def generate_claim_payload(transcription):
#     prompt = f"Generate a claim payload based on the following data: {transcription}"
#     response = openai.Completion.create(
#         model="gpt-4",
#         # model="gpt-4-32k"

#         # model="gpt-3.5-turbo",
#         prompt=prompt,
#         max_tokens=200
#     )
#     return response.choices[0].text   


# def generate_claim_payload(transcription):
#     try:
#         prompt = f"Generate a claim payload based on the following data: {transcription}"
        
#         response = openai.Completion.create(
#             model="gpt-4",  # Or any available model
#             prompt=prompt,
#             max_tokens=200
#         )
        
#         return response.choices[0].text

#     except InvalidRequestError as e:
#         if "insufficient_quota" in str(e):
#             print("Error: Insufficient quota. Please check your OpenAI plan and billing details.")
#         else:
#             print(f"Invalid request: {e}")
#     except APIConnectionError as e:
#         print(f"Error: Failed to connect to OpenAI API: {e}")
#     except APIError as e:
#         print(f"Error: OpenAI API error occurred: {e}")
#     except RateLimitError as e:
#         print(f"Error: Rate limit exceeded. Try again later: {e}")
#     except Exception as e:
#         print(f"An unexpected error occurred: {e}")
  

    
def generate_claim_payload(transcription):
   
    # Construct the request payload
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    
    payload = {
        "model": "gpt-4",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant for generating insurance claim payloads."},
            {"role": "user", "content": f""" 
            
            Generate a claim payload based on the following data: {transcription}" 
Ensure that all fields listed below are included in the output JSON. If data for any field is missing or unknown, use the value `null`. 
The output must strictly be in valid JSON format with no additional text, comments, or explanations. 

The fields required in the payload are:

You are a claims handler and are provided with a text file with customer's claim details. All fields except IncidentSubTypeID, IncidentTypeID, FaultRatingID should be populated from the text. For IncidentSubTypeID and IncidentTypeID identify the closest IncidentSubType from IncidentSubType key value pairs and assign IncidentSubTypeID and IncidentTypeID accordingly.
For FaultRatingID please use the FaultRating key value pair.
Ensure that all fields listed below are included in the output JSON. If data for any field is missing or unknown, use the value `null`.
The output must strictly be in valid JSON format with no additional text, comments, or explanations.
 
policy_number - Insurance Policy Number (varchar)
incident_discovery_date - The date and time of the loss in ISO 8601 format (e.g., YYYY-MM-DD).
reported_by - Name of customer reporting the claim (Varchar)
reported_via - should always be VoiceClaims (Varchar)
claimant - Name of insurer which should be name of person reporting the claim unless explicitly mentioned (Varchar)
incident_description - Description of incident (Text)
incident_location - Complete address of incident (Varchar)
incident_type_id - Pick from below table (Int)
incident_subtype_id - Pick from below table (Int)
fault_rating - Pick from below table (Int)
contact_name - Name of person reporting the claim (Varchar)
contact_phone_number - Contact of person reporting the claim (Varchar)
contact_email - Email of person reporting the claim (Varchar)
contact_address - Address of person reporting the claim (Text)contact_pincode -> Pincode of person reporting the claim (Varchar)
sessionUser - cbareh1
 
IncidentSubType key value pair:
IncidentSubType, IncidentSubTypeID, IncidentTypeID
Critical Incident, 2, 1
Ransomware, 3, NULL
Virus, 4, NULL
Email Phishing, 5, NULL
SMS Phishing, 6, NULL
Fake websites, 7, NULL
Phony Job offers, 8, NULL
Mobile Payment Exploits, 9, NULL
ATM Skimming, 10, NULL
Personal Data Theft, 11, NULL
Financial Data Theft, 12, NULL
Ransomware, 13, 1
Virus, 14, 1
Email Phishing, 15, 2
SMS Phishing, 16, 2
Fake websites, 17, 3
Phony Job offers, 18, 3
Mobile Payment Exploits, 19, 4
ATM Skimming, 20, 4
Personal Data Theft, 21, 5
Financial Data Theft, 22, 5
 
FaultRating key value pair:
Insured at Fault: 1
Under investigation: 2
Third party at fault: 3
Others: 4
has context menu


continue the response generation without asking for further acknowledgement please send me only json request not anything else.
            
            """}
        ],
        "max_tokens": 1000
    }
   
    try:
        # Make the API request
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()  # Raise an exception for HTTP errors
        result = response.json()
        claim_details = result["choices"][0]["message"]["content"].strip()
        print(claim_details)

        return claim_details
   
    except requests.exceptions.RequestException as e:
        return f"Request error: {e}"
    except KeyError:
        return "Unexpected response format from OpenAI API."


def save_audio_file(audio_file):
    # Save the audio file to the server
    audio_file_path = f'./recordings/{audio_file.filename}'  # Update this path as needed
    audio_file.save(audio_file_path)
    return audio_file_path

def store_claim_data(audio_path, transcription, claim_payload):
    conn = get_db_connection()
    cur = conn.cursor()

    # Insert the claim data into the database
    cur.execute("""
        INSERT INTO claims (audio_path, transcription, claim_payload)
        VALUES (%s, %s, %s)
    """, (audio_path, transcription, json.dumps(claim_payload)))

    conn.commit()
    cur.close()
    conn.close()


# model = pipeline('text-generation', model='your-llm-model')

@app.route('/api/validate-policy', methods=['POST'])
def validate_policy():
    data = request.get_json()
    policy_number = data.get('policyNumber')
    # Validate the policy number (dummy validation for example)
    if policy_number == '123456':
        log_action(f"Policy {policy_number} validated successfully")
        return jsonify({'valid': True})
    else:
        log_error(f"Policy validation failed for {policy_number}")
        return jsonify({'valid': False})


@app.route('/api/generate-response', methods=['POST'])
def generate_response():
    data = request.get_json()
    user_input = data.get('input')
    #  response = model(user_input)
    # Placeholder response
    response = f"Simulated response for input: {user_input}"
    log_action(f"Generated response for input: {user_input}")
    return jsonify({'response': response})


# def generate_claim_number():
#     """Generate a unique claim number."""
#     return f"CLM-{uuid.uuid4().hex[:8].upper()}"

def generate_claim_number():
    """Generate a unique and incrementing claim number."""
    try:
        # Get a database connection
        conn = get_db_connection()
        cur = conn.cursor()

        # Fetch the next value from the sequence
        cur.execute("SELECT nextval('claim_number_seq')")
        next_val = cur.fetchone()[0]

        # Generate the claim number with a prefix
        claim_number = f"CLM-{next_val:06d}"  # Example: CLM-000123

        cur.close()
        conn.close()

        return claim_number

    except Exception as e:
        logging.error(f"Error generating claim number: {e}")
        raise



def store_claim_details(claim_number, claim_payload):
    """Stores the claim data along with the claim number in the database."""
    try:
        conn = get_db_connection()  # Get the database connection
        cur = conn.cursor()

        # Insert claim number and claim data into the claims table
        query = """
            INSERT INTO claim_details (claim_number, claim_payload)
            VALUES (%s, %s)
        """
        cur.execute(query, (claim_number, json.dumps(claim_payload)))  # Insert claim number and payload as JSON

        conn.commit()  # Commit the transaction
        cur.close()
        conn.close()

    except Exception as e:
        print(f"Error storing claim data: {e}")
        raise  # Raise the exception to handle it in the API

@app.route('/api/submit-claim', methods=['POST'])
def submit_claim():
    """API endpoint to submit claim data."""
    try:
        # Parse incoming JSON data from the request
        claim_payload = request.get_json()  # Get claim data from the request body


        # URL of the API endpoint
        url = "https://1qn12sfq-5000.inc1.devtunnels.ms/api/create-vocal-claim"

        # Headers (optional, adjust as necessary)
        headers = {
             "Content-Type": "application/json"
        }       

    # Sending the POST request
        response = requests.post(url, json=claim_payload, headers=headers)

        # Generate a claim number
        claim_number = response.json()['claim_number']
        # generate_claim_number()  # Call the function to generate a unique claim number

        # Store the claim number and payload in the database
        store_claim_details(claim_number, claim_payload)

        # Respond with a success message and the claim number
        return jsonify({
            'success': True,
            'claimNumber': claim_number,
            'message': 'Claim submitted successfully!'
        }), 200

    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error submitting claim: {str(e)}'
        }), 500

# Function to fetch claim details by claim number
def get_claim_details(claim_number):
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        query = """
            SELECT claim_number, claim_payload
            FROM claim_details
            WHERE claim_number = %s
        """
        cur.execute(query, (claim_number,))
        result = cur.fetchone()
        cur.close()
        conn.close()
        if result:
            return {
                "claimNumber": result[0],
                "claimPayload": result[1]
            }
        return None
    except Exception as e:
        print(f"Error fetching claim data: {e}")
        raise

# API to fetch claim details by claim number
@app.route('/api/get-claim', methods=['GET'])
def fetch_claim():
    try:
        claim_number = request.args.get('claimNumber')
        if not claim_number:
            return jsonify({'success': False, 'message': 'Claim number is required'}), 400
        claim_details = get_claim_details(claim_number)
        if claim_details:
            return jsonify({'success': True, 'claimDetails': claim_details}), 200
        return jsonify({'success': False, 'message': 'Claim not found'}), 404
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error fetching claim: {str(e)}'
        }), 500


if __name__ == '__main__':
    app.run(debug=True)

