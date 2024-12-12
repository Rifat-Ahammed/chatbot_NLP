import json
from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

with open('Data.json', 'r', encoding='utf-8') as file:
    intents = json.load(file)

# Prepare training data
training_sentences = []
training_labels = []


for intent in intents['intents']:
    for example in intent['examples']:
        training_sentences.append(example)
        training_labels.append(intent['intent'])
        

# Label encoding the labels
label_encoder = LabelEncoder()
training_labels_encoded = label_encoder.fit_transform(training_labels)

# Creating a pipeline with TF-IDF and Logistic Regression
model = make_pipeline(TfidfVectorizer(), LogisticRegression())
model.fit(training_sentences, training_labels_encoded)


# Function to predict intent using the trained model
def predict_intent(user_input):
    user_input = user_input.strip().lower()
    prediction = model.predict([user_input])
    intent = label_encoder.inverse_transform(prediction)
    return intent[0]


# Function to get response based on intent, role, and language
def get_response(intent, role, language):
    role = role.capitalize()
    language = language.upper()
    
    for i in intents['intents']:
        if i['intent'] == intent:
            if role in i['responses']:
                role_response = i['responses'].get(role,{})
                response = role_response.get(language, role_response.get('EN'))
                #response = i['responses'][role].get(language, i['responses'][role].get('EN'))
                #print(f"Response found: {response}")
                return response
    return "Does this solve your issue? If not, please talk to our help center.\nAr tai išsprendžia jūsų problemą? Jei ne, susisiekite su mūsų pagalbos centru."


# Language-specific action mapping
action_mapping = {
    'EN': {
        'start': 'start/paleisti',
        'set_language': 'set_language/nustatyti_kalbą',
        'set_user_type': 'set_user_type',
        'choose_issue': 'choose_issue'
    },
    'LT': {
        'start': 'start/paleisti',
        'set_language': 'set_language/nustatyti_kalbą',
        'set_user_type': 'nustatyti_vartotojo_tipą',
        'choose_issue': 'pasirinkti_problema'
    }
}


# Initialize the chatbot flow (state is now local per request)
@app.route('/chatbot', methods=['POST'])
def chatbot():
    data = request.json
    action = data.get('action', '').lower()
    language = data.get('language', 'EN')  # Default to 'EN' if language is not provided
    
    # Initialize the state for each interaction without the user_id
    user_type = data.get('user_type')  # Expecting the user_type to be in the request
    issue = None

    # Normalize user_type to match JSON structure
    if language == 'LT':
        if user_type and user_type.lower() == 'studentas':
            user_type = 'Student'
        elif user_type and user_type.lower() == 'darbuotojas':
            user_type = 'Employee'
    elif language == 'EN':
        if user_type and user_type.lower() == 'student':
            user_type = 'Student'
        elif user_type and user_type.lower() == 'employee':
            user_type = 'Employee'

    # Mapping the action dynamically to the selected language
    if language == 'LT':
        action = action_mapping['LT'].get(action.split('/')[0], action)
    elif language == 'EN':
        action = action_mapping['EN'].get(action.split('/')[0], action)

    # Handling start action for both English and Lithuanian
    if action == action_mapping[language]['start']:
        greeting_message = {
            'message': "Hi! How can I help you?\nLabas! Kaip galiu tau padėti?",
            'options': ['EN', 'LT']  # Language options
        }
        return jsonify(greeting_message)
    
    elif action == action_mapping[language]['set_language']:
        language = data.get('language', 'EN')  # Default to English if not provided
        if language not in ['EN', 'LT']:
            return jsonify({'error': 'Invalid language choice\nNetinkamas kalbos pasirinkimas.'})

        if language == 'EN':
            user_type_message = { 
                'message': 'Please choose your role:',
                'options': ['Student', 'Employee']
            }
        else:  # If language is LT
            user_type_message = {
                'message': 'Prašome pasirinkti savo vaidmenį:',
                'options': ['Studentas', 'Darbuotojas']
            }

        return jsonify(user_type_message)

    elif action == action_mapping[language]['set_user_type']:
        # Normalize user_type based on the language
        user_type_input = data.get('user_type', '').lower()
        if language == 'EN':
            # Normalize English inputs
            if user_type_input == 'student':
                user_type = 'Student'
            elif user_type_input == 'employee':
                user_type = 'Employee'
            else:
                return jsonify({'error': 'Invalid user type. Please select either Student or Employee.'})
        elif language == 'LT':
            # Normalize Lithuanian inputs
            if user_type_input == 'studentas':
                user_type = 'Student'
            elif user_type_input == 'darbuotojas':
                user_type = 'Employee'
            else:
                return jsonify({'error': 'Netinkamas vartotojo tipas. Pasirinkite Studentą arba Darbuotoją.'})

        # Set options based on the user type and language
        options = []
        if user_type == 'Student':
            if language == 'EN':
                options = [
                    'AIS related issue',
                    'Microsoft related issue',
                    'Password related issue'
                ]
            else:  # If language is LT
                options = [
                    'Su AIS susijusi problema',
                    'Su Microsoft susijusi problema',
                    'Su slaptažodžiu susijusi problema'
                ]
        elif user_type == 'Employee':
            if language == 'EN':
                options = [
                    'AIS related issue',
                    'Microsoft related issue',
                    'Password related issue',
                    '2FA issue'
                ]
            else:  # If language is LT
                options = [
                    'Su AIS susijusi problema',
                    'Su Microsoft susijusi problema',
                    'Su slaptažodžiu susijusi problema',
                    '2AF problema'
                ]

        if not options:
            return jsonify({'error': 'No options available for this user type'})

        options_message = {
            'message': 'Please select your issue:\nPasirinkite problemą:',
            'options': options
        }
        return jsonify(options_message)

    elif action == action_mapping[language]['choose_issue']:
        issue = data.get('issue')

        intent = predict_intent(issue)  # Predict the intent using the ML model

        # Get response based on selected issue, role, and language
        response_message = get_response(intent, user_type, language)
        return jsonify({'response': response_message})

    return jsonify({'error': 'Invalid action\nNetinkamas veiksmas'})


if __name__ == '__main__':
    app.run(debug=True)
