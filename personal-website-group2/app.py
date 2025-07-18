from flask import Flask, send_from_directory, request, jsonify
import random
#initilaize the flask app
app = Flask(__name__)

#define routes
@app.route('/')
def home():
    return send_from_directory("", "index.html")

# #serve the static files
@app.route('/<path:filename>')
def static_files(filename):
    return send_from_directory("", filename)

#define fetch greeting route
@app.route('/hello')
def send_hello():
    return "Hey, from Flask"

@app.route('/submit', methods = ['POST'])
def submit():
    data = request.json
    name = data.get('myname')
    return jsonify({"message": f"Hello {name}, I have received your data"})

@app.route('/funfact')
def funfact():
    facts = ["i am 23", "i just graduated", 'i dont know how to swim', "i am good at math"]
    return jsonify({"fun_fact": random.choice(facts)})

@app.route('/simplepost', methods = ['POST'])
def simple_post():
    data = request.json
    message = data.get('message')
    return jsonify({"response": f"Your message has been received: {message}"})

#start the flask server:
if __name__ == "__main__":
    app.run(debug=True)
    
'''
 Task:
    Create a form with an input field where the user can type a message (like “Hello world!”).
    When the user clicks the submit button, your code should:
     Send the message to a Flask POST route.
    Flask should return a response like “Your message has been received: [message]”.
    Display the response message on the webpage.
'''

