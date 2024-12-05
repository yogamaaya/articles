from flask import Flask, request, jsonify, render_template, redirect, url_for
from message_handler import receive_message
app = Flask(__name__)
# Simple storage to keep messages in memory
messages = []

@app.route('/')
def chat():
    return render_template('chat.html')
@app.route('/submit', methods=['POST', 'GET'])
def submit_message():
    new_messages = receive_message()
    global messages
    messages = new_messages  # Update the messages with the latest messages
    print("HELLO FROM /submit call! msg received")
    print(messages)
    return new_messages

@app.route('/chat', methods=['GET'])
def display_chat():
    return render_template('chat.html', messages=messages)


if __name__ == '__main__':
    app.run(debug=True)
