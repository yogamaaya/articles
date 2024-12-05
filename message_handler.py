from flask import request, jsonify
from chat import test_on_submit
messages = []


def receive_message():
    if request.content_type == 'application/json':
        data = request.get_json()
        message = data.get('message', '')
        print("message received ", message)
        if message:
            messages.append(message)
            reply = test_on_submit(message)
            messages.append(reply)
            print("response: ", messages)
            return jsonify({'message': message, 'reply': reply, 'messages': messages})
        return jsonify({'error': 'Empty message'}), 400
    return jsonify({'error': 'Invalid Content-Type'}), 400