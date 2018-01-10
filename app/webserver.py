import json
from http.server import BaseHTTPRequestHandler, HTTPServer

from app import main_enron

print('intializing model...')
model, vectorizer = main_enron.initialize_model()

class HTTPRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'text')
        self.end_headers()

        message = {'response': 'model and vectorizer loaded and trained'}
        self.wfile.write(bytes(json.dumps(message), 'utf8'))

    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        data = self.rfile.read(content_length).decode('utf-8')

        vectorized_data = vectorizer.get_vectorized_data([data])
        prediction = model.predict(vectorized_data)
        self.send_response(200)
        self.send_header('Content-type', 'text')
        self.end_headers()

        message = {'prediction': str(prediction[0])}
        self.wfile.write(bytes(json.dumps(message), 'utf8'))


def run():
    PORT = 8080
    print('starting server on port {}...'.format(PORT))

    server_address = ('127.0.0.1', PORT)
    httpd = HTTPServer(server_address, HTTPRequestHandler)

    print('running server...')
    httpd.serve_forever()


run()
