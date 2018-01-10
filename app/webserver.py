import web

from app import main_enron
from app.constants import CATEGORY_4_MAP

class WebServer:
    def GET(self):
        return 'model and vectorizer loaded and trained'

    def POST(self):
        model, vectorizer = main_enron.initialize_model()
        data = web.data().decode("utf-8")

        vectorized_input = vectorizer.get_vectorized_data([data])
        prediction = model.predict(vectorized_input)

        return {'prediction': CATEGORY_4_MAP[prediction[0]]}


if __name__ == "__main__":
    urls = ('/', 'WebServer')
    app = web.application(urls, globals())
    app.run()
