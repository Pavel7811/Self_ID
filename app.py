from flask import Flask, render_template, url_for, Response
from flask import request
from face_compare import faces_compare
app = Flask(__name__)


# обработка поступающих веб-запросов
# работа с главной страницей
@app.route('/', methods=["GET", "POST"])
def home():
    # отображение главной страницы
    if request.method == "GET":
        return render_template("home.html", result={})
    elif request.method == 'POST':
        # получение изображения от пользователя и передача его в функцию сравнения лиц
        try:
            image = request.files["image"]
        except Exception as err:
            return render_template("home.html", result={'error': 'Ошибка загрузки изображения! Пожалуйста '
                                                                 'убедитесь, что изображение загружено.'})
        if image:
            result = faces_compare(filestorage_image=image)
            return render_template("result.html", result=result)


if __name__ == "__main__":
    app.run(debug=True)
