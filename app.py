from flask import Flask, render_template
from flask import request
from werkzeug import datastructures

from face_compare import faces_compare

app = Flask(__name__)


@app.route('/', methods=["GET", "POST"])
def method_one():
    if request.method == "GET":
        return render_template("pasport-selfi.html", result={})
    elif request.method == 'POST':
        filestorage_image: datastructures.FileStorage = request.files["image"]
        if filestorage_image:
            result = faces_compare(filestorage_image=filestorage_image)

            return render_template("pasport-selfi.html", result=result)
        else:
            return render_template("pasport-selfi.html", result={'error': 'Ошибка загрузки изображения!'})

    #return render_template("pasport-selfi.html")

@app.route('/get_id', methods=["POST"])
def get_id():
    """
    API method for person ifentification
    :return:        json-file with required params
    """
    if request.method == 'POST':
        filestorage_image: datastructures.FileStorage = request.files["image"]
        person_id: str = request.args.get('username')

        result = faces_compare(filestorage_image=filestorage_image)

        return result

@app.route('/about')
def about():
    return "About page"


if __name__ == "__main__":
    app.run(debug=False)
