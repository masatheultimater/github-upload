from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def index():
    message = "Hello!"
    li = ["Hello", "World"]
    dic = {"name":"AI Academy", "lang":"Python"}

    return render_template('index.html', message=message, li=li, dic=dic)

if __name__ == '__main__':
    app.run(port=12345, debug=False)
