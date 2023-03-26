from flask import Flask, render_template, request, flash, session
import pickle

app = Flask(__name__)


model = pickle.load(open("model.pkl", "rb"))


@app.route("/")
def home():
    return render_template("main.html")


@app.route("/ContactUs")
def contactUs():
    return render_template("ContactUs.html")


@app.route("/AboutUs")
def aboutUs():
    return render_template("AboutUs.html")


@app.route("/predict", methods=["POST"])
def predict():

    lst = []

    gender = int(request.form["gender"])
    age = int(request.form["age"])
    smoking = int(request.form["smoking"])
    anxiety = int(request.form["Anxiety"])
    peerPressure = int(request.form["peerPressure"])
    chronicDisease = int(request.form["chronicDisease"])
    coughing = int(request.form["Coughing"])
    shortnessOfBreath = int(request.form["shortnessOfBreath"])
    chestPain = int(request.form["chestPain"])

    prediction = model.predict(
        [
            [
                gender,
                age,
                smoking,
                anxiety,
                peerPressure,
                chronicDisease,
                coughing,
                shortnessOfBreath,
                chestPain,
            ]
        ]
    )

    if prediction == [1]:
        result = "کم"
    else:
        result = "زیاد"

    msg = "احتمال ابتلای بیماری براساس پاسخ سوال ها {0} است".format(result)

    flash(msg)
    


    return render_template("main.html")


if __name__ == "__main__":
    app.secret_key = 'super secret key'
    app.run(debug=True)
