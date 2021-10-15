from flask import Flask
from flask import jsonify,request,render_template
import numpy as np
import pickle
from os import getcwd
app = Flask(__name__)


modelfile = open("./models/rf_final.pkl",'rb')
model_load = pickle.load(modelfile)
modelfile.close()

ratingModel = open("./models/user_final_rating.pkl",'rb')
model_rating = pickle.load(ratingModel)
ratingModel.close()

recList = model_rating.loc["gordy313"].sort_values(ascending=False)[0:5]
print ("TEST", type(recList))
print(recList.index.tolist())


@app.route('/')
def home():
   #return "Hello World"

   return render_template("index.html")

@app.route("/predict", methods=['POST'])
def predict():
    if (request.method == 'POST'):
        uid = [x for x in request.form.values()]
        print("REQUEST")
        print(uid[0])
        try:
            recList = model_rating.loc[uid[0]].sort_values(ascending=False)[0:5].index.tolist()
            print (recList)
        except:
            recList = "Invalid User"
        output = recList
        return render_template('index.html', prediction_text='Recommended Output : {}'.format(output))
    else :
        return render_template('index.html')

@app.route("/predict_api", methods=['POST', 'GET'])
def predict_api():
    print(" request.method :",request.method)
    if (request.method == 'POST'):
        data = request.get_json()
        return jsonify(model_load.predict([np.array(list(data.values()))]).tolist())
    else:
        return render_template('index.html')

if __name__ == '__main__':
   app.run()