from flask import Flask, request, jsonify
import joblib
import traceback
import pandas as pd
import numpy as np

app = Flask(__name__)

@app.route('/predict',methods=['POST'])
def predict():
    if lr:
        try:
            json_ = request.json
            print(json_)
            query = pd.DataFrame(json_)
            query = query.reindex(columns=model_columns)

            prediction = list(lr.predict(query))

            return jsonify({'prediction': str(prediction)})
        
        except:

            return jsonify({'trace': traceback.format_exc()})
        
    else:
        print('Train the model first')
        return ('No model here to use')
    
if __name__ == '__main__':
    try:
        port = int(sys.argv[1])
    except:
        port = 12345

    lr = joblib.load("model.pkl")
    print('Model loaded')
    model_columns = joblib.load("model_columns.pkl")
    print('Model columns loaded')

    app.run(port=port, debug=True)


# iris.rename(columns= {"sepal length (cm)": "sepal_length",
#                       "sepal width (cm)": "sepal_width",
#                       "petal length (cm)": "petal_length",
#                       "petal width (cm)": "petal_width"},
#                       inplace=True)
# class flower(Resource):
#     def get(self, i, j, k, l):
#         for i in iris["sepal_length"] and j in iris["sepal_width"] and k in iris["petal_length"] and l in iris["petal_width"]:
#             return {"target": iris['target']}
#         return 
    
# api.add_resource(flower, "/flower/<float:sepal_length>/<float:sepal_width>/<float:petal_length>/<float:petal_width>")

# if __name__ == "__main__":
#     app.run(debug=True)