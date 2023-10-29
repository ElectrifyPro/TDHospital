# Sample participant submission for testing
from flask import Flask, jsonify, request
import tensorflow as tf
import pandas as pd
import random

app = Flask(__name__)


class Solution:
    def __init__(self):
        #Initialize any global variables here
        self.model = tf.keras.models.load_model('example.h5')

    def calculate_death_prob(self, timeknown, cost, reflex, sex, blood, bloodchem1, bloodchem2, temperature, race,
                             heart, psych1, glucose, psych2, dose, psych3, bp, bloodchem3, confidence, bloodchem4,
                             comorbidity, totalcost, breathing, age, sleep, dnr, bloodchem5, pdeath, meals, pain,
                             primary, psych4, disability, administratorcost, urine, diabetes, income, extraprimary,
                             bloodchem6, education, psych5, psych6, information, cancer):
        
        """
        This function should return your final prediction!
        """
        labels = ['reflex', 'sex', 'blood', 'bloodchem1', 'bloodchem2', 'temperature', 'heart', 'psych1', 'glucose', 'psych2', 'dose', 'bloodchem3', 'confidence', 'bloodchem4', 'comorbidity', 'breathing', 'age']
        values = [float(x) for x in [reflex, sex, blood, bloodchem1, bloodchem2, temperature, heart, psych1, glucose, psych2, dose, bloodchem3, confidence, bloodchem4, comorbidity, breathing, age]]
        df = dict()
        for label, value in zip(labels, values):
            df[label] = [value]
        df = pd.DataFrame(df)
        df.replace('', 0, inplace=True)
        df.fillna(0, inplace=True)
        prediction = self.model.predict(df.to_numpy())
        return float(prediction[0][0])


# BOILERPLATE
@app.route("/death_probability", methods=["POST"])
def q1():
    solution = Solution()
    data = request.get_json()
    return {
        "probability": solution.calculate_death_prob(data['timeknown'], data['cost'], data['reflex'], data['sex'], data['blood'],
                                            data['bloodchem1'], data['bloodchem2'], data['temperature'], data['race'],
                                            data['heart'], data['psych1'], data['glucose'], data['psych2'],
                                            data['dose'], data['psych3'], data['bp'], data['bloodchem3'],
                                            data['confidence'], data['bloodchem4'], data['comorbidity'],
                                            data['totalcost'], data['breathing'], data['age'], data['sleep'],
                                            data['dnr'], data['bloodchem5'], data['pdeath'], data['meals'],
                                            data['pain'], data['primary'], data['psych4'], data['disability'],
                                            data['administratorcost'], data['urine'], data['diabetes'], data['income'],
                                            data['extraprimary'], data['bloodchem6'], data['education'], data['psych5'],
                                            data['psych6'], data['information'], data['cancer'])}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5555)
