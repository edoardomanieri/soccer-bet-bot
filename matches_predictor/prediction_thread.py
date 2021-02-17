import numpy as np
import pandas as pd
from matches_predictor.models.ef import train_set, input_stream, prediction
import os


def run(in_q, prob_threshold):
    file_path = os.path.dirname(os.path.abspath(__file__))

    cat_cols = ['home', 'away', 'campionato', 'date', 'id_partita']
    to_drop_cols = ['home', 'away', 'date', 'id_partita']
    outcome_cols = ['home_final_score', 'away_final_score', 'final_ef']

    train_df = pd.read_csv(f"{file_path}/../res/dataframes/training_ef.csv", header=0, index_col=0)
    cols_used = [col for col in train_df.columns if col not in to_drop_cols + outcome_cols]

    clf = train_set.Modeling.get_prod_model()

    while True:
        input_df = pd.DataFrame(in_q.get())
        input_df.drop(columns=['fixture_id'], inplace=True)
        input_prematch_odds = input_stream.Preprocessing.execute(input_df, train_df, cat_cols)
        test_X = input_df[cols_used]
        prediction.get_predict_proba(clf, test_X, input_df)
        predictions_df = prediction.prematch_odds_based(input_df, input_prematch_odds)
        prediction_obj = prediction.Prediction(predictions_df)
        if prediction_obj.probability > prob_threshold:
            # todo: betbot insert
            pass
        print(f"{prediction_obj.home}-{prediction_obj.away}, \
              minute: {prediction_obj.minute}, \
              probability: {prediction_obj.probability}, \
              model_probability: {prediction_obj.model_probability}, \
              eventual prediction: {prediction_obj.prediction}\n")
