from matches_predictor import pipeline
import numpy as np

def main():
    final_df = pipeline.run(True, True, True, True)
    final_df = final_df[final_df['minute'] < 85]
    final_df['probability_final_result'] = np.max(final_df[['probability_final_result_1', 'probability_final_result_X', 'probability_final_result_2']].values, axis=1)
    print(final_df.loc[:, ['home', 'away', 'minute', 'home_score', 'away_score','probability_final_result', 'prediction_final_result','probability_final_over', 'prediction_final_over']])


if __name__ == "__main__":
    main()