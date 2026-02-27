import pandas as pd
import os
from catboost import CatBoostClassifier
from feature_upload import load_features, ENGINE

#в итоге, предсказания для 87% юзеров. К сожалению, не хватило железа для обучения на 2-месячных логах



def get_recommendation(user_id: str)-> dict:
    uid = int(user_id)
    query = f"""
        SELECT post_id
        FROM pavel_golovin_data_for_recommendations
        WHERE user_id = {uid}
        ORDER BY user_topic_ctr DESC, prob DESC
        LIMIT 5
    """
    df = pd.read_sql(query, ENGINE)
    return df["post_id"].tolist()


if __name__ == "__main__":
  base_dir = os.path.dirname(os.path.dirname(__file__))
  model_path = os.path.join(base_dir, "models", "catboost_model.cbm")
  model = CatBoostClassifier().load_model(model_path)

  df = load_features()
  x = df[df["action"] != "like"] # убрали старые лайки
  x.drop(["target"], axis=1)
  probs = model.predict_proba(x)[:, 1]  # вероятность класса 1
  x["prob"] = probs
  x["target"] = model.predict(x)
  x = x[x["target"] == 1]
  x = x[["user_id", "post_id", "user_topic_ctr", "prob"]]
  x.to_sql("pavel_golovin_data_for_recommendations", con=ENGINE, if_exists="replace", index=False)

  
