import pandas as pd
from typing import List
from fastapi import FastAPI
from sqlalchemy import text
from feature_upload import ENGINE
from schema import PostGet




app = FastAPI()

@app.get("/post/recommendations/", response_model=List[PostGet])
def recommended_posts(id: int) -> List[PostGet]:
    query = text("""
        SELECT f.post_id, p.text, p.topic
        FROM pavel_golovin_data_for_recommendations f
        JOIN post_text_df p ON f.post_id = p.post_id
        WHERE f.user_id = :uid
        ORDER BY f.prob DESC, f.user_topic_ctr DESC
        LIMIT 5
    """)
    df = pd.read_sql(query, ENGINE, params={"uid": id})
    resp = []
    for row in df.itertuples(index=False):
        resp.append(PostGet(id=int(row.post_id), text=row.text, topic=row.topic))
    return resp