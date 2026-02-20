import pandas as pd
from sqlalchemy import create_engine


# все пользователи совершали действия за последние 2 месяца. Мощностей хватило на обучение только на последних 2 неделях(87% пользователей)

# Добавленые фичи:
#     User CTR: Отношение лайков к просмотрам (насколько пользователь "лайкающий").
#     Количество просмотров: Общая активность пользователя.
#     Предпочтения по топикам: Какой CTR у пользователя в каждом из топиков (topic). Это самый важный признак для контентной модели.
 

#     Post CTR: Средний CTR поста (насколько он "виральный").
#     Популярность топика: Насколько часто смотрят посты данной тематики.

QUERY_FEED = """
    SELECT timestamp, user_id, post_id, target, action
    FROM feed_data
    WHERE timestamp > '2021-12-14'
    ORDER BY timestamp
"""

QUERY_USERS = "SELECT * FROM user_data"
QUERY_POSTS = "SELECT post_id, topic FROM post_text_df" 


ENGINE = create_engine(
        "postgresql://robot-startml-ro:pheiph0hahj1Vaif@"
        "postgres.lab.karpov.courses:6432/startml"
    )


def batch_load_sql(query: str) -> pd.DataFrame:
    CHUNKSIZE = 100000
    engine = ENGINE
    conn = engine.connect().execution_options(stream_results=True)
    chunks = []
    for chunk_dataframe in pd.read_sql(query, conn, chunksize=CHUNKSIZE):
        chunks.append(chunk_dataframe)
    conn.close()
    return pd.concat(chunks, ignore_index=True)


def add_new_features() -> pd.DataFrame:
    engine = ENGINE
    feed = batch_load_sql(QUERY_FEED)
    users = pd.read_sql(QUERY_USERS, engine)
    posts = pd.read_sql(QUERY_POSTS, engine)
    df = feed.merge(users, on='user_id', how='left')
    df = df.merge(posts, on='post_id', how='left')

    df['hour'] = pd.to_datetime(df['timestamp']).dt.hour # new feature
    df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek # new feature
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int) # new feature

    df["user_views"] = df.groupby("user_id")["post_id"].transform("count") # new feature
    df["is_like"] = (df["action"] == "like").astype(int)
    df["likes"] = df.groupby("user_id")["is_like"].transform("sum")
    df["user_ctr"] = df["likes"] / df["user_views"].replace(0, 1) # new feature

    df["post_views"] = df.groupby("post_id")["user_id"].transform("count")
    df["post_likes"] = df.groupby("post_id")["is_like"].transform("sum")
    df["post_ctr"] = df["post_likes"] / df["post_views"].replace(0, 1) # new feature

    df["topic_popularity"] = df.groupby("topic")["action"].transform("count") # new feature

    user_topic_likes = df.groupby(["user_id", "topic"])["is_like"].transform("sum")
    user_topic_posts = df.groupby(["user_id", "topic"])["post_id"].transform("count")
    df["user_topic_ctr"] = user_topic_likes / user_topic_posts.replace(0, 1)
 
    df.drop(["timestamp", "likes", "day_of_week", "is_like", "post_likes", "post_views"], axis=1, inplace=True)
    return df



def load_features() -> pd.DataFrame:
    return batch_load_sql("SELECT * FROM pavel_golovin_data_for_training")



if __name__ == "__main__":
    df = add_new_features()
    df.to_sql('pavel_golovin_data_for_training', con=ENGINE, if_exists="replace")
    df = load_features()
    print(df.head())