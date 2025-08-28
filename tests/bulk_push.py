from sqlalchemy import create_engine
import pandas as pd

engine = create_engine(
    "postgresql+psycopg2://neondb_owner:npg_DbOuR93hmzjH@ep-mute-bread-a1hy59mo-pooler.ap-southeast-1.aws.neon.tech:5432/neondb",
    connect_args={
        "sslmode": "require",
        "keepalives": 1,
        "keepalives_idle": 30,
        "keepalives_interval": 10,
        "keepalives_count": 5
    }
)


df = pd.read_csv(r'D:\Data Science\Data Science Projects\NASA_Near_Earth_Object_Detection\Data\data.csv')
df.drop(columns=['Unnamed: 0'], inplace=True)
df = df.head(2)
df.to_sql('test_neo', con=engine, if_exists='replace', index=False)
print('Pushed to Database')
