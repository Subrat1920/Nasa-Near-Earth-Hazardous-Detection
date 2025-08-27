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

query = "SELECT * FROM neo_table;"

# ✅ Pass engine directly (works with SQLAlchemy 2.x)
df = pd.read_sql(query, engine)
df.to_csv("neo_table_data.csv", index=False)
