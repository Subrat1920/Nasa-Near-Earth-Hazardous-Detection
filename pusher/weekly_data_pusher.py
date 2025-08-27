import pandas as pd
from src.utils.utils import fetch_data
from sqlalchemy import create_engine
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os

load_dotenv()
DATABASE_URL = os.getenv('DATABASE_URL')
NASA_API_KEY = os.getenv('NASA_API_KEY')


def append_new_data():
    print("[INFO] Starting NEO data fetch...")

    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=6)

    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")
    print(f"[INFO] Fetching data from {start_str} to {end_str}")

    data = fetch_data(start_date=start_str, end_date=end_str)

    neo_dict = {
        "neo_reference_id": [],
        "name": [],
        "absolute_magnitude_h": [],
        "min_diameter_m": [],
        "max_diameter_m": [],
        "close_approach_date": [],
        "epoch_date_close_approach": [],
        "miss_distance_km": [],
        "relative_velocity_kph": [],
        "is_potentially_hazardous": [],
        "is_sentry_object": [],
    }

    if data and "near_earth_objects" in data:
        print("[INFO] Parsing asteroid data...")
        for date_key, asteroids in data["near_earth_objects"].items():
            print(f"[DEBUG] Processing {len(asteroids)} asteroids for {date_key}")
            for asteroid in asteroids:
                neo_dict["neo_reference_id"].append(asteroid.get("neo_reference_id"))
                neo_dict["name"].append(asteroid.get("name"))
                neo_dict["absolute_magnitude_h"].append(asteroid.get("absolute_magnitude_h"))
                neo_dict["min_diameter_m"].append(
                    asteroid.get("estimated_diameter", {}).get("meters", {}).get("estimated_diameter_min")
                )
                neo_dict["max_diameter_m"].append(
                    asteroid.get("estimated_diameter", {}).get("meters", {}).get("estimated_diameter_max")
                )
                neo_dict["close_approach_date"].append(
                    asteroid.get("close_approach_data", [{}])[0].get("close_approach_date")
                )
                neo_dict["epoch_date_close_approach"].append(
                    asteroid.get("close_approach_data", [{}])[0].get("epoch_date_close_approach")
                )
                neo_dict["miss_distance_km"].append(
                    asteroid.get("close_approach_data", [{}])[0].get("miss_distance", {}).get("kilometers")
                )
                neo_dict["relative_velocity_kph"].append(
                    asteroid.get("close_approach_data", [{}])[0].get("relative_velocity", {}).get("kilometers_per_hour")
                )
                neo_dict["is_potentially_hazardous"].append(
                    asteroid.get("is_potentially_hazardous_asteroid")
                )
                neo_dict["is_sentry_object"].append(
                    asteroid.get("is_sentry_object")
                )
    else:
        print("[WARN] No data received from API.")

    neo_df = pd.DataFrame(neo_dict)
    print(f"[INFO] Created DataFrame with {len(neo_df)} rows and {len(neo_df.columns)} columns.")

    # Type casting
    neo_df["close_approach_date"] = pd.to_datetime(neo_df["close_approach_date"], errors="coerce")
    neo_df["epoch_date_close_approach"] = pd.to_datetime(
        neo_df["epoch_date_close_approach"], unit="ms", errors="coerce"
    )
    numeric_cols = ["min_diameter_m", "max_diameter_m", "miss_distance_km", "relative_velocity_kph"]
    neo_df[numeric_cols] = neo_df[numeric_cols].apply(pd.to_numeric, errors="coerce")

    print("[INFO] DataFrame cleaned and ready.")
    return neo_df


def append_to_database(df, database_url, table_name="neo_table"):
    if df.empty:
        print("[WARN] No data to append to the database.")
        return

    print(f"[INFO] Connecting to database: {database_url}")
    engine = create_engine(database_url)

    try:
        df.to_sql(table_name, engine, if_exists="append", index=False)
        print(f"[SUCCESS] Inserted {len(df)} rows into '{table_name}'.")
    except Exception as e:
        print(f"[ERROR] Failed to insert into database: {e}")
    finally:
        engine.dispose()
        print("[INFO] Database connection closed.")


if __name__ == "__main__":
    print("[START] NEO Data Pipeline Execution")
    neo_df = append_new_data()
    append_to_database(neo_df, DATABASE_URL)
    print("[END] NEO Data Pipeline Execution")
