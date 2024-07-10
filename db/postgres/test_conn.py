import psycopg2

try:
    conn = psycopg2.connect(
        host="localhost",
        database="amazon_reviews",
        user="admin",
        password="adminpassword",
    )
    print("Connected to PostgreSQL successfully!")

    cur = conn.cursor()
    cur.execute("SELECT version();")
    db_version = cur.fetchone()
    print(f"PostgreSQL database version: {db_version}")

    cur.close()
except (Exception, psycopg2.Error) as error:
    print("Error while connecting to PostgreSQL", error)
finally:
    if conn:
        conn.close()
        print("PostgreSQL connection is closed")
