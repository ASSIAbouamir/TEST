import sqlite3

def inspect_db():
    conn = sqlite3.connect("chroma_db/chroma.sqlite3")
    cursor = conn.cursor()
    
    # Get all tables
    cursor.execute("SELECT name, sql FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()
    
    for table_name, schema in tables:
        print(f"Table: {table_name}")
        print("Schema:")
        print(schema)
        print("-" * 60)
        
        # Get some sample rows or row count
        cursor.execute(f"SELECT COUNT(*) FROM [{table_name}]")
        count = cursor.fetchone()[0]
        print(f"Row count: {count}\n")

if __name__ == "__main__":
    inspect_db()
