import asyncio
import asyncpg

async def check_database():
    # Connect to database
    conn = await asyncpg.connect(
        'postgresql://neondb_owner:npg_t0Y9KrBOxIMC@ep-jolly-fire-a1brhwgp-pooler.ap-southeast-1.aws.neon.tech/neondb'
    )

    # List all tables
    tables = await conn.fetch(
        "SELECT tablename FROM pg_catalog.pg_tables WHERE schemaname NOT IN ('pg_catalog', 'information_schema') ORDER BY tablename"
    )

    print("\n=== DATABASE TABLES ===")
    for row in tables:
        print(f"  - {row['tablename']}")

    # If there are session-related tables, query them
    table_names = [row['tablename'] for row in tables]

    for table_name in table_names:
        if 'session' in table_name.lower() or 'conversation' in table_name.lower() or 'message' in table_name.lower():
            print(f"\n=== TABLE: {table_name} ===")

            # Get schema
            schema = await conn.fetch(f"""
                SELECT column_name, data_type
                FROM information_schema.columns
                WHERE table_name = '{table_name}'
                ORDER BY ordinal_position
            """)

            print("Columns:")
            for col in schema:
                print(f"  - {col['column_name']}: {col['data_type']}")

            # Get sample data (limit 5)
            count_result = await conn.fetchval(f'SELECT COUNT(*) FROM "{table_name}"')
            print(f"\nTotal rows: {count_result}")

            if count_result > 0:
                rows = await conn.fetch(f'SELECT * FROM "{table_name}" LIMIT 5')
                print(f"\nSample data (first 5 rows):")
                for i, row in enumerate(rows, 1):
                    print(f"\nRow {i}:")
                    for key, value in dict(row).items():
                        # Truncate long values
                        if isinstance(value, str) and len(value) > 200:
                            value = value[:200] + "..."
                        print(f"  {key}: {value}")

    await conn.close()
    print("\n=== DONE ===")

# Run the async function
asyncio.run(check_database())
