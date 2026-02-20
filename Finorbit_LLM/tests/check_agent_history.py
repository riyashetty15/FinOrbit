import asyncio
import asyncpg
import json

async def check_agent_history():
    conn = await asyncpg.connect(
        'postgresql://neondb_owner:npg_t0Y9KrBOxIMC@ep-jolly-fire-a1brhwgp-pooler.ap-southeast-1.aws.neon.tech/neondb'
    )

    # Check agent_messages table schema
    print("\n=== agent_messages TABLE SCHEMA ===")
    schema = await conn.fetch("""
        SELECT column_name, data_type
        FROM information_schema.columns
        WHERE table_name = 'agent_messages'
        ORDER BY ordinal_position
    """)
    for col in schema:
        print(f"  - {col['column_name']}: {col['data_type']}")

    # Count messages
    count = await conn.fetchval('SELECT COUNT(*) FROM agent_messages')
    print(f"\nTotal messages: {count}")

    # Get most recent 10 agent_messages
    if count > 0:
        print("\n=== MOST RECENT 10 AGENT MESSAGES ===")
        messages = await conn.fetch("""
            SELECT * FROM agent_messages
            ORDER BY created_at DESC
            LIMIT 10
        """)

        for i, msg in enumerate(messages, 1):
            print(f"\n--- Message {i} ---")
            for key, value in dict(msg).items():
                if isinstance(value, str) and len(value) > 500:
                    print(f"{key}: {value[:500]}...")
                else:
                    print(f"{key}: {value}")

    # Check if we can find any session that has multiple messages (conversation)
    print("\n\n=== SESSIONS WITH MULTIPLE MESSAGES ===")
    multi_msg_sessions = await conn.fetch("""
        SELECT session_id, COUNT(*) as msg_count
        FROM agent_messages
        GROUP BY session_id
        HAVING COUNT(*) > 1
        ORDER BY COUNT(*) DESC
        LIMIT 5
    """)

    for session in multi_msg_sessions:
        print(f"\nSession: {session['session_id']} ({session['msg_count']} messages)")

        # Get messages for this session
        session_messages = await conn.fetch("""
            SELECT id, message_data, created_at
            FROM agent_messages
            WHERE session_id = $1
            ORDER BY created_at ASC
        """, session['session_id'])

        for msg in session_messages:
            data_preview = msg['message_data'][:150] + "..." if len(msg['message_data']) > 150 else msg['message_data']
            print(f"  [{msg['created_at']}] {data_preview}")

    await conn.close()

asyncio.run(check_agent_history())
