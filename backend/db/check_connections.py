"""
Check and kill idle database connections
Run: docker-compose exec backend python db/check_connections.py
"""
import os
import psycopg2

def check_and_clean_connections():
    """Check for idle connections and optionally clean them"""
    conn = psycopg2.connect(
        host=os.getenv('DB_HOST', 'postgres'),
        port=os.getenv('DB_PORT', '5432'),
        database=os.getenv('DB_NAME', 'rag_db'),
        user=os.getenv('DB_USER', 'rag_user'),
        password=os.getenv('DB_PASSWORD', 'rag_password')
    )
    conn.autocommit = True
    cursor = conn.cursor()
    
    print("\n" + "="*60)
    print("DATABASE CONNECTION STATUS")
    print("="*60)
    
    # Check all connections
    cursor.execute("""
        SELECT 
            pid,
            usename,
            application_name,
            client_addr,
            state,
            state_change,
            now() - state_change as idle_duration,
            query
        FROM pg_stat_activity 
        WHERE datname = 'rag_db'
        ORDER BY state_change;
    """)
    
    connections = cursor.fetchall()
    
    print(f"\nTotal connections: {len(connections)}")
    print("\nConnection Details:")
    print("-"*60)
    
    idle_pids = []
    for conn_info in connections:
        pid, user, app, addr, state, change, duration, query = conn_info
        print(f"\nPID: {pid}")
        print(f"  User: {user}")
        print(f"  App: {app}")
        print(f"  State: {state}")
        print(f"  Duration: {duration}")
        print(f"  Query: {query[:80]}..." if query and len(query) > 80 else f"  Query: {query}")
        
        if state == 'idle in transaction':
            idle_pids.append(pid)
            print(f"  ⚠️  IDLE IN TRANSACTION - BLOCKING!")
    
    # Kill idle transactions
    if idle_pids:
        print("\n" + "="*60)
        print(f"Found {len(idle_pids)} idle transactions")
        response = input("Kill these connections? (y/n): ")
        
        if response.lower() == 'y':
            for pid in idle_pids:
                try:
                    cursor.execute(f"SELECT pg_terminate_backend({pid})")
                    print(f"  ✓ Killed PID {pid}")
                except Exception as e:
                    print(f"  ✗ Failed to kill PID {pid}: {e}")
            print("\n✓ Idle connections cleaned")
        else:
            print("\nNo connections killed")
    else:
        print("\n✓ No problematic connections found")
    
    print("="*60 + "\n")
    cursor.close()
    conn.close()

if __name__ == "__main__":
    check_and_clean_connections()