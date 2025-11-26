import os
import psycopg2
from psycopg2.extras import RealDictCursor
from pgvector.psycopg2 import register_vector

class Database:
    """Database connection handler for PostgreSQL with pgvector"""
    
    def __init__(self):
        self.conn = None
        self.connect()
    
    def connect(self):
        """Establish connection to PostgreSQL database"""
        try:
            self.conn = psycopg2.connect(
                host=os.getenv('DB_HOST', 'localhost'),
                port=os.getenv('DB_PORT', '5432'),
                database=os.getenv('DB_NAME', 'rag_db'),
                user=os.getenv('DB_USER', 'rag_user'),
                password=os.getenv('DB_PASSWORD', 'rag_password')
            )
            # Register pgvector type
            register_vector(self.conn)
            print("✓ Database connected successfully")
        except Exception as e:
            print(f"✗ Database connection failed: {e}")
            raise
    
    def get_cursor(self, dict_cursor=True):
        """Get a database cursor"""
        if dict_cursor:
            return self.conn.cursor(cursor_factory=RealDictCursor)
        return self.conn.cursor()
    
    def execute_query(self, query, params=None, fetch=False):
        """Execute a query and optionally fetch results"""
        with self.get_cursor() as cursor:
            cursor.execute(query, params)
            if fetch:
                return cursor.fetchall()
            self.conn.commit()
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            print("✓ Database connection closed")
    
    def test_connection(self):
        """Test database connection and pgvector extension"""
        try:
            with self.get_cursor() as cursor:
                cursor.execute("SELECT version();")
                db_version = cursor.fetchone()
                
                cursor.execute("SELECT * FROM pg_extension WHERE extname = 'vector';")
                vector_ext = cursor.fetchone()
                
                if vector_ext:
                    print(f"✓ PostgreSQL: {db_version['version']}")
                    print(f"✓ pgvector extension is installed")
                    return True
                else:
                    print("✗ pgvector extension not found")
                    return False
        except Exception as e:
            print(f"✗ Connection test failed: {e}")
            return False

# Singleton instance
db = Database()

