
import sqlite3
import json
import numpy as np
from datetime import datetime

class Memory:
    """AI's long-term memory"""
    
    def __init__(self, db_path="ai_memory.db"):
        self.db_path = db_path
        self.init_db()
    
    def init_db(self):
        """Create memory tables"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        # Knowledge table
        c.execute('''
            CREATE TABLE IF NOT EXISTS knowledge (
                id INTEGER PRIMARY KEY,
                topic TEXT,
                content TEXT,
                source TEXT,
                timestamp TEXT,
                importance REAL,
                embedding BLOB
            )
        ''')
        
        # Conversations table
        c.execute('''
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY,
                user_input TEXT,
                ai_response TEXT,
                context TEXT,
                timestamp TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def store_knowledge(self, topic, content, source="unknown", importance=1.0, embedding=None):
        """Learn something new"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute('''
            INSERT INTO knowledge (topic, content, source, timestamp, importance, embedding)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (topic, content, source, datetime.now().isoformat(), importance, 
              json.dumps(embedding.tolist()) if embedding is not None else None))
        
        conn.commit()
        conn.close()
    
    def recall(self, topic=None, limit=10):
        """Retrieve memories"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        if topic:
            c.execute('''
                SELECT topic, content, source, importance 
                FROM knowledge 
                WHERE topic LIKE ? 
                ORDER BY importance DESC, timestamp DESC 
                LIMIT ?
            ''', (f'%{topic}%', limit))
        else:
            c.execute('''
                SELECT topic, content, source, importance 
                FROM knowledge 
                ORDER BY timestamp DESC 
                LIMIT ?
            ''', (limit,))
        
        results = c.fetchall()
        conn.close()
        return results
    
    def store_conversation(self, user_input, ai_response, context=""):
        """Remember conversations"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute('''
            INSERT INTO conversations (user_input, ai_response, context, timestamp)
            VALUES (?, ?, ?, ?)
        ''', (user_input, ai_response, context, datetime.now().isoformat()))
        
        conn.commit()
        conn.close()
    
    def get_conversation_history(self, limit=20):
        """Get past conversations"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute('''
            SELECT user_input, ai_response, context 
            FROM conversations 
            ORDER BY timestamp DESC 
            LIMIT ?
        ''', (limit,))
        
        results = c.fetchall()
        conn.close()
        return results
