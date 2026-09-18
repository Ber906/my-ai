
from .neural_network import NeuralNetwork
from .tokenizer import SimpleTokenizer
from .memory import Memory
import numpy as np
import random

class AutonomousAI:
    """The complete AI with personality"""
    
    def __init__(self):
        self.memory = Memory()
        self.tokenizer = SimpleTokenizer(vocab_size=1000)
        self.brain = NeuralNetwork(layers=[128, 64, 32, 16, 8])
        
        # Personality traits (affects response generation)
        self.personality = {
            'creativity': 0.7,
            'formality': 0.3,
            'enthusiasm': 0.8,
            'directness': 0.9
        }
        
        self.response_patterns = []
        self.is_trained = False
        
        # Load previous knowledge
        self.brain.load()
    
    def process_input(self, text):
        """Convert user input to neural input"""
        return self.tokenizer.text_to_vector(text)
    
    def generate_response(self, user_input, context=""):
        """Generate unique response - HINDI SCRIPTED"""
        
        # Get relevant memories
        memories = self.memory.recall(topic=user_input, limit=5)
        past_conversations = self.memory.get_conversation_history(limit=5)
        
        # Create context vector
        context_text = user_input + " " + " ".join([m[1] for m in memories])
        input_vector = self.process_input(context_text)
        
        # Neural network thinking
        if self.is_trained:
            thought_vector = self.brain.predict(input_vector.reshape(1, -1))
            thought_pattern = thought_vector[0]
        else:
            # Random but personality-influenced
            thought_pattern = np.random.rand(8)
        
        # Generate response based on thought pattern + personality
        response = self._formulate_response(
            user_input, 
            thought_pattern, 
            memories, 
            past_conversations
        )
        
        # Store this interaction
        self.memory.store_conversation(user_input, response, context)
        
        return response
    
    def _formulate_response(self, user_input, thought_pattern, memories, past_conversations):
        """Formulate unique response - sariling construction"""
        
        # Analyze thought pattern
        creativity = thought_pattern[0] * self.personality['creativity']
        enthusiasm = thought_pattern[1] * self.personality['enthusiasm']
        directness = thought_pattern[2] * self.personality['directness']
        
        # Build response components
        components = []
        
        # Opening based on enthusiasm
        if enthusiasm > 0.6:
            openings = ["Sige!", "G!", "Sure,", "Okey,"]
        else:
            openings = ["Hmm,", "Okay,", "So,"]
        components.append(random.choice(openings))
        
        # Main content based on memories
        if memories:
            # Use learned knowledge
            relevant = memories[0][1][:200]
            if directness > 0.5:
                components.append(f"based on what I learned: {relevant}")
            else:
                components.append(f"I remember reading that {relevant}")
        else:
            # Generate based on pattern
            words = user_input.split()
            if len(words) > 2:
                # Create variation
                shuffled = words[1:] + [words[0]]
                components.append(" ".join(shuffled))
        
        # Add unique twist based on creativity
        if creativity > 0.7:
            twists = [
                "Pero wait, may angle pa dyan.",
                "Tapos iniisip ko pa...",
                "Actually, interesting yang point mo."
            ]
            components.append(random.choice(twists))
        
        # Combine uniquely
        response = " ".join(components)
        
        # Post-process for natural flow
        response = self._naturalize(response)
        
        return response
    
    def _naturalize(self, text):
        """Make response sound natural"""
        # Simple transformations
        text = text.capitalize()
        if not text.endswith(('.', '!', '?')):
            text += '.'
        
        # Remove extra spaces
        text = ' '.join(text.split())
        
        return text
    
    def learn_from_text(self, text, source="user"):
        """Self-learning from any text"""
        # Tokenize and store
        words = text.split()
        
        # Build vocab if not done
        if not self.tokenizer.word_to_idx:
            self.tokenizer.build_vocab([text])
        
        # Store in memory
        self.memory.store_knowledge(
            topic="learned",
            content=text[:1000],
            source=source,
            importance=len(words) / 100
        )
        
        # Train brain on this text if enough data
        if len(self.memory.recall(limit=100)) > 10:
            self._self_train()
    
    def _self_train(self):
        """Train on accumulated knowledge"""
        memories = self.memory.recall(limit=50)
        
        if len(memories) < 5:
            return
        
        # Prepare training data
        texts = [m[1] for m in memories]
        
        # Build vocab if needed
        if len(self.tokenizer.word_to_idx) < 10:
            self.tokenizer.build_vocab(texts)
        
        # Create training vectors
        X = []
        y = []
        
        for i, text in enumerate(texts[:-1]):
            input_vec = self.tokenizer.text_to_vector(text)
            target_vec = self.tokenizer.text_to_vector(texts[i+1])
            
            # Create simple target (response prediction)
            target = np.array([
                np.mean(target_vec[:8]),  # sentiment-ish
                np.std(target_vec),       # complexity
                len(text) / 1000,         # length factor
                random.random(),          # creativity marker
                0.5, 0.5, 0.5, 0.5        # placeholders
            ])
            
            X.append(input_vec)
            y.append(target)
        
        X = np.array(X)
        y = np.array(y)
        
        # Train
        print("🧠 Self-training in progress...")
        self.brain.train(X, y, epochs=50)
        self.brain.save()
        self.is_trained = True
        print("✅ Training complete!")
    
    def get_stats(self):
        """AI's self-awareness stats"""
        memories = self.memory.recall(limit=1000)
        conversations = self.memory.get_conversation_history(limit=1000)
        
        return {
            'knowledge_items': len(memories),
            'conversations_had': len(conversations),
            'is_trained': self.is_trained,
            'vocab_size': len(self.tokenizer.word_to_idx),
            'personality': self.personality
        }
