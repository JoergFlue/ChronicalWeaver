"""Conversation summarization utilities"""
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class ConversationSummarizer:
    """Creates summaries of conversation history"""
    
    def __init__(self):
        pass
    
    def create_summary(self, messages: List[str], target_length: int = 200) -> str:
        """Create a summary of conversation messages"""
        try:
            # Simple extractive summarization for now
            # In a full implementation, this would use an LLM or specialized summarization model
            
            if len(messages) <= 3:
                return " | ".join(messages)
            
            # Extract key information
            key_points = []
            character_mentions = set()
            locations = set()
            actions = []
            
            for message in messages:
                # Extract character names (simple heuristic)
                words = message.split()
                for i, word in enumerate(words):
                    if word.istitle() and len(word) > 2:
                        character_mentions.add(word)
                
                # Extract actions (messages with verbs)
                if any(verb in message.lower() for verb in ['went', 'said', 'did', 'came', 'left', 'found']):
                    actions.append(message[:100])  # Truncate long actions
            
            # Build summary
            summary_parts = []
            
            if character_mentions:
                char_list = list(character_mentions)[:3]  # Limit to 3 characters
                summary_parts.append(f"Characters involved: {', '.join(char_list)}")
            
            if actions:
                key_actions = actions[:3]  # Limit to 3 key actions
                summary_parts.append(f"Key events: {' | '.join(key_actions)}")
            
            summary = ". ".join(summary_parts)
            
            # Truncate if too long
            if len(summary) > target_length:
                summary = summary[:target_length-3] + "..."
            
            return summary or "Conversation in progress"
            
        except Exception as e:
            logger.error(f"Error creating summary: {str(e)}")
            return "Summary unavailable"
    
    def extract_key_events(self, messages: List[str]) -> List[str]:
        """Extract key events from conversation"""
        events = []
        
        try:
            for message in messages:
                # Look for action indicators
                if any(indicator in message.lower() for indicator in 
                      ['suddenly', 'then', 'next', 'after', 'before', 'when']):
                    events.append(message[:150])  # Truncate long events
            
            return events[:5]  # Return top 5 events
            
        except Exception as e:
            logger.error(f"Error extracting key events: {str(e)}")
            return []
    
    def extract_character_developments(self, messages: List[str]) -> Dict[str, List[str]]:
        """Extract character developments from conversation"""
        developments = {}
        
        try:
            # Simple implementation - in practice would use NLP
            for message in messages:
                # Look for character development keywords
                if any(keyword in message.lower() for keyword in 
                      ['feels', 'thinks', 'believes', 'remembers', 'learns']):
                    # Extract character name (simplified)
                    words = message.split()
                    for word in words:
                        if word.istitle() and len(word) > 2:
                            if word not in developments:
                                developments[word] = []
                            developments[word].append(message[:100])
                            break
            
            return developments
            
        except Exception as e:
            logger.error(f"Error extracting character developments: {str(e)}")
            return {}
