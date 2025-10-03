import json
import os
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import tiktoken
import anthropic
import openai
from pathlib import Path

# ============================================
# 1. DATA MODELS
# ============================================

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

# ============================================
# CORRECTED DATA MODELS
# ============================================

@dataclass
class Reply:
    """A reply within a thread (cannot have its own thread)"""
    message_id: str
    text: str
    user: str
    timestamp: str
    
    def to_dict(self):
        return {
            'message_id': self.message_id,
            'text': self.text,
            'user': self.user,
            'timestamp': self.timestamp
        }

@dataclass
class TopLevelMessage:
    """A top-level message that may or may not have a thread"""
    message_id: str
    text: str
    user: str
    timestamp: str
    thread: List[Reply] = field(default_factory=list)  # Empty list if no thread
    
    @property
    def has_thread(self) -> bool:
        return len(self.thread) > 0
    
    def to_dict(self):
        return {
            'message_id': self.message_id,
            'text': self.text,
            'user': self.user,
            'timestamp': self.timestamp,
            'thread': [reply.to_dict() for reply in self.thread]
        }

@dataclass
class Channel:
    channel_id: str
    channel_name: str
    messages: List[TopLevelMessage]  # Only top-level messages
    
    def to_dict(self):
        return {
            'channel_id': self.channel_id,
            'channel_name': self.channel_name,
            'messages': [msg.to_dict() for msg in self.messages]
        }
    
class ChunkType(Enum):
    INDIVIDUAL_MESSAGE = "individual_message"
    THREAD = "thread"
    SLIDING_WINDOW = "sliding_window"

@dataclass
class Chunk:
    chunk_id: str
    chunk_type: ChunkType
    channel_id: str
    channel_name: str
    content: str  # The actual text content
    metadata: Dict[str, Any]  # message_ids, timestamps, etc.
    token_count: int
    
    def to_dict(self):
        return {
            'chunk_id': self.chunk_id,
            'chunk_type': self.chunk_type.value,
            'channel_id': self.channel_id,
            'channel_name': self.channel_name,
            'content': self.content,
            'metadata': self.metadata,
            'token_count': self.token_count
        }

@dataclass
class SyntheticQuery:
    query_id: str
    query_text: str
    chunk_id: str
    persona: str  # "technical_expert", "beginner", "typo_prone", etc.
    difficulty: str  # "easy", "medium", "hard"
    metadata: Dict[str, Any]

# ============================================
# 2. POST-PROCESSOR: Raw Slack → Cleaned Format
# ============================================


# ============================================
# 3. CHUNKER: Create 3 Chunking Strategies
# ============================================

class SlackChunker:
    """Create chunks from cleaned Slack data"""
    
    def __init__(self, model: str = "gpt-4"):
        self.encoder = tiktoken.encoding_for_model(model)
        self.max_tokens = 4096
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.encoder.encode(text))
    
    def chunk_channel(self, channel: Channel) -> Dict[ChunkType, List[Chunk]]:
        """Apply all 3 chunking strategies"""
        return {
            ChunkType.INDIVIDUAL_MESSAGE: self._chunk_individual_messages(channel),
            ChunkType.THREAD: self._chunk_threads(channel),
            ChunkType.SLIDING_WINDOW: self._chunk_sliding_window(channel)
        }
    
    def _chunk_individual_messages(self, channel: Channel) -> List[Chunk]:
        """Strategy 1: Each message (top-level or reply) is its own chunk"""
        chunks = []
        
        for msg in channel.messages:
            # Top-level message
            chunks.append(Chunk(
                chunk_id=f"{channel.channel_id}_{msg.message_id}_msg",
                chunk_type=ChunkType.INDIVIDUAL_MESSAGE,
                channel_id=channel.channel_id,
                channel_name=channel.channel_name,
                content=msg.text,
                metadata={
                    'message_id': msg.message_id,
                    'user': msg.user,
                    'timestamp': msg.timestamp,
                    'is_thread_parent': msg.has_thread,
                    'reply_count': len(msg.thread)
                },
                token_count=self.count_tokens(msg.text)
            ))
            
            # Each reply as separate chunk
            for reply in msg.thread:
                chunks.append(Chunk(
                    chunk_id=f"{channel.channel_id}_{reply.message_id}_reply",
                    chunk_type=ChunkType.INDIVIDUAL_MESSAGE,
                    channel_id=channel.channel_id,
                    channel_name=channel.channel_name,
                    content=reply.text,
                    metadata={
                        'message_id': reply.message_id,
                        'user': reply.user,
                        'timestamp': reply.timestamp,
                        'parent_message_id': msg.message_id,
                        'is_thread_reply': True
                    },
                    token_count=self.count_tokens(reply.text)
                ))
        
        return chunks
    
    def _chunk_threads(self, channel: Channel) -> List[Chunk]:
        """Strategy 2: Each thread (parent + all replies) OR standalone message is a chunk"""
        chunks = []
        
        for msg in channel.messages:
            if msg.has_thread:
                # Combine parent message + all replies into one chunk
                thread_content = f"[Original Message by {msg.user}]\n{msg.text}\n\n"
                thread_content += "\n\n".join([
                    f"[Reply by {reply.user}]\n{reply.text}"
                    for reply in msg.thread
                ])
                
                chunks.append(Chunk(
                    chunk_id=f"{channel.channel_id}_{msg.message_id}_thread",
                    chunk_type=ChunkType.THREAD,
                    channel_id=channel.channel_id,
                    channel_name=channel.channel_name,
                    content=thread_content,
                    metadata={
                        'parent_message_id': msg.message_id,
                        'reply_count': len(msg.thread),
                        'reply_message_ids': [r.message_id for r in msg.thread],
                        'participants': list(set([msg.user] + [r.user for r in msg.thread]))
                    },
                    token_count=self.count_tokens(thread_content)
                ))
            else:
                # Standalone message (no thread)
                chunks.append(Chunk(
                    chunk_id=f"{channel.channel_id}_{msg.message_id}_standalone",
                    chunk_type=ChunkType.THREAD,
                    channel_id=channel.channel_id,
                    channel_name=channel.channel_name,
                    content=msg.text,
                    metadata={
                        'message_id': msg.message_id,
                        'user': msg.user,
                        'timestamp': msg.timestamp,
                        'is_standalone': True
                    },
                    token_count=self.count_tokens(msg.text)
                ))
        
        return chunks
    
    def _chunk_sliding_window(self, channel: Channel) -> List[Chunk]:
        """Strategy 3: Sliding window of top-level messages only (no overlap, max 4096 tokens)"""
        chunks = []
        
        current_window = []
        current_tokens = 0
        window_idx = 0
        
        for msg in channel.messages:
            # Use only the top-level message text (ignore thread replies for this strategy)
            msg_text = msg.text
            msg_tokens = self.count_tokens(msg_text)
            
            # If single message exceeds limit, truncate it
            if msg_tokens > self.max_tokens:
                # Save current window if it has content
                if current_window:
                    chunks.append(self._create_window_chunk(
                        channel, current_window, window_idx
                    ))
                    window_idx += 1
                    current_window = []
                    current_tokens = 0
                
                # Create chunk with truncated message
                truncated_text = self._truncate_to_tokens(msg_text, self.max_tokens)
                chunks.append(Chunk(
                    chunk_id=f"{channel.channel_id}_window_{window_idx}",
                    chunk_type=ChunkType.SLIDING_WINDOW,
                    channel_id=channel.channel_id,
                    channel_name=channel.channel_name,
                    content=truncated_text,
                    metadata={
                        'message_ids': [msg.message_id],
                        'message_count': 1,
                        'truncated': True
                    },
                    token_count=self.count_tokens(truncated_text)
                ))
                window_idx += 1
                continue
            
            # Check if adding this message would exceed limit
            if current_tokens + msg_tokens > self.max_tokens:
                # Save current window
                chunks.append(self._create_window_chunk(
                    channel, current_window, window_idx
                ))
                window_idx += 1
                
                # Start new window
                current_window = [msg]
                current_tokens = msg_tokens
            else:
                # Add to current window
                current_window.append(msg)
                current_tokens += msg_tokens
        
        # Save final window
        if current_window:
            chunks.append(self._create_window_chunk(
                channel, current_window, window_idx
            ))
        
        return chunks
    
    def _create_window_chunk(self, channel: Channel, messages: List[TopLevelMessage], idx: int) -> Chunk:
        """Helper to create a sliding window chunk from top-level messages"""
        content = "\n\n---\n\n".join([
            f"[Message by {msg.user} at {msg.timestamp}]\n{msg.text}"
            for msg in messages
        ])
        
        return Chunk(
            chunk_id=f"{channel.channel_id}_window_{idx}",
            chunk_type=ChunkType.SLIDING_WINDOW,
            channel_id=channel.channel_id,
            channel_name=channel.channel_name,
            content=content,
            metadata={
                'message_ids': [msg.message_id for msg in messages],
                'message_count': len(messages),
                'start_timestamp': messages[0].timestamp,
                'end_timestamp': messages[-1].timestamp,
                'has_threaded_messages': any(msg.has_thread for msg in messages)
            },
            token_count=self.count_tokens(content)
        )
    
    def _truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """Truncate text to fit within token limit"""
        tokens = self.encoder.encode(text)
        if len(tokens) <= max_tokens:
            return text
        return self.encoder.decode(tokens[:max_tokens]) + "..."

# ============================================
# 4. QUERY GENERATOR: Create Synthetic Queries
# ============================================

class QueryGenerator:
    """Generate diverse synthetic queries for chunks"""
    
    PERSONAS = {
        'technical_expert': {
            'description': 'A highly technical user with deep domain knowledge who uses precise terminology and asks sophisticated questions',
            'temperature': 0.7
        },
        'beginner': {
            'description': 'A beginner who is unfamiliar with domain-specific terminology and asks basic, sometimes vague questions',
            'temperature': 0.8
        },
        'typo_prone': {
            'description': 'A user who frequently makes typos, uses abbreviations, and writes informal queries',
            'temperature': 0.9
        },
        'vague': {
            'description': 'A user who asks imprecise questions without enough context, requiring interpretation',
            'temperature': 0.8
        },
        'precise': {
            'description': 'A user who provides extensive context and asks very specific, detailed questions',
            'temperature': 0.6
        }
    }
    
    def __init__(self, api_key: str, provider: str = "anthropic"):
        """
        provider: "anthropic" or "openai"
        """
        self.provider = provider
        if provider == "anthropic":
            self.client = anthropic.Anthropic(api_key=api_key)
        else:
            self.client = openai.OpenAI(api_key=api_key)
    
    def generate_queries_for_chunk(
        self, 
        chunk: Chunk, 
        num_queries: int = 3,
        channel_context: str = ""
    ) -> List[SyntheticQuery]:
        """Generate multiple queries for a single chunk with different personas"""
        queries = []
        
        # Rotate through personas
        persona_list = list(self.PERSONAS.keys())
        
        for i in range(num_queries):
            persona_name = persona_list[i % len(persona_list)]
            persona_config = self.PERSONAS[persona_name]
            
            query_text = self._generate_single_query(
                chunk, persona_name, persona_config, channel_context
            )
            
            if query_text:
                queries.append(SyntheticQuery(
                    query_id=f"{chunk.chunk_id}_query_{i}",
                    query_text=query_text,
                    chunk_id=chunk.chunk_id,
                    persona=persona_name,
                    difficulty=self._assess_difficulty(query_text, chunk.content),
                    metadata={
                        'chunk_type': chunk.chunk_type.value,
                        'channel_name': chunk.channel_name
                    }
                ))
        
        return queries
    
    def _generate_single_query(
        self, 
        chunk: Chunk, 
        persona_name: str, 
        persona_config: Dict,
        channel_context: str
    ) -> str:
        """Generate a single query using LLM"""
        
        system_prompt = f"""You are roleplaying as a user searching through Slack messages.

PERSONA: {persona_name}
{persona_config['description']}

Your task is to generate a realistic search query that this user would write to find the information in the given Slack message(s).

Guidelines:
- Stay in character for this persona
- The query should be a natural question someone would ask
- The answer to the query MUST be contained in the provided message content
- Make the query specific enough to be answerable but realistic
- DO NOT directly copy phrases from the message
- Output ONLY the query text, nothing else

Channel context: {channel_context if channel_context else "General discussion channel"}
"""

        user_prompt = f"""Message content to create a query for:

{chunk.content}

Generate a query that this {persona_name} would write to find this information:"""

        try:
            if self.provider == "anthropic":
                response = self.client.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=200,
                    temperature=persona_config['temperature'],
                    system=system_prompt,
                    messages=[{
                        "role": "user",
                        "content": user_prompt
                    }]
                )
                return response.content[0].text.strip()
            else:  # openai
                response = self.client.chat.completions.create(
                    model="gpt-4",
                    max_tokens=200,
                    temperature=persona_config['temperature'],
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ]
                )
                return response.choices[0].message.content.strip()
        
        except Exception as e:
            print(f"Error generating query: {e}")
            return None
    
    def _assess_difficulty(self, query: str, content: str) -> str:
        """Simple heuristic to assess query difficulty"""
        query_len = len(query.split())
        content_len = len(content.split())
        
        # More complex logic could go here
        if query_len < 5:
            return "easy"
        elif query_len < 15:
            return "medium"
        else:
            return "hard"

# ============================================
# 5. VALIDATOR: Ensure Query-Chunk Relevance
# ============================================

class QueryValidator:
    """Validate that queries can actually be answered by their chunks"""
    
    def __init__(self, api_key: str, provider: str = "anthropic"):
        self.provider = provider
        if provider == "anthropic":
            self.client = anthropic.Anthropic(api_key=api_key)
        else:
            self.client = openai.OpenAI(api_key=api_key)
    
    def validate_query(self, query: SyntheticQuery, chunk: Chunk) -> Tuple[bool, str]:
        """
        Validate if chunk contains answer to query
        Returns: (is_valid, reason)
        """
        
        system_prompt = """You are a validator checking if a Slack message contains the answer to a query.

Output ONLY a JSON object with this format:
{"valid": true/false, "reason": "explanation"}

The message must actually contain enough information to answer the query."""

        user_prompt = f"""Query: {query.query_text}

Message content:
{chunk.content}

Does this message contain the answer to the query?"""

        try:
            if self.provider == "anthropic":
                response = self.client.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=150,
                    temperature=0,
                    system=system_prompt,
                    messages=[{
                        "role": "user",
                        "content": user_prompt
                    }]
                )
                result = json.loads(response.content[0].text.strip())
            else:
                response = self.client.chat.completions.create(
                    model="gpt-4",
                    max_tokens=150,
                    temperature=0,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ]
                )
                result = json.loads(response.choices[0].message.content.strip())
            
            return result.get('valid', False), result.get('reason', '')
        
        except Exception as e:
            print(f"Validation error: {e}")
            return False, f"Error: {e}"

# ============================================
# 6. PIPELINE: Orchestrate Everything
# ============================================

class SyntheticDataPipeline:
    """Full pipeline from raw Slack dump to validated query-chunk pairs"""
    
    def __init__(
        self, 
        archive_dir: str,
        output_dir: str,
        api_key: str,
        provider: str = "anthropic",
        queries_per_chunk: int = 3
    ):
        self.archive_dir = archive_dir
        self.output_dir = output_dir
        self.queries_per_chunk = queries_per_chunk
        
        self.post_processor = SlackPostProcessor(archive_dir)
        self.chunker = SlackChunker()
        self.query_generator = QueryGenerator(api_key, provider)
        self.validator = QueryValidator(api_key, provider)
    
    def process_workspace(self, workspace_name: str):
        """Full pipeline for one workspace"""
        
        print(f"\n{'='*60}")
        print(f"Processing workspace: {workspace_name}")
        print(f"{'='*60}\n")
        
        # Step 1: Post-process raw data
        print("📝 Step 1: Post-processing raw Slack data...")
        channels = self.post_processor.process_workspace(workspace_name)
        print(f"   ✅ Processed {len(channels)} channels")
        
        # Save cleaned channels
        cleaned_dir = f"{self.output_dir}/{workspace_name}/cleaned_channels"
        for channel in channels:
            self.post_processor.save_cleaned_channel(channel, cleaned_dir)
        print(f"   ✅ Saved cleaned channels to {cleaned_dir}")
        
        # Step 2: Create chunks
        print("\n🔪 Step 2: Creating chunks...")
        all_chunks = {}
        for channel in channels:
            chunks_by_type = self.chunker.chunk_channel(channel)
            for chunk_type, chunks in chunks_by_type.items():
                if chunk_type not in all_chunks:
                    all_chunks[chunk_type] = []
                all_chunks[chunk_type].extend(chunks)
        
        for chunk_type, chunks in all_chunks.items():
            print(f"   ✅ {chunk_type.value}: {len(chunks)} chunks")
        
        # Save chunks
        chunks_dir = f"{self.output_dir}/{workspace_name}/chunks"
        os.makedirs(chunks_dir, exist_ok=True)
        for chunk_type, chunks in all_chunks.items():
            chunk_file = f"{chunks_dir}/{chunk_type.value}.jsonl"
            with open(chunk_file, 'w', encoding='utf-8') as f:
                for chunk in chunks:
                    f.write(json.dumps(chunk.to_dict()) + '\n')
            print(f"   💾 Saved to {chunk_file}")
        
        # Step 3: Generate queries
        print("\n❓ Step 3: Generating synthetic queries...")
        all_queries = []
        
        # Sample chunks for query generation (you might want to do all)
        sample_chunks = []
        for chunk_type, chunks in all_chunks.items():
            sample_chunks.extend(chunks[:100])  # Sample 100 from each type
        
        for i, chunk in enumerate(sample_chunks, 1):
            if i % 10 == 0:
                print(f"   Progress: {i}/{len(sample_chunks)} chunks")
            
            queries = self.query_generator.generate_queries_for_chunk(
                chunk, 
                num_queries=self.queries_per_chunk,
                channel_context=f"#{chunk.channel_name}"
            )
            all_queries.extend(queries)
        
        print(f"   ✅ Generated {len(all_queries)} queries")
        
        # Step 4: Validate queries
        print("\n✔️  Step 4: Validating query-chunk pairs...")
        validated_pairs = []
        
        # Create chunk lookup
        chunk_lookup = {}
        for chunk_type, chunks in all_chunks.items():
            for chunk in chunks:
                chunk_lookup[chunk.chunk_id] = chunk
        
        for i, query in enumerate(all_queries, 1):
            if i % 10 == 0:
                print(f"   Progress: {i}/{len(all_queries)} queries")
            
            chunk = chunk_lookup.get(query.chunk_id)
            if chunk:
                is_valid, reason = self.validator.validate_query(query, chunk)
                if is_valid:
                    validated_pairs.append({
                        'query': query.query_text,
                        'chunk_id': query.chunk_id,
                        'chunk_content': chunk.content,
                        'persona': query.persona,
                        'difficulty': query.difficulty,
                        'metadata': query.metadata
                    })
        
        print(f"   ✅ {len(validated_pairs)}/{len(all_queries)} queries validated")
        
        # Save validated pairs
        output_file = f"{self.output_dir}/{workspace_name}/validated_query_chunk_pairs.jsonl"
        with open(output_file, 'w', encoding='utf-8') as f:
            for pair in validated_pairs:
                f.write(json.dumps(pair) + '\n')
        
        print(f"\n🎉 Pipeline complete!")
        print(f"   📊 Final dataset: {len(validated_pairs)} query-chunk pairs")
        print(f"   💾 Saved to: {output_file}")

# ============================================
# 7. USAGE
# ============================================

if __name__ == "__main__":
    pipeline = SyntheticDataPipeline(
        archive_dir="slack_archive",
        output_dir="synthetic_data",
        api_key="your-anthropic-or-openai-key",
        provider="anthropic",  # or "openai"
        queries_per_chunk=3
    )
    
    pipeline.process_workspace("Flutter_Community_TGT6YF2J1")