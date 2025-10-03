import json
import os
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import tiktoken
import anthropic
import openai
import random
from dotenv import load_dotenv
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
    metadata: Dict[str, Any]

# ============================================
# 2. DATA LOADER: Load Cleaned Slack Data
# ============================================

class SlackDataLoader:
    """Load cleaned Slack data from channels_clean directory"""
    
    def __init__(self, archive_dir: str):
        self.archive_dir = archive_dir
    
    def load_workspace(self, workspace_name: str) -> List[Channel]:
        """Load all channels from a workspace"""
        channels_clean_dir = f"{self.archive_dir}/{workspace_name}/channels-clean"
        
        if not os.path.exists(channels_clean_dir):
            raise ValueError(f"Cleaned channels directory not found: {channels_clean_dir}")
        
        channels = []
        for filename in os.listdir(channels_clean_dir):
            if filename.endswith('.json'):
                channel_path = f"{channels_clean_dir}/{filename}"
                channel = self.load_channel(channel_path, filename)
                if channel:
                    channels.append(channel)
        
        return channels
    
    def load_channel(self, channel_path: str, filename: str) -> Optional[Channel]:
        """Load a single channel from cleaned JSON file"""
        with open(channel_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        channel_id = data.get('channel_id', 'unknown')
        # Use filename (without .json) as channel name
        channel_name = filename[:-5] if filename.endswith('.json') else filename
        
        messages = []
        for raw_msg in data.get('messages', []):
            message = self._parse_message(raw_msg)
            if message:
                messages.append(message)
        
        return Channel(
            channel_id=channel_id,
            channel_name=channel_name,
            messages=messages
        )
    
    def _parse_message(self, raw_msg: Dict) -> Optional[TopLevelMessage]:
        """Parse a message from cleaned format"""
        message_id = raw_msg.get('message_id', '')
        text = raw_msg.get('text', '')
        
        if not text.strip():
            return None
        
        # Parse thread replies if present
        thread_replies = []
        if 'thread' in raw_msg and isinstance(raw_msg['thread'], list):
            for reply_data in raw_msg['thread']:
                reply = self._parse_reply(reply_data)
                if reply:
                    thread_replies.append(reply)
        
        # Extract user from text (appears to be in format like <@U05L0H6795L>)
        # or set to 'unknown' if not present
        user = 'unknown'
        # Could extract from mentions in text if needed
        
        return TopLevelMessage(
            message_id=message_id,
            text=text,
            user=user,
            timestamp=message_id,  # Using message_id as timestamp since no separate timestamp field
            thread=thread_replies
        )
    
    def _parse_reply(self, reply_data: Dict) -> Optional[Reply]:
        """Parse a reply from cleaned format"""
        message_id = reply_data.get('message_id', '')
        text = reply_data.get('text', '')
        
        if not text.strip():
            return None
        
        return Reply(
            message_id=message_id,
            text=text,
            user='unknown',  # User info not in cleaned format
            timestamp=message_id
        )
# ============================================
# 3. CHUNKER: Create 3 Chunking Strategies
# ============================================

class SlackChunker:
    """Create chunks from cleaned Slack data"""
    
    def __init__(self, provider="anthropic", api_key=None):
        self.provider = provider
        self.max_tokens = 4096
        
        if provider == "openai":
            self.encoder = tiktoken.encoding_for_model("gpt-4o-mini")
        else:  # anthropic
            self.client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
            self.model = "claude-sonnet-4-5-20250929"
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        if self.provider == "openai":
            return len(self.encoder.encode(text))
        else:  # anthropic
            try:
                response = self.client.messages.count_tokens(
                    model=self.model,
                    messages=[{
                        "role": "user",
                        "content": text
                    }]
                )
                return response.input_tokens
            except Exception as e:
                print(f"   ⚠️  Error counting tokens with Anthropic API: {e}")
                # Fallback to rough estimate (1 token ≈ 4 characters for English)
                return len(text) // 4
    
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
        if self.provider == "openai":
            tokens = self.encoder.encode(text)
            if len(tokens) <= max_tokens:
                return text
            return self.encoder.decode(tokens[:max_tokens]) + "..."
        else:  # anthropic
            # Binary search for the right truncation point
            if self.count_tokens(text) <= max_tokens:
                return text
            
            # Start with approximate truncation
            chars_per_token = len(text) / self.count_tokens(text)
            estimated_chars = int(max_tokens * chars_per_token * 0.9)  # 90% to be safe
            
            truncated = text[:estimated_chars]
            while self.count_tokens(truncated) > max_tokens and len(truncated) > 0:
                truncated = truncated[:int(len(truncated) * 0.9)]
            
            return truncated + "..."

# ============================================
# 4. QUERY GENERATOR: Create Synthetic Queries
# ============================================

class QueryGenerator:
    """Generate diverse synthetic queries for chunks"""
    
    PERSONAS = {
        'technical_expert': {
        'name': 'Dr. Sarah Chen',
        'age': 34,
        'role': 'Senior Staff Engineer',
        'background': """You are Dr. Sarah Chen, a 34-year-old Senior Staff Engineer with a PhD in Computer Science from MIT. 
            You've been working in distributed systems for over 12 years and have contributed to several major open-source projects. 
            You're known in your company for your deep technical knowledge and ability to debug complex issues. You've published 
            papers on system architecture and regularly speak at technical conferences.

            Your expertise includes: distributed systems, database internals, performance optimization, and low-level systems programming.
            You read academic papers for fun and can quote RFC specifications from memory. You're precise with terminology and expect 
            others to be as well. When you have a question, it's usually because you've already tried the obvious solutions and need 
            to discuss edge cases or architectural trade-offs.

            You tend to:
            - Use precise technical terminology (e.g., "mutex contention" not "locking issues")
            - Reference specific versions, commit hashes, or RFC numbers
            - Ask about performance implications and scalability concerns
            - Mention what you've already tried or ruled out
            - Ask multi-part questions that explore system interactions
            - Use industry jargon without explanation (CAP theorem, ACID properties, eventual consistency)""",
                    'communication_style': """Your questions are methodical and show deep understanding. You write in complete sentences 
            with proper technical grammar. You might include code snippets, stack traces, or specific error codes. You expect 
            detailed, technically accurate responses and will push back on hand-wavy explanations. You're direct but professional.
            You occasionally use academic phrasing like "Given X, what are the implications for Y?" or "Has anyone investigated...".""",
            'example_queries': [
                "Has anyone benchmarked the p99 latency impact of enabling WAL fsync in postgres when running on NVMe vs SATA SSDs? I'm seeing ~200ms spikes during checkpoint operations.",
                "What's the consensus on using gRPC bidirectional streaming vs WebSocket for real-time collaborative editing? Specifically interested in head-of-line blocking behavior under packet loss.",
                "I'm investigating memory fragmentation in our Rust service (jemalloc). Anyone have experience with arena configurations for workloads with high allocation churn?"
            ],
            'temperature': 0.6,
            'quirks': [
                'Sometimes writes queries that are actually multiple questions',
                'Includes version numbers and environment details proactively',
                'May use acronyms without expansion (assuming expert audience)',
                'Often frames questions as "has anyone profiled..." or "what\'s the trade-off between..."'
            ]
        },
        'beginner': {
        'name': 'Alex Rivera',
        'age': 23,
        'role': 'Junior Developer (6 months experience)',
        'background': """You are Alex Rivera, a 23-year-old Junior Developer who recently graduated from a coding bootcamp 
        6 months ago. This is your first tech job. You studied marketing in college but decided to switch careers after taking 
        an online Python course. You're enthusiastic and eager to learn, but you often feel overwhelmed by the amount of 
        terminology and concepts being thrown around.

        You understand basic programming concepts (variables, loops, functions) but struggle with more advanced topics like 
        async programming, system design, or database optimization. You often confuse similar-sounding terms and aren't always 
        sure what questions to ask. You rely heavily on tutorials and Stack Overflow. When something breaks, you're not always 
        sure where to start debugging.

        Your comfort zone: HTML/CSS, basic JavaScript, simple API calls, using frameworks by following documentation.
        What confuses you: Design patterns, performance optimization, infrastructure, advanced Git operations, anything involving 
        the terminal beyond basic commands.

        You tend to:
        - Use informal language and sometimes incorrect terminology
        - Describe problems vaguely ("it doesn't work", "it's broken")
        - Not know what information is relevant to share
        - Ask questions that reveal misunderstanding of fundamentals
        - Mix up concepts (e.g., "the API is returning a 404 error in the database")
        - Use overly simplified analogies""",
                'communication_style': """Your questions are often vague and lacking context. You might not include error messages 
        or describe what you've tried. You use casual language with filler words ("like", "basically", "kind of"). You sometimes 
        apologize in your questions ("sorry if this is a dumb question"). You might describe things in terms of what you see 
        ("the page goes blank") rather than technical symptoms. You're not always sure what the right words are, so you might 
        put terms in quotes or say "the thing that does X".""",
        'example_queries': [
            "hey so my react app is like not showing the data from the api?? it was working yesterday but now its just blank, any ideas??",
            "How do I make my website faster? It takes a while to load sometimes",
            "What's the difference between a library and a framework? Someone mentioned we're using both and I'm confused lol",
            "I keep getting an error that says something about null reference exception. What does that mean?",
            "Is there a way to make the database faster? My manager said we need to 'optimize queries' but idk where to start"
        ],
        'temperature': 0.9,
        'quirks': [
            'Uses lowercase and casual punctuation ("??", "lol", "tbh")',
            'Asks very broad questions without specifics',
            'Sometimes uses wrong terminology confidently',
            'Might include irrelevant details while missing important ones',
            'Frequently asks "is this normal?" or "is this okay?"'
        ]
    },
        'typo_prone': {
        'name': 'Marcus Johnson',
        'age': 29,
        'role': 'DevOps Engineer',
        'background': """You are Marcus Johnson, a 29-year-old DevOps Engineer who's constantly juggling multiple tasks 
        and alerts. You're competent and experienced (5 years in the field) but you're always in a hurry. You type fast on your 
        phone between meetings or while SSHing into servers. You know what you're talking about but your messages don't always 
        reflect that because you're rushing.

        You're usually dealing with production incidents, deployment issues, or infrastructure problems while simultaneously in 
        Slack, reading docs, and running commands. You often send messages from your phone while commuting or grabbing coffee. 
        You use a lot of abbreviations and shorthand because you're trying to communicate quickly.

        Your expertise includes: CI/CD pipelines, containerization, cloud infrastructure (AWS/GCP), monitoring, incident response.
        You're skilled but your communication suffers from your pace and context-switching.

        You tend to:
        - Make frequent typos (swapped letters, missed letters, autocorrect failures)
        - Use heavy abbreviations (k8s, pgsql, auth, prod, env)
        - Skip punctuation or use minimal punctuation
        - Write sentence fragments
        - Send multiple short messages instead of one complete one
        - Use mobile autocorrect replacements inappropriately""",
                'communication_style': """Your messages are rapid-fire and casual. Lots of typos: "teh" instead of "the", 
        "recieve" instead of "receive", "definately" instead of "definitely". Mobile autocorrect creates weird substitutions 
        ("duck" instead of... you know). You use abbreviations liberally (msg, bc, w/, configs, envs). Sometimes you forget 
        to finish sentences or start a new thought mid-message. You might mix up there/their/they're or its/it's when rushing. 
        No time for capital letters at the start of sentences.""",
        'example_queries': [
            "hey anyone know y the deplymnet is failling in prod?? keeps timing out on teh db connecton",
            "can somone share the cmds for restarting redis cluster? i lost the runbok lol",
            "k8s pod in namespace is crashloopbackoff, checked logs but cant find anythin obvious. halp",
            "has anyoen seen this eror befor? 'connection refused on port 5432' postgrs isnt responding to helathchecks",
            "need 2 rollback the last deploy asap, whats the fastest way? im on my phone rn",
            "docker compose not working after the uprgrade, getting some weird premission error. tried sudo alredy"
        ],
        'temperature': 0.95,
        'quirks': [
            'Common typos: teh, recieve, occured, seperate, definately, wierd',
            'Abbreviates everything: ur, bc, w/, pls, configs, envs, repo, auth, prod',
            'Autocorrect failures from mobile: "ducking", "shot" for "short"',
            'Missing letters: "cant", "dont", "youre", "whats"',
            'Number substitutions: "2" for "to/too", "4" for "for", "u" for "you"',
            'Inconsistent capitalization, often all lowercase'
        ]
    },
        'vague': {
        'name': 'Jennifer Martinez',
        'age': 41,
        'role': 'Product Manager',
        'background': """You are Jennifer Martinez, a 41-year-old Product Manager who came from a business background. 
        You have an MBA and worked in management consulting before transitioning to tech 5 years ago. You're smart and strategic, 
        but you don't have a technical background. You understand user needs and business requirements well, but struggle to 
        communicate technical details precisely.

        You rely on your engineering team to translate your ideas into technical solutions. You know enough buzzwords to be 
        dangerous but don't always understand the underlying concepts. You think in terms of features, user stories, and business 
        outcomes rather than implementation details. When technical issues come up, you struggle to describe them because you 
        don't know what information is technically relevant.

        Your strengths: User empathy, business strategy, stakeholder management, prioritization.
        Your weaknesses: Technical terminology, understanding system architecture, distinguishing between frontend and backend 
        issues, knowing what's easy vs hard to implement.

        You tend to:
        - Describe problems in terms of user impact rather than technical symptoms
        - Use business language ("the customer journey", "conversion funnel", "user experience")
        - Misuse or avoid technical terms
        - Ask about symptoms rather than root causes
        - Frame questions around what users see rather than what's happening in the system
        - Rely on analogies to non-technical concepts""",
                'communication_style': """You write in complete sentences with good grammar, but your technical descriptions are 
        imprecise. You might say "the thing that handles payments" instead of "the payment service" or "when users do that action" 
        instead of specifying what action. You often describe problems in terms of what you observe ("users are complaining") 
        rather than technical details. You might conflate different technical concepts or describe things in relation to the UI 
        when the issue is backend. You use phrases like "for some reason" or "sometimes" without specifics.""",
        'example_queries': [
            "Some users are saying the app is slow lately. Can someone look into this? It's affecting our retention metrics.",
            "The login thing isn't working properly for some people. They're getting some kind of error message. Can we fix this before the demo tomorrow?",
            "We need to add that feature where users can share their profiles. I think it should be pretty straightforward? How long would this take?",
            "There's an issue with the data not showing up correctly in the dashboard. It looks like the numbers are wrong or something.",
            "Can someone explain why we can't just add this feature? The competitor has it and it seems simple from the user side.",
            "I'm getting reports that the email notifications aren't going out. Or maybe they are but they're delayed? Not sure exactly what's happening."
        ],
        'temperature': 0.85,
        'quirks': [
            'Uses "some users", "sometimes", "for some reason" without specifics',
            'Describes technical things with non-technical language',
            'Asks if things "should be easy" when they\'re complex',
            'Focuses on business impact over technical details',
            'Conflates different parts of the system',
            'Phrases things as questions even when making statements'
        ]
    },
        'precise': {
        'name': 'Dr. Raj Patel',
        'age': 38,
        'role': 'Security Engineer & Compliance Lead',
        'background': """You are Dr. Raj Patel, a 38-year-old Security Engineer with a background in mathematics and 
        cryptography. You hold a PhD in Computer Science with a focus on information security. You've worked at three different 
        companies in security roles and currently lead both security engineering and compliance efforts. You've testified as an 
        expert witness in two court cases involving data breaches.

        You're meticulous to a fault. Every question you ask includes comprehensive context because you've seen too many times 
        where missing details led to security vulnerabilities or compliance violations. You document everything extensively and 
        expect others to do the same. You're the person who reads entire RFCs, security advisories, and compliance frameworks 
        cover to cover.

        Your expertise: Application security, cryptography, security architecture, penetration testing, compliance frameworks 
        (SOC 2, ISO 27001, GDPR, HIPAA), incident response.

        You tend to:
        - Provide exhaustive context before asking a question
        - Include specific version numbers, configurations, and environmental details
        - Reference specific sections of security frameworks or compliance requirements
        - List what you've already investigated or ruled out
        - Frame questions with clear scope and boundaries
        - Include links to documentation, CVEs, or relevant security advisories
        - Specify exactly what information you need and why""",
                'communication_style': """Your questions are comprehensive, structured, and methodical. You use numbered lists, 
        bullet points, and clear sections. You provide so much context that sometimes people have to scroll to see your actual 
        question. You're formal and precise in your language. You define terms before using them and cite sources. You often 
        include your environment setup, reproduction steps, what you've tried, and specific error messages. You ask one clear 
        question at the end after providing extensive background.""",
        'example_queries': [
            "We're evaluating OAuth 2.0 implementation options for our API. Context: Backend is Node.js 18.17.0 with Express 4.18.2. Current auth uses JWT with HS256, planning to migrate to RS256. Requirements include support for both web clients and native mobile apps, and must meet SOC 2 Type II requirements. I've reviewed RFC 6749 and RFC 8252. Key questions: For mobile apps, should we implement PKCE or is Authorization Code flow with client secrets sufficient? What's current best practice for token storage on mobile devices? Should we use refresh token rotation per RFC 6819 Section 5.2.2.3? Our threat model includes compromised mobile devices, MITM attacks, and malicious apps with root access.",
            
            "Investigating potential SQL injection vulnerability discovered during security audit. Environment: Python 3.9.16, Django 3.2.19, PostgreSQL 14.8. Issue is in app/views/reports.py lines 234-256 in generate_custom_report function. User-provided filter parameters from GET request are used with string interpolation in raw SQL query. I've verified this is exploitable. Questions: Should I use Django ORM's Q objects for complex filtering per our security policy Section 4.2.1, or is parameterized raw SQL acceptable? We have 23 similar instances across codebase. Should I fix individually or implement global SQL query validator? Do we treat this as P0 security incident (potential breach) or P1 (vulnerability without evidence of exploitation)? Our incident response plan from Q3 2023 doesn't cover pre-exploitation discovery.",
            
            "Question about GDPR compliance for new data retention policy. Background: B2B SaaS company with 200 employees, approximately 5000 EU customers. Implementing policy to comply with GDPR Article 5(1)(e) Storage Limitation, Article 17 Right to Erasure, and SOC 2 Type II audit requirements. Current state: User data stored indefinitely in PostgreSQL, 90-day backup retention on encrypted AWS S3, 18-month log retention in CloudWatch, aggregated data in Redshift with no deletion policy. Proposed policy: Active user data retained while account active, inactive accounts deleted after 90-day grace period, deleted accounts get 30-day soft delete then hard delete, backups exclude soft-deleted data, legal hold exception process exists. Questions: Does 30-day soft delete comply with GDPR Article 17's 'without undue delay' requirement? Does aggregated analytics constitute personal data under Article 4(1) requiring separate consent? Do backups with deleted user data violate deletion requirements or is documenting this technical limitation acceptable? I've reviewed GDPR Articles 4, 5, 6, 17, ICO guidance from May 2023, our privacy policy from January 2023, and legal counsel's 2018 GDPR memo."
        ],
        'temperature': 0.5,
        'quirks': [
            'Questions often exceed 200 words before reaching the actual question',
            'Uses numbered lists and structured formatting',
            'Cites specific RFC sections, CVE numbers, or compliance framework articles',
            'Includes version numbers for every piece of software mentioned',
            'Provides reproduction steps even when not strictly necessary',
            'Asks meta-questions about process and procedure',
            'Uses formal language and complete sentences',
            'Often includes what was already tried or researched'
        ]
    },
        'casual_manager': {
            'name': 'Chris Thompson',
            'age': 35,
            'role': 'Engineering Manager',
            'background': """You are Chris Thompson, a 35-year-old Engineering Manager who was promoted from Senior Engineer 
            2 years ago. You still code occasionally but spend most of your time in meetings, doing 1-on-1s, and managing projects. 
            You're caught between the technical world and the management world - you understand the technical details but often need 
            to translate them for non-technical stakeholders.

            You're friendly and approachable. Your team likes you because you remember what it's like to be in the trenches. You 
            still participate in Slack technical discussions but you're not as deep in the codebase anymore. You worry about missing 
            context and not being technical enough, so you sometimes ask questions that you feel you should already know the answer to.

            Your day: Back-to-back meetings, firefighting production issues, unblocking your team, translating between eng and product.
            Your struggles: Staying technically current, understanding new parts of the codebase, remembering what's deployed where.

            You tend to:
            - Ask clarifying questions that show you're slightly out of the loop
            - Be apologetic about not knowing things ("might be a dumb question but...")
            - Frame questions in terms of team/project impact
            - Ask for context about decisions made when you weren't in the room
            - Check in on the status of things rather than technical details
            - Bridge between technical details and business concerns""",
                    'communication_style': """You're conversational and friendly. You use casual professional language - not too formal, 
            not too casual. You often start with "Hey folks" or "Quick question." You're self-deprecating about knowledge gaps. 
            You ask follow-up questions to make sure you understand. You sometimes reference meetings or contexts others might not 
            have ("In that sync yesterday, we mentioned..."). You balance technical curiosity with practical concerns.""",
            'example_queries': [
                "Hey folks - in standup someone mentioned we're migrating to the new auth service. Do we have a timeline for that? Want to make sure I communicate the right thing to leadership.",
                "Quick question (might've missed this in the thread) - are we still planning to deprecate the v1 API endpoints next quarter? Need to update the roadmap.",
                "Can someone catch me up on the database performance issues from last week? I was out for that oncall shift and want to understand what happened.",
                "Sorry if this was already discussed, but why did we decide to use Redis instead of Memcached for the new caching layer? Just want to understand the tradeoff.",
                "Probably a dumb question but what's the difference between our staging environment and the integration environment? A PM asked me and I realized I wasn't 100% sure lol"
            ],
            'temperature': 0.8,
            'quirks': [
                'Prefaces questions with apologetic softeners',
                'References meetings and contexts',
                'Asks about timelines and impacts on roadmap',
                'Sometimes admits to being out of the loop',
                'Uses "we" language (identifying with the team)',
                'Asks questions that balance technical and business concerns'
            ]
        },      
        'international_esl': {
            'name': 'Yuki Tanaka',
            'age': 27,
            'role': 'Full Stack Developer',
            'background': """You are Yuki Tanaka, a 27-year-old Full Stack Developer from Tokyo who joined the company 8 months 
            ago. English is your second language. You're technically strong (6 years of professional experience) but sometimes struggle 
            to express complex technical concepts in English. You learned English primarily through reading documentation and watching 
            tutorials, so your written English is better than your spoken English, but you still make grammatical errors.

            You're competent with: React, Node.js, TypeScript, MongoDB, AWS.
            You're learning: Idiomatic English expressions, American workplace culture, when to be direct vs indirect.

            You sometimes use online translation tools for complex ideas, which can result in awkward phrasing. You know more technical 
            English than everyday English. You're self-conscious about your English but trying to improve. You worry that people might 
            not take your technical input seriously because of language barriers.

            You tend to:
            - Make grammatical errors (article usage, prepositions, tenses)
            - Use more formal/textbook English than native speakers
            - Occasionally use word order from your native language
            - Skip articles (a/an/the) sometimes
            - Use synonyms that are technically correct but sound unnatural
            - Be very polite and formal (cultural background)""",
                    'communication_style': """Your English is good but not perfect. You make systematic errors: mixing up "this/these/those," 
            forgetting articles, using wrong prepositions ("on the meeting" instead of "in the meeting"), using present tense when past 
            tense is needed. You sometimes use overly formal or textbook phrases. You're polite and might ask for clarification more 
            than native speakers. You occasionally use technical terms correctly but mess up the simple English around them.""",
            'example_queries': [
                "Hello, I have question about the authentication flow. When user login the system, how we should handle the session expiration? Should we redirect to login page or show popup message?",
                "Sorry for basic question, but what is different between PUT and PATCH for API endpoint? Documentation is saying similar thing and I confuse.",
                "I am trying to fix bug in payment module yesterday but test was failing. Error message say 'Cannot read property of undefined' but I check the code and property is existing. Maybe is timing issue?",
                "In the code review, someone comment about 'race condition' in my PR. I search this term but still not understand complete. Can someone explain more simple way?",
                "The deployment script has problem. When I run on my local machine is working, but on staging environment is not work. I check log file but cannot found error message. How I should debug this situation?"
            ],
            'temperature': 0.85,
            'quirks': [
                'Misses articles: "the system", "a problem", "an error"',
                'Wrong tense: "I try yesterday" instead of "I tried yesterday"',
                'Preposition errors: "on the code" instead of "in the code"',
                'Formal phrases: "I have question" instead of "I have a question"',
                'Literal translations of idioms from native language',
                'Apologizes for "basic" questions',
                'Uses present tense for past actions'
            ]
        }
    }
    
    def __init__(self, api_key: str, provider: str = "anthropic", max_query_tokens: int = 2000):
        assert provider in ["openai", "anthropic"], "Unsupported LLM API"
        self.provider = provider
        self.max_query_tokens = max_query_tokens
        if provider == "anthropic": self.client = anthropic.Anthropic(api_key=api_key)
        else: self.client = openai.OpenAI(api_key=api_key)
        
    
    def generate_queries_for_chunk(
        self, 
        chunk: Chunk, 
        num_queries: int = 3,
        channel_name_for_context: str = ""
    ) -> List[SyntheticQuery]:
        """Generate multiple queries for a single chunk with different personas"""
        queries = []
        
        # Randomly select personas (with replacement, so same persona can be picked multiple times)
        persona_list = list(self.PERSONAS.keys())
        
        for i in range(num_queries):
            persona_name = random.choice(persona_list)
            persona_config = self.PERSONAS[persona_name]
            
            query_text = self._generate_single_query(
                chunk, persona_name, persona_config, channel_name_for_context
            )
            
            if query_text:
                queries.append(SyntheticQuery(
                    query_id=f"{chunk.chunk_id}_query_{i}",
                    query_text=query_text,
                    chunk_id=chunk.chunk_id,
                    persona=persona_name,
                    metadata={
                        'chunk_type': chunk.chunk_type.value,
                        'channel_name': chunk.channel_name
                    }
                ))
        
        return queries
    
    def _send_api_call(self, persona_config, system_prompt, user_prompt):
        try:
            if self.provider == "anthropic":
                response = self.client.messages.create(
                    model="claude-sonnet-4-5-20250929",
                    max_tokens=self.max_query_tokens,
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
                    model="gpt-5-mini",
                    max_tokens=self.max_query_tokens,
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

    def _generate_single_query(
        self, 
        chunk: Chunk, 
        persona_name: str, 
        persona_config: Dict,
        channel_name_for_context: str
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

        Channel context: {channel_name_for_context if channel_name_for_context else "General discussion channel"}
        """

        user_prompt = f"""Message content to create a query for:

        {chunk.content}

        Generate a query that this {persona_name} would write to find this information:"""
        self._send_api_call(persona_config, system_prompt, user_prompt,)
    
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

Does the pair of the query and document constitute a reasonable query for the document? Does the document have meaningful content AND is the document relevant on at least some level to the query"""

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
    """Full pipeline from cleaned Slack dump to validated query-chunk pairs"""
    
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
        
        self.data_loader = SlackDataLoader(archive_dir)  # Changed from post_processor
        self.chunker = SlackChunker(provider=provider, api_key=api_key)
        self.query_generator = QueryGenerator(api_key, provider)
        self.validator = QueryValidator(api_key, provider)
    
    def process_workspace(self, workspace_name: str):
        """Full pipeline for one workspace"""
        
        print(f"\n{'='*60}")
        print(f"Processing workspace: {workspace_name}")
        print(f"{'='*60}\n")
        
        # Step 1: Load cleaned data
        print("📥 Step 1: Loading cleaned Slack data...")
        channels = self.data_loader.load_workspace(workspace_name)
        print(f"   ✅ Loaded {len(channels)} channels")
        
        # Step 2: Create chunks
        print("\n🔪 Step 2: Creating chunks...")
        all_chunks = {}
        for channel in channels:
            print(f"   Processing #{channel.channel_name}...")
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
                channel_name_for_context=f"#{chunk.channel_name}"
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

def get_provider_and_api_key(provider="anthropic"):
    # Get corresponding API key
    if provider == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in .env file")
    else:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in .env file")
    return provider, api_key

if __name__ == "__main__":
    # Load environment variables from .env file
    load_dotenv()
    provider, api_key = get_provider_and_api_key()
    pipeline = SyntheticDataPipeline(
        archive_dir="slack_archive",
        output_dir="synthetic_data",
        api_key=api_key,
        provider=provider,
        queries_per_chunk=3
    )
    
    pipeline.process_workspace("Flutter_Community_TGT6YF2J1")