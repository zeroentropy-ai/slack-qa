#!/usr/bin/env python3
"""
Setup Slack collection and index all Slack archive data.
Supports both channels-clean and raw channels formats.
"""
import json
import os
import uuid
import requests
from typing import List, Dict, Any, Iterator
from tqdm import tqdm
import time

# Solr configuration
SOLR_URL = "http://localhost:8983/solr"
COLLECTION = "training-slack"
SLACK_ARCHIVE_DIR = "../slack_archive"

def check_core_exists(core_name: str) -> bool:
    """Check if a Solr core exists"""
    try:
        url = f"{SOLR_URL}/admin/cores"
        params = {"action": "STATUS"}
        
        response = requests.get(url, params=params)
        if response.status_code == 200:
            cores = response.json().get('status', {})
            return core_name in cores
        else:
            print(f"❌ Failed to check cores: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error checking cores: {e}")
        return False

def parse_clean_format(file_path: str, workspace_name: str, channel_name: str) -> Iterator[Dict[str, Any]]:
    """Parse channels-clean format and yield documents"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        channel_id = data.get('channel_id', channel_name)
        messages = data.get('messages', [])
        
        for message in messages:
            # Main message
            doc_id = f"{workspace_name}_{channel_id}_{message['message_id']}"
            
            document = {
                "id": doc_id,
                "content": message.get('text', ''),
                "workspace": workspace_name,
                "channel": channel_name,
                "channel_id": channel_id,
                "message_id": message['message_id'],
                "timestamp": message.get('timestamp', ''),
                "type": "message"
            }
            
            yield document
            
            # Thread replies
            for reply in message.get('thread', []):
                reply_doc_id = f"{workspace_name}_{channel_id}_{reply['message_id']}"
                
                reply_document = {
                    "id": reply_doc_id,
                    "content": reply.get('text', ''),
                    "workspace": workspace_name,
                    "channel": channel_name,
                    "channel_id": channel_id,
                    "message_id": reply['message_id'],
                    "timestamp": reply.get('timestamp', ''),
                    "parent_message_id": message['message_id'],
                    "type": "reply"
                }
                
                yield reply_document
                
    except Exception as e:
        print(f"Warning: Error parsing {file_path}: {e}")

def parse_raw_format(file_path: str, workspace_name: str, channel_name: str) -> Iterator[Dict[str, Any]]:
    """Parse raw channels format and yield documents"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle channel info
        channel_info = data.get('channel_info', {})
        channel_id = channel_info.get('id', channel_name)
        
        # Handle messages
        messages = data.get('messages', [])
        
        for message in messages:
            # Skip deleted messages
            if message.get('text') == 'This message was deleted.':
                continue
                
            # Create document ID
            ts = message.get('ts', str(uuid.uuid4()))
            doc_id = f"{workspace_name}_{channel_id}_{ts}"
            
            # Extract text content
            text = message.get('text', '')
            
            # If no text but has blocks, try to extract from blocks
            if not text and 'blocks' in message:
                extracted_text = extract_text_from_blocks(message['blocks'])
                if extracted_text:
                    text = extracted_text
            
            if text:  # Only index messages with text content
                document = {
                    "id": doc_id,
                    "content": text,
                    "workspace": workspace_name,
                    "channel": channel_name,
                    "channel_id": channel_id,
                    "message_id": ts,
                    "timestamp": ts,
                    "user": message.get('user', ''),
                    "type": "message"
                }
                
                yield document
                
    except Exception as e:
        print(f"Warning: Error parsing {file_path}: {e}")

def extract_text_from_blocks(blocks: List[Dict]) -> str:
    """Extract text content from Slack blocks"""
    text_parts = []
    
    for block in blocks:
        if block.get('type') == 'rich_text':
            elements = block.get('elements', [])
            for element in elements:
                if element.get('type') == 'rich_text_section':
                    sub_elements = element.get('elements', [])
                    for sub_element in sub_elements:
                        if sub_element.get('type') == 'text':
                            text_parts.append(sub_element.get('text', ''))
    
    return ' '.join(text_parts).strip()

def get_workspace_documents(workspace_filter: str = None) -> Iterator[Dict[str, Any]]:
    """Iterate through Slack archive data and yield documents"""
    
    if not os.path.exists(SLACK_ARCHIVE_DIR):
        print(f"❌ Slack archive directory not found: {SLACK_ARCHIVE_DIR}")
        return
    
    # Find matching workspace directory
    workspace_dirs = []
    for workspace_dir in os.listdir(SLACK_ARCHIVE_DIR):
        workspace_path = os.path.join(SLACK_ARCHIVE_DIR, workspace_dir)
        
        if not os.path.isdir(workspace_path):
            continue
            
        workspace_name = workspace_dir.split('_')[0]  # Extract workspace name before ID
        
        # Filter by workspace if specified
        if workspace_filter and workspace_name.lower() != workspace_filter.lower():
            continue
            
        workspace_dirs.append((workspace_dir, workspace_name, workspace_path))
    
    if not workspace_dirs:
        print(f"❌ No workspaces found matching filter: {workspace_filter}")
        return
    
    # Process matching workspaces
    for workspace_dir, workspace_name, workspace_path in workspace_dirs:
        print(f"Processing workspace: {workspace_name}")
        
        # Check for channels-clean first (preferred format)
        clean_channels_dir = os.path.join(workspace_path, "channels-clean")
        raw_channels_dir = os.path.join(workspace_path, "channels")
        
        if os.path.exists(clean_channels_dir):
            # Use clean format
            for channel_file in os.listdir(clean_channels_dir):
                if channel_file.endswith('.json'):
                    channel_name = channel_file[:-5]  # Remove .json
                    file_path = os.path.join(clean_channels_dir, channel_file)
                    
                    yield from parse_clean_format(file_path, workspace_name, channel_name)
                    
        elif os.path.exists(raw_channels_dir):
            # Use raw format
            for channel_file in os.listdir(raw_channels_dir):
                if channel_file.endswith('.json'):
                    channel_name = channel_file[:-5]  # Remove .json
                    file_path = os.path.join(raw_channels_dir, channel_file)
                    
                    yield from parse_raw_format(file_path, workspace_name, channel_name)

def index_documents(documents: List[Dict[str, Any]], collection: str) -> bool:
    """Index a batch of documents to Solr"""
    try:
        url = f"{SOLR_URL}/{collection}/update/json/docs"
        
        response = requests.post(
            url,
            json=documents,
            headers={"Content-Type": "application/json"},
            params={"commit": "true"}
        )
        
        if response.status_code == 200:
            return True
        else:
            print(f"❌ Indexing failed: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error indexing documents: {e}")
        return False

def count_total_documents(workspace_filter: str = None) -> int:
    """Count total documents for progress tracking"""
    total = 0
    
    for workspace_dir in os.listdir(SLACK_ARCHIVE_DIR):
        workspace_path = os.path.join(SLACK_ARCHIVE_DIR, workspace_dir)
        
        if not os.path.isdir(workspace_path):
            continue
            
        workspace_name = workspace_dir.split('_')[0]
        
        # Filter by workspace if specified
        if workspace_filter and workspace_name.lower() != workspace_filter.lower():
            continue
            
        # Check for channels-clean first
        clean_channels_dir = os.path.join(workspace_path, "channels-clean")
        raw_channels_dir = os.path.join(workspace_path, "channels")
        
        channels_dir = clean_channels_dir if os.path.exists(clean_channels_dir) else raw_channels_dir
        
        if os.path.exists(channels_dir):
            for channel_file in os.listdir(channels_dir):
                if channel_file.endswith('.json'):
                    try:
                        file_path = os.path.join(channels_dir, channel_file)
                        with open(file_path, 'r') as f:
                            data = json.load(f)
                            
                        if 'messages' in data:
                            # Clean format
                            for message in data['messages']:
                                total += 1 + len(message.get('thread', []))
                        else:
                            # Raw format
                            total += len(data.get('messages', []))
                    except:
                        continue
    
    return total

def setup_slack_collection(workspace_filter: str = None):
    """Main function to setup Slack collection and index data"""
    print("🔧 SETTING UP SLACK COLLECTION")
    print("=" * 50)
    
    if workspace_filter:
        print(f"Filtering to workspace: {workspace_filter}")
    
    # Check if core already exists
    if check_core_exists(COLLECTION):
        print(f"Core '{COLLECTION}' already exists")
    else:
        print(f"❌ Core '{COLLECTION}' does not exist. Please create it first.")
        return
    
    print("Loading Slack documents...")
    
    # Count total for progress
    print("Counting documents...")
    total_docs = count_total_documents(workspace_filter)
    print(f"Found approximately {total_docs:,} documents to index")
    
    # Index documents in batches
    batch_size = 1000
    batch = []
    indexed_count = 0
    
    with tqdm(total=total_docs, desc="Indexing documents") as pbar:
        for document in get_workspace_documents(workspace_filter):
            batch.append(document)
            
            if len(batch) >= batch_size:
                if index_documents(batch, COLLECTION):
                    indexed_count += len(batch)
                    pbar.update(len(batch))
                else:
                    print(f"Failed to index batch of {len(batch)} documents")
                
                batch = []
                time.sleep(0.1)  # Small delay to avoid overwhelming Solr
        
        # Index remaining documents
        if batch:
            if index_documents(batch, COLLECTION):
                indexed_count += len(batch)
                pbar.update(len(batch))
    
    print(f"\n✅ Indexing completed!")
    print(f"Total documents indexed: {indexed_count:,}")
    
    # Verify indexing
    try:
        response = requests.get(f"{SOLR_URL}/{COLLECTION}/select?q=*:*&rows=0")
        if response.status_code == 200:
            result = response.json()
            doc_count = result['response']['numFound']
            print(f"Verification: {doc_count:,} documents in collection")
        else:
            print("Could not verify document count")
    except Exception as e:
        print(f"Could not verify indexing: {e}")

if __name__ == "__main__":
    import sys
    workspace_filter = sys.argv[1] if len(sys.argv) > 1 else None
    setup_slack_collection(workspace_filter)