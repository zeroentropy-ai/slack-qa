import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional


def generate_deterministic_uuid(text: str, timestamp: float) -> str:
    """Generate a deterministic UUID from text and timestamp."""
    combined = f"{text}:{timestamp}"
    namespace = uuid.NAMESPACE_DNS # arbitrary namespace
    return str(uuid.uuid5(namespace, combined))


def process_message(message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Process a single message and extract relevant fields."""
    # Get text field
    text = message.get("text")
    if text is None:
        return None
    
    # Get timestamp and generate message_id
    # client_msg_id = message.get("client_msg_id")
    ts = message.get("ts")
    message_id = generate_deterministic_uuid(text, float(ts))
  
    processed = {
        "message_id": message_id,
        "timestamp": ts,
        "text": text
    }
    
    # Process thread_replies if they exist
    if "thread_replies" in message and isinstance(message["thread_replies"], list):
        thread_replies = message["thread_replies"]
        # Skip the first element (identical to parent)
        thread_replies = thread_replies[1:]
        
        processed_replies = []
        for reply in thread_replies:
            processed_reply = process_message(reply)
            if processed_reply is not None:
                processed_replies.append(processed_reply)
        
        # Only add thread_replies if there are any after processing
        if processed_replies:
            processed["thread"] = processed_replies
    
    return processed


def process_json_file(input_path: Path) -> Optional[Dict[str, Any]]:
    """Process a single JSON file and return cleaned data."""
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract channel_id
        channel_id = None
        if "channel_info" in data and isinstance(data["channel_info"], dict):
            channel_id = data["channel_info"].get("id")
        
        # Process messages
        processed_messages = []
        if "messages" in data and isinstance(data["messages"], list):
            for message in data["messages"]:
                processed_msg = process_message(message)
                if processed_msg is not None:
                    processed_messages.append(processed_msg)
        
        # Build output structure
        output = {
            "channel_id": channel_id,
            "messages": processed_messages
        }
        
        return output
        
    except Exception as e:
        print(f"Error processing {input_path}: {e}", file=sys.stderr)
        return None


def main():
    if len(sys.argv) != 2:
        print("Usage: python script.py <directory>", file=sys.stderr)
        sys.exit(1)
    
    input_dir = Path(sys.argv[1])
    
    if not input_dir.exists():
        print(f"Error: Directory '{input_dir}' does not exist", file=sys.stderr)
        sys.exit(1)
    
    if not input_dir.is_dir():
        print(f"Error: '{input_dir}' is not a directory", file=sys.stderr)
        sys.exit(1)
    
    # Create output directory
    output_dir = Path(f"{input_dir}-clean")
    output_dir.mkdir(exist_ok=True)
    
    # Process all JSON files
    json_files = list(input_dir.glob("*.json"))
    
    if not json_files:
        print(f"No JSON files found in '{input_dir}'", file=sys.stderr)
        sys.exit(1)
    
    print(f"Processing {len(json_files)} JSON file(s)...")
    
    for json_file in json_files:
        print(f"Processing: {json_file.name}")
        
        processed_data = process_json_file(json_file)
        
        if processed_data is not None:
            output_path = output_dir / json_file.name
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(processed_data, f, indent=2, ensure_ascii=False)
            print(f"  ✓ Saved to: {output_path}")
        else:
            print(f"  ✗ Failed to process")
    
    print(f"\nDone! Processed files saved to: {output_dir}")


if __name__ == "__main__":
    main()
