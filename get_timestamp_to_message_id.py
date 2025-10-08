import json
import os

# Path to your Slack archive folder
base_dir = "slack_archive/Modal_Community_T031JJZ7Q6T/channels-clean"

# Iterate through all files in the folder
timestamp_to_message_id = {}
for filename in os.listdir(base_dir):
    if filename.endswith(".json"):
        file_path = os.path.join(base_dir, filename)

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                all_messages = data["messages"]
                for message in all_messages[:]:
                    for inner_message in message.get("thread", []):
                        all_messages.append(inner_message)

                for message in all_messages:
                    assert message["timestamp"] not in timestamp_to_message_id or timestamp_to_message_id[message["timestamp"]] == message["message_id"]
                    timestamp_to_message_id[message["timestamp"]] = message["message_id"]
        except json.JSONDecodeError as e:
            print(f"⚠️ Error decoding {filename}: {e}")
        except Exception as e:
            print(f"❌ Failed to process {filename}: {repr(e)}")

with open(f"timestamp_to_message_id.json", "w") as f:
    f.write(json.dumps(timestamp_to_message_id, indent=4))
print(f"Len: {len(timestamp_to_message_id)}")
