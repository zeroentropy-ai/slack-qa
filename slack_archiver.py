import requests
import json
import time
import os
from datetime import datetime

class SlackArchiver:
    def __init__(self, token):
        self.token = token
        self.headers = {"Authorization": f"Bearer {token}"}
        self.base_url = "https://slack.com/api"
    
    def _api_call(self, endpoint, params=None):
        """Make API call with rate limiting"""
        response = requests.get(
            f"{self.base_url}/{endpoint}",
            headers=self.headers,
            params=params or {}
        )
        data = response.json()
        
        if not data.get('ok'):
            print(f"❌ Error on {endpoint}: {data.get('error')}")
            return data
        
        time.sleep(0.5)  # Rate limiting
        return data
    
    def get_users(self):
        """Get all users"""
        print("📥 Fetching users...")
        return self._api_call('users.list')
    
    def get_channels(self):
        """Get all channels (public, private, DMs)"""
        print("📥 Fetching channels...")
        channels = []
        cursor = None
        
        params = {
            'types': 'public_channel,private_channel,mpim,im',
            'limit': 200
        }
        
        while True:
            if cursor:
                params['cursor'] = cursor
            
            data = self._api_call('conversations.list', params)
            if data.get('ok'):
                channels.extend(data.get('channels', []))
            
            cursor = data.get('response_metadata', {}).get('next_cursor')
            if not cursor:
                break
        
        return channels
    
    def get_channel_history(self, channel_id):
        """Get all messages from a channel"""
        messages = []
        cursor = None
        
        while True:
            params = {
                'channel': channel_id,
                'limit': 200
            }
            if cursor:
                params['cursor'] = cursor
            
            data = self._api_call('conversations.history', params)
            if data.get('ok'):
                messages.extend(data.get('messages', []))
            
            cursor = data.get('response_metadata', {}).get('next_cursor')
            if not cursor:
                break
        
        return messages
    
    def get_thread_replies(self, channel_id, thread_ts):
        """Get all replies in a thread"""
        params = {
            'channel': channel_id,
            'ts': thread_ts
        }
        data = self._api_call('conversations.replies', params)
        return data.get('messages', []) if data.get('ok') else []
    
    def archive_workspace(self, output_dir='slack_archive'):
        """Archive entire workspace"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save metadata
        timestamp = datetime.now().isoformat()
        metadata = {
            'archived_at': timestamp,
            'workspace': 'ZeroEntropy Community',
            'user': 'xalejo1999'
        }
        
        # 1. Get and save users
        users_data = self.get_users()
        with open(f'{output_dir}/users.json', 'w', encoding='utf-8') as f:
            json.dump(users_data, f, indent=2, ensure_ascii=False)
        print(f"✅ Saved {len(users_data.get('members', []))} users")
        
        # 2. Get and save channels
        channels = self.get_channels()
        with open(f'{output_dir}/channels.json', 'w', encoding='utf-8') as f:
            json.dump(channels, f, indent=2, ensure_ascii=False)
        print(f"✅ Found {len(channels)} channels")
        
        # 3. Get messages for each channel
        channel_dir = f'{output_dir}/channels'
        os.makedirs(channel_dir, exist_ok=True)
        
        for i, channel in enumerate(channels, 1):
            channel_id = channel['id']
            channel_name = channel.get('name', channel_id)
            
            # Handle DMs (they don't have names)
            if channel.get('is_im'):
                channel_name = f"dm_{channel_id}"
            elif channel.get('is_mpim'):
                channel_name = f"group_dm_{channel_id}"
            
            print(f"[{i}/{len(channels)}] 💬 Fetching #{channel_name}...")
            
            messages = self.get_channel_history(channel_id)
            
            # Get thread replies
            threads_fetched = 0
            for message in messages:
                if message.get('thread_ts') and message.get('reply_count', 0) > 0:
                    # Only fetch if this is the parent message
                    if message.get('ts') == message.get('thread_ts'):
                        replies = self.get_thread_replies(channel_id, message['thread_ts'])
                        message['thread_replies'] = replies
                        threads_fetched += 1
            
            # Save channel data
            channel_data = {
                'channel_info': channel,
                'message_count': len(messages),
                'messages': messages
            }
            
            safe_name = channel_name.replace('/', '_').replace('\\', '_')
            with open(f'{channel_dir}/{safe_name}.json', 'w', encoding='utf-8') as f:
                json.dump(channel_data, f, indent=2, ensure_ascii=False)
            
            print(f"   ✅ {len(messages)} messages, {threads_fetched} threads")
        
        # Save metadata
        with open(f'{output_dir}/metadata.json', 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n🎉 Archive complete! Saved to '{output_dir}/'")
        print(f"📊 Total: {len(channels)} channels archived")

# Usage
TOKEN = "xoxp-7878678554402-9489359346982-9626258575062-38d6e8a9ad40801d66a86792dc450868"

archiver = SlackArchiver(TOKEN)
archiver.archive_workspace()