import requests
import json
import time
import os
from datetime import datetime
from urllib.parse import urlparse

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
        
        time.sleep(0.5)
        return data
    
    def download_file(self, url, filepath):
        """Download a file from Slack"""
        try:
            response = requests.get(url, headers=self.headers, stream=True)
            if response.status_code == 200:
                os.makedirs(os.path.dirname(filepath), exist_ok=True)
                with open(filepath, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                return True
        except Exception as e:
            print(f"   ⚠️  Failed to download: {e}")
        return False
    
    def get_users(self):
        """Get all users"""
        print("📥 Fetching users...")
        return self._api_call('users.list')
    
    def get_team_info(self):
        """Get workspace info"""
        print("📥 Fetching team info...")
        return self._api_call('team.info')
    
    def get_emoji(self):
        """Get custom emoji"""
        print("📥 Fetching custom emoji...")
        return self._api_call('emoji.list')
    
    def get_channels(self):
        """Get all channels"""
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
    
    def get_channel_members(self, channel_id):
        """Get members of a channel"""
        members = []
        cursor = None
        
        while True:
            params = {'channel': channel_id, 'limit': 200}
            if cursor:
                params['cursor'] = cursor
            
            data = self._api_call('conversations.members', params)
            if data.get('ok'):
                members.extend(data.get('members', []))
            
            cursor = data.get('response_metadata', {}).get('next_cursor')
            if not cursor:
                break
        
        return members
    
    def get_pinned_items(self, channel_id):
        """Get pinned messages in a channel"""
        data = self._api_call('pins.list', {'channel': channel_id})
        return data.get('items', []) if data.get('ok') else []
    
    def get_bookmarks(self, channel_id):
        """Get channel bookmarks"""
        data = self._api_call('bookmarks.list', {'channel_id': channel_id})
        return data.get('bookmarks', []) if data.get('ok') else []
    
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
    
    def extract_files_from_messages(self, messages):
        """Extract all file URLs from messages"""
        files = []
        for message in messages:
            if 'files' in message:
                for file in message['files']:
                    files.append({
                        'id': file.get('id'),
                        'name': file.get('name'),
                        'url': file.get('url_private'),
                        'url_download': file.get('url_private_download'),
                        'mimetype': file.get('mimetype'),
                        'size': file.get('size'),
                        'timestamp': message.get('ts')
                    })
        return files
    
    def archive_workspace(self, output_dir='slack_archive', download_files=True):
        """Archive entire workspace"""
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().isoformat()
        
        # 1. Team info
        team_info = self.get_team_info()
        with open(f'{output_dir}/team_info.json', 'w', encoding='utf-8') as f:
            json.dump(team_info, f, indent=2, ensure_ascii=False)
        
        # 2. Users
        users_data = self.get_users()
        with open(f'{output_dir}/users.json', 'w', encoding='utf-8') as f:
            json.dump(users_data, f, indent=2, ensure_ascii=False)
        print(f"✅ Saved {len(users_data.get('members', []))} users")
        
        # 3. Custom emoji
        emoji_data = self.get_emoji()
        with open(f'{output_dir}/emoji.json', 'w', encoding='utf-8') as f:
            json.dump(emoji_data, f, indent=2, ensure_ascii=False)
        if emoji_data.get('ok'):
            print(f"✅ Saved {len(emoji_data.get('emoji', {}))} custom emoji")
        
        # 4. Channels
        channels = self.get_channels()
        with open(f'{output_dir}/channels.json', 'w', encoding='utf-8') as f:
            json.dump(channels, f, indent=2, ensure_ascii=False)
        print(f"✅ Found {len(channels)} channels")
        
        # 5. Channel messages and metadata
        channel_dir = f'{output_dir}/channels'
        files_dir = f'{output_dir}/files'
        os.makedirs(channel_dir, exist_ok=True)
        
        all_files = []
        
        for i, channel in enumerate(channels, 1):
            channel_id = channel['id']
            channel_name = channel.get('name', channel_id)
            
            if channel.get('is_im'):
                channel_name = f"dm_{channel_id}"
            elif channel.get('is_mpim'):
                channel_name = f"group_dm_{channel_id}"
            
            print(f"\n[{i}/{len(channels)}] 💬 Processing #{channel_name}...")
            
            # Get members
            members = self.get_channel_members(channel_id)
            print(f"   👥 {len(members)} members")
            
            # Get pinned items
            pinned = self.get_pinned_items(channel_id)
            if pinned:
                print(f"   📌 {len(pinned)} pinned items")
            
            # Get bookmarks
            bookmarks = self.get_bookmarks(channel_id)
            if bookmarks:
                print(f"   🔖 {len(bookmarks)} bookmarks")
            
            # Get messages
            messages = self.get_channel_history(channel_id)
            
            # Get thread replies
            threads_fetched = 0
            for message in messages:
                if message.get('thread_ts') and message.get('reply_count', 0) > 0:
                    if message.get('ts') == message.get('thread_ts'):
                        replies = self.get_thread_replies(channel_id, message['thread_ts'])
                        message['thread_replies'] = replies
                        threads_fetched += 1
            
            # Extract files
            channel_files = self.extract_files_from_messages(messages)
            all_files.extend([(channel_name, f) for f in channel_files])
            
            # Save channel data
            channel_data = {
                'channel_info': channel,
                'members': members,
                'pinned_items': pinned,
                'bookmarks': bookmarks,
                'message_count': len(messages),
                'messages': messages,
                'files': channel_files
            }
            
            safe_name = channel_name.replace('/', '_').replace('\\', '_')
            with open(f'{channel_dir}/{safe_name}.json', 'w', encoding='utf-8') as f:
                json.dump(channel_data, f, indent=2, ensure_ascii=False)
            
            print(f"   ✅ {len(messages)} messages, {threads_fetched} threads, {len(channel_files)} files")
        
        # 6. Download files
        if download_files and all_files:
            print(f"\n📦 Downloading {len(all_files)} files...")
            os.makedirs(files_dir, exist_ok=True)
            
            for idx, (channel_name, file_info) in enumerate(all_files, 1):
                if file_info.get('url_download'):
                    safe_channel = channel_name.replace('/', '_').replace('\\', '_')
                    safe_filename = file_info['name'].replace('/', '_').replace('\\', '_')
                    filepath = f"{files_dir}/{safe_channel}/{safe_filename}"
                    
                    print(f"   [{idx}/{len(all_files)}] {safe_filename}...", end='')
                    if self.download_file(file_info['url_download'], filepath):
                        print(" ✅")
                    else:
                        print(" ❌")
        
        # Save metadata
        metadata = {
            'archived_at': timestamp,
            'workspace': team_info.get('team', {}).get('name', 'Unknown'),
            'total_channels': len(channels),
            'total_files': len(all_files)
        }
        with open(f'{output_dir}/metadata.json', 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n🎉 Archive complete! Saved to '{output_dir}/'")
        print(f"📊 Summary:")
        print(f"   - {len(channels)} channels")
        print(f"   - {len(all_files)} files")

# Usage
TOKEN = "xoxp-7878678554402-9489359346982-9626258575062-38d6e8a9ad40801d66a86792dc450868"

archiver = SlackArchiver(TOKEN)
archiver.archive_workspace(download_files=True)  # Set to False to skip file downloads