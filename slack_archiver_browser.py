import requests
import json
import time
import os
from datetime import datetime
from urllib.parse import urlparse
from requests_toolbelt.multipart.encoder import MultipartEncoder

class AuthenticationError(Exception):
    """Raised when authentication fails"""
    pass

class SlackArchiver:
    # Global counters
    api_call_counter = 0
    api_call_log = []
    
    # Retry settings
    MAX_RETRIES = 3
    BASE_BACKOFF = 2  # seconds
    
    def __init__(self, token, auth_mode='app', cookies=None, workspace_url=None, save_channel_members=True):
        """
        Initialize archiver with authentication mode
        
        Args:
            token: Either xoxp (app) token or xoxc (browser) token
            auth_mode: 'app' for OAuth app tokens, 'browser' for session tokens
            cookies: Full cookie string from browser (required for browser mode)
            workspace_url: Workspace URL like 'https://modallabscommunity.slack.com' (required for browser mode)
            save_channel_members: Whether to fetch and save channel member lists (default: True)
        """
        self.token = token
        self.auth_mode = auth_mode
        self.cookies = cookies
        self.save_channel_members = save_channel_members
        
        if auth_mode == 'app':
            self.headers = {"Authorization": f"Bearer {token}"}
            self.base_url = "https://slack.com/api"
        elif auth_mode == 'browser':
            if not cookies or not workspace_url:
                raise ValueError("Browser mode requires both cookies and workspace_url")
            self.base_url = f"{workspace_url}/api"
            # Extract xoxd token from cookies if present
            self.xoxd_token = self._extract_xoxd_from_cookies(cookies)
        else:
            raise ValueError("auth_mode must be 'app' or 'browser'")
    
    def _extract_xoxd_from_cookies(self, cookies):
        """Extract xoxd token from cookie string"""
        for cookie in cookies.split('; '):
            if cookie.startswith('d='):
                # URL decode the xoxd token
                import urllib.parse
                return urllib.parse.unquote(cookie[2:])
        return None
    
    def _log_api_call(self, endpoint, params=None, retry_attempt=0):
        """Log each API call"""
        SlackArchiver.api_call_counter += 1
        retry_str = f" (retry {retry_attempt}/{self.MAX_RETRIES})" if retry_attempt > 0 else ""
        log_msg = f"🔌 API Call #{SlackArchiver.api_call_counter}: {endpoint}{retry_str}"
        if params:
            # Log key params (not full data for brevity)
            key_params = {k: v for k, v in params.items() if k in ['channel', 'ts', 'cursor', 'limit']}
            if key_params:
                log_msg += f" | params: {key_params}"
        print(log_msg)
        
        SlackArchiver.api_call_log.append({
            'call_number': SlackArchiver.api_call_counter,
            'endpoint': endpoint,
            'params': params,
            'timestamp': datetime.now().isoformat(),
            'retry_attempt': retry_attempt
        })
    
    def _api_call(self, endpoint, params=None, method='GET'):
        """Make API call with appropriate authentication and retry logic"""
        for attempt in range(self.MAX_RETRIES + 1):
            self._log_api_call(endpoint, params, retry_attempt=attempt)
            
            if self.auth_mode == 'app':
                response, data = self._api_call_app(endpoint, params, method)
            else:
                response, data = self._api_call_browser(endpoint, params, method)
            
            # Check if successful
            if data.get('ok'):
                time.sleep(0.5)  # Rate limit protection
                return data
            
            # Handle errors
            error = data.get('error', 'unknown_error')
            
            # IMMEDIATE STOP on invalid_auth
            if error == 'invalid_auth':
                print(f"\n🚨 AUTHENTICATION FAILED - Auth tokens have been revoked!")
                print(f"🛑 Stopping immediately to preserve progress...")
                raise AuthenticationError(f"Authentication failed on {endpoint}")
            
            if error == 'ratelimited':
                # Get retry-after header or use exponential backoff
                retry_after = int(response.headers.get('Retry-After', self.BASE_BACKOFF * (2 ** attempt)))
                print(f"   ⏸️  Rate limited! Waiting {retry_after}s before retry...")
                time.sleep(retry_after)
                continue
            
            elif attempt < self.MAX_RETRIES:
                # Exponential backoff for other errors
                backoff = self.BASE_BACKOFF * (2 ** attempt)
                print(f"   ⚠️  Error '{error}' - retrying in {backoff}s...")
                time.sleep(backoff)
                continue
            else:
                # Max retries reached
                print(f"   ❌ Failed after {self.MAX_RETRIES} retries: {error}")
                return data
        
        return {'ok': False, 'error': 'max_retries_exceeded'}
    
    def _api_call_app(self, endpoint, params=None, method='GET'):
        """Make API call with OAuth app token"""
        if method == 'GET':
            response = requests.get(
                f"{self.base_url}/{endpoint}",
                headers=self.headers,
                params=params or {}
            )
        else:
            response = requests.post(
                f"{self.base_url}/{endpoint}",
                headers=self.headers,
                data=params or {}
            )
        
        data = response.json()
        return response, data
    
    def _api_call_browser(self, endpoint, params=None, method='GET'):
        """Make API call with browser session tokens"""
        # Prepare multipart form data
        fields = {'token': self.token}
        if params:
            fields.update(params)
        
        multipart_data = MultipartEncoder(fields=fields)
        
        headers = {
            "Accept": "*/*",
            "Accept-Encoding": "gzip, deflate, br",
            "Accept-Language": "en-US,en;q=0.9",
            "Cookie": self.cookies,
            "Content-Type": multipart_data.content_type,
            "Origin": "https://app.slack.com",
            "Referer": "https://app.slack.com/",
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.5 Safari/605.1.15",
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-site",
        }
        
        # Query parameters
        query_params = {
            "_x_id": f"archive-{int(time.time() * 1000)}",
            "_x_version_ts": str(int(time.time())),
            "_x_frontend_build_type": "current",
            "_x_desktop_ia": "4",
            "_x_gantry": "true",
            "fp": "a2"
        }
        
        response = requests.post(
            f"{self.base_url}/{endpoint}",
            headers=headers,
            params=query_params,
            data=multipart_data
        )
        
        data = response.json()
        return response, data
    
    def download_file(self, url, filepath):
        """Download a file from Slack"""
        try:
            if self.auth_mode == 'app':
                headers = self.headers
            else:
                headers = {
                    "Cookie": self.cookies,
                    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.5 Safari/605.1.15"
                }
            
            response = requests.get(url, headers=headers, stream=True)
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
        print("\n📥 Fetching users...")
        users = []
        cursor = None
        params = {'limit': '200'}
        
        while True:
            if cursor:
                params['cursor'] = cursor
            
            data = self._api_call('users.list', params)
            if data.get('ok'):
                users.extend(data.get('members', []))
            else:
                break
            
            cursor = data.get('response_metadata', {}).get('next_cursor')
            if not cursor:
                break
        
        return {'ok': True, 'members': users}

    def get_team_info(self):
        """Get workspace info"""
        print("\n📥 Fetching team info...")
        return self._api_call('team.info')
    
    def get_emoji(self):
        """Get custom emoji"""
        print("\n📥 Fetching custom emoji...")
        return self._api_call('emoji.list')
    
    def get_channels(self):
        """Get all channels"""
        print("\n📥 Fetching channels...")
        channels = []
        cursor = None
        
        params = {
            'types': 'public_channel,private_channel,mpim,im',
            'limit': '200'
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
            params = {'channel': channel_id, 'limit': '200'}
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
                'limit': '200'
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
    
    def _get_existing_users(self, workspace_dir):
        """Check if users.json already exists"""
        users_file = f'{workspace_dir}/users.json'
        if os.path.exists(users_file):
            try:
                with open(users_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return data
            except Exception as e:
                print(f"   ⚠️  Error reading existing users.json: {e}")
                return None
        return None

    def _get_existing_channels(self, channel_dir):
        """Get list of already archived channel files"""
        if not os.path.exists(channel_dir):
            return set()
        
        existing = set()
        for filename in os.listdir(channel_dir):
            if filename.endswith('.json'):
                # Remove .json extension to get channel name
                channel_name = filename[:-5]
                existing.add(channel_name)
        
        return existing

    def archive_workspace(self, output_dir='slack_archive', download_files=True):
        """Archive entire workspace with resume capability"""
        
        print(f"🔐 Authentication mode: {self.auth_mode}")
        print(f"🔄 Max retries: {self.MAX_RETRIES}")
        print(f"⏱️  Base backoff: {self.BASE_BACKOFF}s")
        print(f"👥 Save channel members: {self.save_channel_members}")

        try:
            # Get team info first to determine workspace name
            team_info = self.get_team_info()
            workspace_name = team_info.get('team', {}).get('name', 'unknown_workspace')
            workspace_id = team_info.get('team', {}).get('id', 'unknown_id')
            
            # Create workspace-specific subdirectory
            safe_workspace_name = workspace_name.replace('/', '_').replace('\\', '_').replace(' ', '_')
            workspace_dir = f"{output_dir}/{safe_workspace_name}_{workspace_id}"
            
            os.makedirs(workspace_dir, exist_ok=True)
            print(f"📁 Saving to: {workspace_dir}")
            
            timestamp = datetime.now().isoformat()
            
            # Check what's already been archived
            channel_dir = f'{workspace_dir}/channels'
            os.makedirs(channel_dir, exist_ok=True)
            existing_channels = self._get_existing_channels(channel_dir)
            
            if existing_channels:
                print(f"♻️  Found {len(existing_channels)} already archived channels - will skip them")
            
            # 1. Team info (always refresh)
            with open(f'{workspace_dir}/team_info.json', 'w', encoding='utf-8') as f:
                json.dump(team_info, f, indent=2, ensure_ascii=False)
            
            # 2. Users (always refresh)
            existing_users = self._get_existing_users(workspace_dir)
            if existing_users:
                users_data = existing_users
                user_count = len(users_data.get('members', []))
                print(f"♻️  Using existing users.json ({user_count} users)")
            else:
                print("\n📥 No existing users.json found - fetching users...")
                users_data = self.get_users()
                with open(f'{workspace_dir}/users.json', 'w', encoding='utf-8') as f:
                    json.dump(users_data, f, indent=2, ensure_ascii=False)
                print(f"✅ Saved {len(users_data.get('members', []))} users")
            
            
            # 3. Custom emoji (always refresh)
            emoji_data = self.get_emoji()
            with open(f'{workspace_dir}/emoji.json', 'w', encoding='utf-8') as f:
                json.dump(emoji_data, f, indent=2, ensure_ascii=False)
            if emoji_data.get('ok'):
                print(f"✅ Saved {len(emoji_data.get('emoji', {}))} custom emoji")
            
            # 4. Channels (always refresh the list)
            channels = self.get_channels()
            with open(f'{workspace_dir}/channels.json', 'w', encoding='utf-8') as f:
                json.dump(channels, f, indent=2, ensure_ascii=False)
            print(f"✅ Found {len(channels)} channels")
            
            # 5. Channel messages and metadata
            files_dir = f'{workspace_dir}/files'
            
            all_files = []
            channels_to_process = []
            
            # Filter out already archived channels
            for channel in channels:
                channel_id = channel['id']
                channel_name = channel.get('name', channel_id)
                
                if channel.get('is_im'):
                    channel_name = f"dm_{channel_id}"
                elif channel.get('is_mpim'):
                    channel_name = f"group_dm_{channel_id}"
                
                safe_name = channel_name.replace('/', '_').replace('\\', '_')
                
                if safe_name in existing_channels:
                    print(f"⏭️  Skipping #{channel_name} (already archived)")
                else:
                    channels_to_process.append((channel, channel_name, safe_name))
            
            print(f"\n📋 Processing {len(channels_to_process)} channels ({len(existing_channels)} already done)")
            
            for i, (channel, channel_name, safe_name) in enumerate(channels_to_process, 1):
                channel_id = channel['id']
                
                print(f"\n{'='*60}")
                print(f"[{i}/{len(channels_to_process)}] 💬 Processing #{channel_name}...")
                print(f"{'='*60}")
                
                # Get members
                members = []
                if self.save_channel_members:
                    members = self.get_channel_members(channel_id)
                    print(f"   👥 {len(members)} members")
                else:
                    print(f"   ⏭️  Skipping member list (disabled)")
                
                # Get pinned items
                pinned = self.get_pinned_items(channel_id)
                if pinned:
                    print(f"   📌 {len(pinned)} pinned items")
                
                # Get bookmarks
                bookmarks = self.get_bookmarks(channel_id)
                if bookmarks:
                    print(f"   🔖 {len(bookmarks)} bookmarks")
                
                # Get messages
                print(f"   📨 Fetching message history...")
                messages = self.get_channel_history(channel_id)
                print(f"   ✅ Retrieved {len(messages)} messages")
                
                # Get thread replies
                thread_count = sum(1 for m in messages if m.get('thread_ts') and m.get('reply_count', 0) > 0 and m.get('ts') == m.get('thread_ts'))
                if thread_count > 0:
                    print(f"   🧵 Fetching {thread_count} threads...")
                
                threads_fetched = 0
                for message in messages:
                    if message.get('thread_ts') and message.get('reply_count', 0) > 0:
                        if message.get('ts') == message.get('thread_ts'):
                            replies = self.get_thread_replies(channel_id, message['thread_ts'])
                            message['thread_replies'] = replies
                            threads_fetched += 1
                            if threads_fetched % 10 == 0:
                                print(f"      Progress: {threads_fetched}/{thread_count} threads")
                
                # Extract files
                channel_files = self.extract_files_from_messages(messages)
                all_files.extend([(channel_name, f) for f in channel_files])
                
                # Save channel data (ONLY after all data is successfully fetched)
                channel_data = {
                    'channel_info': channel,
                    'members': members,
                    'pinned_items': pinned,
                    'bookmarks': bookmarks,
                    'message_count': len(messages),
                    'messages': messages,
                    'files': channel_files
                }
                
                with open(f'{channel_dir}/{safe_name}.json', 'w', encoding='utf-8') as f:
                    json.dump(channel_data, f, indent=2, ensure_ascii=False)
                
                print(f"   ✅ Saved: {len(messages)} messages, {threads_fetched} threads, {len(channel_files)} files")
            
            # 6. Download files (if requested)
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
                'auth_mode': self.auth_mode,
                'workspace_name': workspace_name,
                'workspace_id': workspace_id,
                'total_channels': len(channels),
                'total_files': len(all_files),
                'total_api_calls': SlackArchiver.api_call_counter
            }
            with open(f'{workspace_dir}/metadata.json', 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2)
            
            # Save API call log
            with open(f'{workspace_dir}/api_call_log.json', 'w', encoding='utf-8') as f:
                json.dump(SlackArchiver.api_call_log, f, indent=2)
            
            print(f"\n{'='*60}")
            print(f"🎉 Archive complete! Saved to '{workspace_dir}/'")
            print(f"{'='*60}")
            print(f"📊 Summary:")
            print(f"   - Workspace: {workspace_name}")
            print(f"   - Channels: {len(channels)}")
            print(f"   - Files: {len(all_files)}")
            print(f"   - Total API calls: {SlackArchiver.api_call_counter}")
            print(f"   - API call log saved to: api_call_log.json")
        
        except AuthenticationError as e:
            print(f"\n❌ Authentication error: {e}")
            print(f"🔄 Progress has been saved. Re-run with fresh tokens to resume.")
            raise  # Re-raise so the script exits with error code
        
        except KeyboardInterrupt:
            print(f"\n\n⚠️  Interrupted by user")
            print(f"🔄 Progress has been saved. Re-run to resume.")
            raise

# ============================================
# USAGE EXAMPLES
# ============================================

def archive_with_browser_session(xoxc_token, full_cookies, workspace_url):
    archiver = SlackArchiver(
        token=xoxc_token,
        auth_mode='browser',
        cookies=full_cookies,
        workspace_url=workspace_url,
        save_channel_members=False
    )
    archiver.archive_workspace(download_files=True)


if __name__ == "__main__":
    """XOXC_TOKEN = "xoxc-3052645262231-9641512460897-9626513329798-19e797687a5e0bb7539701cd740f4a9b3c98f040ebd6213e7f33577468f85c6d"
    FULL_COOKIES = "utm=%7B%7D; x=f3db5096c114fdcea90c10e9316228dc.1759448167; shown_ssb_redirect_page=1; OptanonConsent=isGpcEnabled=0&datestamp=Thu+Oct+02+2025+16%3A37%3A26+GMT-0700+(Pacific+Daylight+Time)&version=202402.1.0&browserGpcFlag=0&isIABGlobal=false&hosts=&consentId=1ff3be3e-e588-4932-9ac3-2630dd7c33aa&interactionCount=1&isAnonUser=1&landingPath=NotLandingPage&groups=1%3A1%2C3%3A1%2C2%3A1%2C4%3A1&AwaitingReconsent=false; _ga=GA1.1.221978663.1757447861; _ga_QTJQME5M5D=GS2.1.s1759448237$o4$g0$t1759448237$j60$l0$h0; PageCount=37; _cs_id=65bbed2f-e942-a0d8-ff58-7364edf3ae6f.1757447860.3.1759448237.1759448237.1.1791611860287.1.x; _cs_s=1.0.U.9.1759450037198; _li_ss=CkoKBgj5ARDvGwoFCAoQ7xsKBgjdARDvGwoGCOEBEO8bCgYIgQEQ7xsKBgiiARDvGwoJCP____8HEPkbCgYIiQEQ7xsKBgilARDvGw; cjConsent=MHxOfDB8Tnww; cjUser=7ca4c4b8-116d-45aa-ad4c-88a5d183fc3d; ssb_instance_id=b9822ad1-6df9-40d2-8374-d0b286d41559; d=xoxd-mqA7OJJlphk%2F%2FScObW0GkyuXX1uA9bx4okOK5v5k%2BTBnuVQhGvMED1cwV5mFJhAJzTMlL7rkS8aHZ8cLTdRlNLn4vwsnWc8boUtzbctLTUqDXDCBuU838gAWgdNiZwtN4InZpBxIPZVu4O3JSLRdDyiv%2F%2BUsV5KujC3wY56%2Bx4%2FpGgbshsgD6EyFIZMKf4hII4Hin8lhq%2BgzNtFWBXnQVVEHUyI%3D; no_download_ssb_banner=1; show_download_ssb_banner=1; shown_download_ssb_modal=1; lc=1759448208; _cs_cvars=%7B%226%22%3A%5B%22is_paid_plan%22%2C%22false%22%5D%7D; _fbp=fb.1.1759438148806.71444673446120306; optimizelySession=0; agentforce_chatID=; _gcl_au=1.1.536689637.1757447861.707819349.1759440091.1759440092; _lc2_fpi_js=e00b11ac9c9b--01k4r0wbxafwq3ab8a66d6tj9e; _li_dcdm_c=.slack.com; d-s=1758655485; _cs_c=0; _lc2_fpi=e00b11ac9c9b--01k4r0wbxafwq3ab8a66d6tj9e; tz=-420; b=.f3db5096c114fdcea90c10e9316228dc"
    WORKSPACE_URL = "https://modallabscommunity.slack.com"
    modallabs = (XOXC_TOKEN, FULL_COOKIES, WORKSPACE_URL)"""

    XOXC_TOKEN = "xoxc-571236512613-9632331502644-9657115562272-7c9b7cd1a20019191d210075710aed405c57b5996d1ebd189a92250049b2502f"
    FULL_COOKIES = "utm=%7B%7D; x=f3db5096c114fdcea90c10e9316228dc.1759516210; no_download_ssb_banner=1; show_download_ssb_banner=1; shown_download_ssb_modal=1; shown_ssb_redirect_page=1; OptanonConsent=isGpcEnabled=0&datestamp=Fri+Oct+03+2025+12%3A30%3A56+GMT-0700+(Pacific+Daylight+Time)&version=202402.1.0&browserGpcFlag=0&isIABGlobal=false&hosts=&consentId=1ff3be3e-e588-4932-9ac3-2630dd7c33aa&interactionCount=1&isAnonUser=1&landingPath=NotLandingPage&groups=1%3A1%2C3%3A1%2C2%3A1%2C4%3A1&AwaitingReconsent=false; _fbp=fb.1.1759438148806.71444673446120306; _cs_s=3.0.U.9.1759521651893; web_cache_last_updated18543652b4683764baf47d66b363e7c0=1759518395577; d=xoxd-D3ntxMXc27YoFu5E1RThAT7u87Rt3ezKnRKv7U4s%2FTyf14Z4QLQWwavzDIOwEeHN%2Fv0rbpJ%2BjQrpJYY6%2BNkJO%2FqTW3jv%2FVCRlvxw5Q3Ysc5jHt0ktcw0DP3VBiWE6vwQE2Pm8KzZo316PorTcLWbBoT84tjdY4qs32lf0cKYwxEHgPhlCMrcH6wIWcOJHORs0T%2BXQnFFmCIs011DWAsY6eLBjeU%3D; lc=1759518364; _ga_QTJQME5M5D=GS2.1.s1759517027$o6$g1$t1759518055$j60$l0$h0; _ga=GA1.1.221978663.1757447861; PageCount=3; _cs_id=65bbed2f-e942-a0d8-ff58-7364edf3ae6f.1757447860.8.1759518029.1759517027.1.1791611860287.1.x; _li_ss=ClkKBgj5ARDvGwoFCAoQ7xsKBgikARDvGwoGCN0BEO8bCgYI4QEQ7xsKBgiBARDvGwoGCKIBEO8bCgkI_____wcQ-RsKBQh-EPAbCgYIiQEQ7xsKBgilARDvGw; cjConsent=MHxOfDB8Tnww; cjUser=7ca4c4b8-116d-45aa-ad4c-88a5d183fc3d; ec=enQtOTYzODE1OTM3NDU0OC1jNzRiOWUzYmE5YWE4OWQ3YzZjNDc2ODZkYmNlMjNmNGM1YmZiYjVhMmUwYmQ3NDgyYzk5MGVkOTU5M2ZkODA5; _cs_cvars=%7B%225%22%3A%5B%22curr_plan%22%2C%22free%22%5D%2C%226%22%3A%5B%22is_paid_plan%22%2C%22false%22%5D%7D; _lc2_fpi_js=e00b11ac9c9b--01k4r0wbxafwq3ab8a66d6tj9e; _li_dcdm_c=.slack.com; d-s=1759513078; ssb_instance_id=b9822ad1-6df9-40d2-8374-d0b286d41559; optimizelySession=0; agentforce_chatID=; _gcl_au=1.1.536689637.1757447861.707819349.1759440091.1759440092; _cs_c=0; _lc2_fpi=e00b11ac9c9b--01k4r0wbxafwq3ab8a66d6tj9e; tz=-420; b=.f3db5096c114fdcea90c10e9316228dc"
    WORKSPACE_URL = "https://fluttercommunity.slack.com"
    flutter = (XOXC_TOKEN, FULL_COOKIES, WORKSPACE_URL)

    """XOXC_TOKEN = "xoxc-5717910232870-9626607001622-9656962142704-09c2e4bebc622e9a0afb9175bae22daf4400ee834580273a7e747056692f1617"
    FULL_COOKIES = "utm=%7B%7D; x=f3db5096c114fdcea90c10e9316228dc.1759465273; d=xoxd-FkHkozQ8MnhQuM5fpzOHEahusVOcpub4d62vFVeb5WVCk7Nyc0dFRsd5VDSTkKqCdIeV8WDP506mN4BZSycUAA6%2FpNTFNFPgi86%2B8xdDKSqFyh3NcCFhz9z%2BVnGX%2BT40OXaZv2TYNJRkHEacWKKiGmxkt%2FS2AzPVFwtlPox8YcfuhYbTCAf5zm3j7tfRrYrpN92gCsAfE9vWTbyqhWm%2BWdx6UEk%3D; d-s=1759461750; _ga_QTJQME5M5D=GS2.1.s1759460948$o5$g0$t1759460948$j60$l0$h0; _cs_id=65bbed2f-e942-a0d8-ff58-7364edf3ae6f.1757447860.7.1759458693.1759458693.1.1791611860287.1.x; shown_ssb_redirect_page=1; OptanonConsent=isGpcEnabled=0&datestamp=Thu+Oct+02+2025+16%3A37%3A26+GMT-0700+(Pacific+Daylight+Time)&version=202402.1.0&browserGpcFlag=0&isIABGlobal=false&hosts=&consentId=1ff3be3e-e588-4932-9ac3-2630dd7c33aa&interactionCount=1&isAnonUser=1&landingPath=NotLandingPage&groups=1%3A1%2C3%3A1%2C2%3A1%2C4%3A1&AwaitingReconsent=false; _ga=GA1.1.221978663.1757447861; _li_ss=CkoKBgj5ARDvGwoFCAoQ7xsKBgjdARDvGwoGCOEBEO8bCgYIgQEQ7xsKBgiiARDvGwoJCP____8HEPkbCgYIiQEQ7xsKBgilARDvGw; cjConsent=MHxOfDB8Tnww; cjUser=7ca4c4b8-116d-45aa-ad4c-88a5d183fc3d; ssb_instance_id=b9822ad1-6df9-40d2-8374-d0b286d41559; no_download_ssb_banner=1; show_download_ssb_banner=1; shown_download_ssb_modal=1; lc=1759448208; _fbp=fb.1.1759438148806.71444673446120306; optimizelySession=0; agentforce_chatID=; _gcl_au=1.1.536689637.1757447861.707819349.1759440091.1759440092; _cs_c=0; _lc2_fpi=e00b11ac9c9b--01k4r0wbxafwq3ab8a66d6tj9e; tz=-420; b=.f3db5096c114fdcea90c10e9316228dc"
    WORKSPACE_URL = "https://paradedb.slack.com"
    paradedb = (XOXC_TOKEN, FULL_COOKIES, WORKSPACE_URL)"""

    XOXC_TOKEN = "xoxc-10309498066-9657018397168-9618679318295-aedd922a0ecf736d97516b1066143e00ab3c666d8dc92329568a18858785e09a"
    FULL_COOKIES = "utm=%7B%7D; x=f3db5096c114fdcea90c10e9316228dc.1759516210; no_download_ssb_banner=1; show_download_ssb_banner=1; shown_download_ssb_modal=1; OptanonConsent=isGpcEnabled=0&datestamp=Fri+Oct+03+2025+12%3A33%3A14+GMT-0700+(Pacific+Daylight+Time)&version=202402.1.0&browserGpcFlag=0&isIABGlobal=false&hosts=&consentId=1ff3be3e-e588-4932-9ac3-2630dd7c33aa&interactionCount=1&isAnonUser=1&landingPath=NotLandingPage&groups=1%3A1%2C3%3A1%2C2%3A1%2C4%3A1&AwaitingReconsent=false; shown_ssb_redirect_page=1; _fbp=fb.1.1759438148806.71444673446120306; _cs_s=3.0.U.9.1759521785849; web_cache_last_updated18543652b4683764baf47d66b363e7c0=1759518395577; d=xoxd-D3ntxMXc27YoFu5E1RThAT7u87Rt3ezKnRKv7U4s%2FTyf14Z4QLQWwavzDIOwEeHN%2Fv0rbpJ%2BjQrpJYY6%2BNkJO%2FqTW3jv%2FVCRlvxw5Q3Ysc5jHt0ktcw0DP3VBiWE6vwQE2Pm8KzZo316PorTcLWbBoT84tjdY4qs32lf0cKYwxEHgPhlCMrcH6wIWcOJHORs0T%2BXQnFFmCIs011DWAsY6eLBjeU%3D; lc=1759518364; _ga_QTJQME5M5D=GS2.1.s1759517027$o6$g1$t1759518055$j60$l0$h0; _ga=GA1.1.221978663.1757447861; PageCount=3; _cs_id=65bbed2f-e942-a0d8-ff58-7364edf3ae6f.1757447860.8.1759518029.1759517027.1.1791611860287.1.x; _li_ss=ClkKBgj5ARDvGwoFCAoQ7xsKBgikARDvGwoGCN0BEO8bCgYI4QEQ7xsKBgiBARDvGwoGCKIBEO8bCgkI_____wcQ-RsKBQh-EPAbCgYIiQEQ7xsKBgilARDvGw; cjConsent=MHxOfDB8Tnww; cjUser=7ca4c4b8-116d-45aa-ad4c-88a5d183fc3d; ec=enQtOTYzODE1OTM3NDU0OC1jNzRiOWUzYmE5YWE4OWQ3YzZjNDc2ODZkYmNlMjNmNGM1YmZiYjVhMmUwYmQ3NDgyYzk5MGVkOTU5M2ZkODA5; _cs_cvars=%7B%225%22%3A%5B%22curr_plan%22%2C%22free%22%5D%2C%226%22%3A%5B%22is_paid_plan%22%2C%22false%22%5D%7D; _lc2_fpi_js=e00b11ac9c9b--01k4r0wbxafwq3ab8a66d6tj9e; _li_dcdm_c=.slack.com; d-s=1759513078; ssb_instance_id=b9822ad1-6df9-40d2-8374-d0b286d41559; optimizelySession=0; agentforce_chatID=; _gcl_au=1.1.536689637.1757447861.707819349.1759440091.1759440092; _cs_c=0; _lc2_fpi=e00b11ac9c9b--01k4r0wbxafwq3ab8a66d6tj9e; tz=-420; b=.f3db5096c114fdcea90c10e9316228dc"
    WORKSPACE_URL = "https://product-school.slack.com"
    productschool = (XOXC_TOKEN, FULL_COOKIES, WORKSPACE_URL)

    XOXC_TOKEN = "xoxc-3278029974-9625301388741-9647445078145-2f6a7a7fdaec530a7591351ab386e081bdf17414a9956bf40572e3df60ca8728"
    FULL_COOKIES = "utm=%7B%7D; x=f3db5096c114fdcea90c10e9316228dc.1759516210; shown_ssb_redirect_page=1; no_download_ssb_banner=1; show_download_ssb_banner=1; shown_download_ssb_modal=1; OptanonConsent=isGpcEnabled=0&datestamp=Fri+Oct+03+2025+12%3A15%3A34+GMT-0700+(Pacific+Daylight+Time)&version=202402.1.0&browserGpcFlag=0&isIABGlobal=false&hosts=&consentId=1ff3be3e-e588-4932-9ac3-2630dd7c33aa&interactionCount=1&isAnonUser=1&landingPath=NotLandingPage&groups=1%3A1%2C3%3A1%2C2%3A1%2C4%3A1&AwaitingReconsent=false; _fbp=fb.1.1759438148806.71444673446120306; _cs_s=3.0.U.9.1759520727841; web_cache_last_updated18543652b4683764baf47d66b363e7c0=1759518395577; d=xoxd-D3ntxMXc27YoFu5E1RThAT7u87Rt3ezKnRKv7U4s%2FTyf14Z4QLQWwavzDIOwEeHN%2Fv0rbpJ%2BjQrpJYY6%2BNkJO%2FqTW3jv%2FVCRlvxw5Q3Ysc5jHt0ktcw0DP3VBiWE6vwQE2Pm8KzZo316PorTcLWbBoT84tjdY4qs32lf0cKYwxEHgPhlCMrcH6wIWcOJHORs0T%2BXQnFFmCIs011DWAsY6eLBjeU%3D; lc=1759518364; _ga_QTJQME5M5D=GS2.1.s1759517027$o6$g1$t1759518055$j60$l0$h0; _ga=GA1.1.221978663.1757447861; PageCount=3; _cs_id=65bbed2f-e942-a0d8-ff58-7364edf3ae6f.1757447860.8.1759518029.1759517027.1.1791611860287.1.x; _li_ss=ClkKBgj5ARDvGwoFCAoQ7xsKBgikARDvGwoGCN0BEO8bCgYI4QEQ7xsKBgiBARDvGwoGCKIBEO8bCgkI_____wcQ-RsKBQh-EPAbCgYIiQEQ7xsKBgilARDvGw; cjConsent=MHxOfDB8Tnww; cjUser=7ca4c4b8-116d-45aa-ad4c-88a5d183fc3d; ec=enQtOTYzODE1OTM3NDU0OC1jNzRiOWUzYmE5YWE4OWQ3YzZjNDc2ODZkYmNlMjNmNGM1YmZiYjVhMmUwYmQ3NDgyYzk5MGVkOTU5M2ZkODA5; _cs_cvars=%7B%225%22%3A%5B%22curr_plan%22%2C%22free%22%5D%2C%226%22%3A%5B%22is_paid_plan%22%2C%22false%22%5D%7D; _lc2_fpi_js=e00b11ac9c9b--01k4r0wbxafwq3ab8a66d6tj9e; _li_dcdm_c=.slack.com; d-s=1759513078; ssb_instance_id=b9822ad1-6df9-40d2-8374-d0b286d41559; optimizelySession=0; agentforce_chatID=; _gcl_au=1.1.536689637.1757447861.707819349.1759440091.1759440092; _cs_c=0; _lc2_fpi=e00b11ac9c9b--01k4r0wbxafwq3ab8a66d6tj9e; tz=-420; b=.f3db5096c114fdcea90c10e9316228dc"
    WORKSPACE_URL = "https://lara-chat.slack.com"
    larachat = (XOXC_TOKEN, FULL_COOKIES, WORKSPACE_URL)

    """XOXC_TOKEN = "xoxc-256513324021-9613451281991-9662933995600-994150e91ed2bda16992ed84e2d8ae73b9b934df914b1be16a07ad7c598b9f0e"
    FULL_COOKIES = "utm=%7B%7D; x=f3db5096c114fdcea90c10e9316228dc.1759516210; uc=xoxd-D3ntxMXc27YoFu5E1RThAT7u87Rt3ezKnRKv7U4s%2FTyf14Z4QLQWwavzDIOwEeHN%2Fv0rbpJ%2BjQrpJYY6%2BNkJO%2FqTW3jv%2FVCRlvxw5Q3Ysc5jHt0ktcw0DP3VBiWE6vwQE2Pm8KzZo316PorTcLWbBoT84tjdY4qs32lf0cKYwxEHgPhlCMrcH6wIWcOJHORs0T%2BXQnFFmCIs011DWAsY6eLBjeU%3D; no_download_ssb_banner=1; show_download_ssb_banner=1; shown_download_ssb_modal=1; shown_ssb_redirect_page=1; OptanonConsent=isGpcEnabled=0&datestamp=Fri+Oct+03+2025+12%3A06%3A05+GMT-0700+(Pacific+Daylight+Time)&version=202402.1.0&browserGpcFlag=0&isIABGlobal=false&hosts=&consentId=1ff3be3e-e588-4932-9ac3-2630dd7c33aa&interactionCount=1&isAnonUser=1&landingPath=NotLandingPage&groups=1%3A1%2C3%3A1%2C2%3A1%2C4%3A1&AwaitingReconsent=false; _fbp=fb.1.1759438148806.71444673446120306; d=xoxd-jD%2FTl6jkQuUptj90cXPQ%2BOGLsyzgXULYQbPC%2BMQyw6NmYGsGDd3Arn1bviewlOiHJ4AMPwF69a8TQPVqRFLQp54xcpccIstR5DaWCP1FdBm7GN%2Bmgon1q4%2B2WdvLdkC1JT9lWjb9dz8FAlnceRFcBZlbcaWjr1C%2F5%2Bu90Bpql%2BcpsP2yklH8XVDJaDIRpqUgPyqKFeFoVA%3D%3D; lc=1759518364; _cs_s=3.0.U.9.1759520162786; _ga_QTJQME5M5D=GS2.1.s1759517027$o6$g1$t1759518055$j60$l0$h0; _ga=GA1.1.221978663.1757447861; PageCount=3; _cs_id=65bbed2f-e942-a0d8-ff58-7364edf3ae6f.1757447860.8.1759518029.1759517027.1.1791611860287.1.x; _li_ss=ClkKBgj5ARDvGwoFCAoQ7xsKBgikARDvGwoGCN0BEO8bCgYI4QEQ7xsKBgiBARDvGwoGCKIBEO8bCgkI_____wcQ-RsKBQh-EPAbCgYIiQEQ7xsKBgilARDvGw; cjConsent=MHxOfDB8Tnww; cjUser=7ca4c4b8-116d-45aa-ad4c-88a5d183fc3d; ec=enQtOTYzODE1OTM3NDU0OC1jNzRiOWUzYmE5YWE4OWQ3YzZjNDc2ODZkYmNlMjNmNGM1YmZiYjVhMmUwYmQ3NDgyYzk5MGVkOTU5M2ZkODA5; _cs_cvars=%7B%225%22%3A%5B%22curr_plan%22%2C%22free%22%5D%2C%226%22%3A%5B%22is_paid_plan%22%2C%22false%22%5D%7D; _lc2_fpi_js=e00b11ac9c9b--01k4r0wbxafwq3ab8a66d6tj9e; _li_dcdm_c=.slack.com; d-s=1759513078; ssb_instance_id=b9822ad1-6df9-40d2-8374-d0b286d41559; optimizelySession=0; agentforce_chatID=; _gcl_au=1.1.536689637.1757447861.707819349.1759440091.1759440092; _cs_c=0; _lc2_fpi=e00b11ac9c9b--01k4r0wbxafwq3ab8a66d6tj9e; tz=-420; b=.f3db5096c114fdcea90c10e9316228dc"
    WORKSPACE_URL = "https://loveforgames.slack.com"
    loveforgames = (XOXC_TOKEN, FULL_COOKIES, WORKSPACE_URL)"""

    # Choose which method to use:
    # archive_with_app_token()
    archive_with_browser_session(*productschool)