#!/usr/bin/env python3
"""
Download Slack data for training dataset generation
Uses browser authentication with token rotation and rate limiting
"""
import json
import requests
import time
import os
import asyncio
from pathlib import Path
from typing import Dict, List, Any
from tqdm import tqdm
from requests_toolbelt.multipart.encoder import MultipartEncoder

class RateLimiter:
    """Rate limiter for API calls"""
    def __init__(self, max_per_minute=20, num_tokens=1):
        self.max_per_minute = max_per_minute
        self.num_tokens = num_tokens
        self.min_interval = 60.0 / (max_per_minute * num_tokens)
        self.last_request_time = 0

    async def wait_if_needed(self):
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_interval:
            wait_time = self.min_interval - time_since_last
            await asyncio.sleep(wait_time)
        self.last_request_time = time.time()

class SlackTokenPool:
    """Manages multiple tokens with round-robin rotation"""
    def __init__(self, tokens_and_cookies: List[Dict[str, str]]):
        self.credentials = tokens_and_cookies
        self.current_index = 0
        print(f"Initialized with {len(tokens_and_cookies)} token(s)")
    
    def get_next_credentials(self) -> Dict[str, str]:
        creds = self.credentials[self.current_index]
        self.current_index = (self.current_index + 1) % len(self.credentials)
        return creds

class SlackDownloader:
    def __init__(self, workspace_url: str, token_pool: SlackTokenPool):
        self.workspace_url = workspace_url.rstrip('/')
        self.token_pool = token_pool
        self.rate_limiter = RateLimiter(max_per_minute=20, num_tokens=len(token_pool.credentials))
    
    async def api_call(self, method: str, params: Dict = None) -> Dict[str, Any]:
        """Make API call with rate limiting and token rotation"""
        await self.rate_limiter.wait_if_needed()
        
        # Get next credentials
        creds = self.token_pool.get_next_credentials()
        
        # Prepare multipart form data
        fields = {'token': creds['token']}
        if params:
            fields.update(params)
        
        multipart_data = MultipartEncoder(fields=fields)
        
        headers = {
            "Accept": "*/*",
            "Accept-Encoding": "",
            "Accept-Language": "en-US,en;q=0.9",
            "Cookie": creds['cookies'],
            "Content-Type": multipart_data.content_type,
            "Origin": "https://app.slack.com",
            "Referer": "https://app.slack.com/",
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15",
        }
        
        query_params = {
            "_x_id": f"download-{int(time.time() * 1000)}",
            "_x_version_ts": str(int(time.time())),
            "_x_frontend_build_type": "current",
        }
        
        try:
            response = requests.post(
                f"{self.workspace_url}/api/{method}",
                headers=headers,
                params=query_params,
                data=multipart_data
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error calling {method}: {e}")
            return {"ok": False, "error": str(e)}
    
    async def get_channels(self) -> List[Dict[str, Any]]:
        """Get list of public channels"""
        print("Fetching channels...")
        result = await self.api_call("conversations.list", {
            "types": "public_channel",
            "exclude_archived": "true",
            "limit": "1000"
        })
        
        if result.get("ok"):
            channels = result.get("channels", [])
            print(f"Found {len(channels)} public channels")
            return channels
        else:
            print(f"Error fetching channels: {result}")
            return []
    
    async def get_users(self) -> Dict[str, Dict[str, Any]]:
        """Get workspace users"""
        print("Fetching users...")
        result = await self.api_call("users.list", {"limit": "1000"})
        
        if result.get("ok"):
            users = result.get("members", [])
            user_map = {user["id"]: user for user in users}
            print(f"Found {len(users)} users")
            return user_map
        else:
            print(f"Error fetching users: {result.get('error')}")
            return {}
    
    async def get_channel_history(self, channel_id: str, channel_name: str, limit: int = 1000) -> List[Dict[str, Any]]:
        """Get message history for a channel"""
        print(f"Fetching messages from #{channel_name}...")
        
        all_messages = []
        cursor = None
        
        while len(all_messages) < limit:
            params = {
                "channel": channel_id,
                "limit": str(min(200, limit - len(all_messages)))
            }
            if cursor:
                params["cursor"] = cursor
            
            result = await self.api_call("conversations.history", params)
            
            if not result.get("ok"):
                print(f"Error fetching messages: {result.get('error')}")
                break
            
            messages = result.get("messages", [])
            all_messages.extend(messages)
            
            # Get cursor for pagination
            cursor = result.get("response_metadata", {}).get("next_cursor")
            if not cursor:
                break
        
        print(f"  Retrieved {len(all_messages)} messages")
        return all_messages
    
    async def get_thread_replies(self, channel_id: str, thread_ts: str) -> List[Dict[str, Any]]:
        """Get replies in a thread"""
        result = await self.api_call("conversations.replies", {
            "channel": channel_id,
            "ts": thread_ts
        })
        
        if result.get("ok"):
            return result.get("messages", [])
        return []
    
    def select_channels(self, channels: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Let user select which channels to download"""
        print(f"\nAvailable channels:")
        for i, channel in enumerate(channels):
            members = channel.get("num_members", 0)
            print(f"{i+1:3}. #{channel['name']:<20} ({members} members)")
        
        print(f"\nSelect channels to download:")
        print("Enter channel numbers separated by commas (e.g., 1,3,5)")
        print("Or enter channel names separated by commas (e.g., general,random)")
        print("Or 'all' for all channels")
        
        selection = input("Selection: ").strip()
        
        if selection.lower() == 'all':
            return channels
        
        selected_channels = []
        
        # Try parsing as numbers first
        if selection.replace(',', '').replace(' ', '').isdigit():
            try:
                indices = [int(x.strip()) - 1 for x in selection.split(',')]
                selected_channels = [channels[i] for i in indices if 0 <= i < len(channels)]
            except (ValueError, IndexError):
                print("Invalid selection")
                return []
        else:
            # Parse as channel names
            names = [name.strip().lstrip('#') for name in selection.split(',')]
            channel_map = {ch['name']: ch for ch in channels}
            selected_channels = [channel_map[name] for name in names if name in channel_map]
            
            # Show which channels weren't found
            missing = [name for name in names if name not in channel_map]
            if missing:
                print(f"Channels not found: {missing}")
        
        if selected_channels:
            print(f"Selected {len(selected_channels)} channels:")
            for ch in selected_channels:
                print(f"  #{ch['name']}")
        
        return selected_channels
    
    async def download_workspace_data(self, output_dir: str = "slack_training_data", 
                                     messages_per_channel: int = 1000):
        """Download data from workspace"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Get workspace info
        users = await self.get_users()
        channels = await self.get_channels()
        
        if not channels:
            print("No channels found!")
            return
        
        # Sort channels by member count (most active first)
        channels.sort(key=lambda x: x.get("num_members", 0), reverse=True)
        
        # Let user select channels
        selected_channels = self.select_channels(channels)
        if not selected_channels:
            print("No channels selected!")
            return
        
        workspace_data = {
            "workspace_url": self.workspace_url,
            "users": users,
            "channels": [],
            "download_timestamp": time.time()
        }
        
        for channel in tqdm(selected_channels, desc="Processing channels"):
            channel_id = channel["id"]
            channel_name = channel["name"]
            
            # Get messages
            messages = await self.get_channel_history(channel_id, channel_name, messages_per_channel)
            
            # Get thread replies for threaded messages
            for message in messages:
                if message.get("thread_ts") and message.get("reply_count", 0) > 0:
                    replies = await self.get_thread_replies(channel_id, message["thread_ts"])
                    message["replies"] = replies[1:]  # Exclude parent message
            
            channel_data = {
                "id": channel_id,
                "name": channel_name,
                "info": channel,
                "messages": messages
            }
            workspace_data["channels"].append(channel_data)
        
        # Save data
        output_file = output_path / "workspace_data.json"
        with open(output_file, 'w') as f:
            json.dump(workspace_data, f, indent=2)
        
        print(f"Data saved to {output_file}")
        
        # Generate summary
        total_messages = sum(len(ch["messages"]) for ch in workspace_data["channels"])
        print(f"\nSummary:")
        print(f"Channels: {len(workspace_data['channels'])}")
        print(f"Users: {len(users)}")
        print(f"Total messages: {total_messages}")
        
        return workspace_data

async def main():
    print("Slack Data Downloader with Token Rotation")
    print("=" * 40)
    
    # Use predefined tokens from agentic_search.py
    tokens_and_cookies = [
        {
            "token": "xoxc-3052645262231-9697857150834-9691497212867-37f1525a905e8505827d929693c18f405ddd3fd12167cbcda985ad33ad8f9dc3",
            "cookies": 'utm=%7B%7D; b=.a5bc76f488ac86ccc37120cf98c842d6; x=a5bc76f488ac86ccc37120cf98c842d6.1760476907; OptanonConsent=isGpcEnabled=0&datestamp=Tue+Oct+14+2025+14%3A29%3A22+GMT-0700+(Pacific+Daylight+Time)&version=202402.1.0&browserGpcFlag=0&isIABGlobal=false&hosts=&consentId=5304d73c-f71d-4e85-a496-29147eab897a&interactionCount=1&isAnonUser=1&landingPath=NotLandingPage&groups=1%3A1%2C3%3A1%2C2%3A1%2C4%3A1&AwaitingReconsent=false; d=xoxd-%2BYOrnqBVqe8psoS5%2BHYoJaX%2F8RbXjLnNNqxp98d9B0TubF90rrxMMASHnUnAdjl2NUz%2BAkH%2BPlXnm%2BXsPMpVPmsUHTWkTC8MEzlxRc5gJ0igjdF%2BFr3oPLzvDG2esxuYev5aBL85ZuzxrcnECKSAM%2BNlW92V0zagayns6wLWjrGu3b18ZL0dqv71C60MOTIL0xXp3Jk%3D; lc=1760477361; d-s=1760477361; shown_ssb_redirect_page=1; shown_download_ssb_modal=1; show_download_ssb_banner=1; no_download_ssb_banner=1; tz=-420'
        }
    ]
    
    WORKSPACE_URL = "https://modallabscommunity.slack.com"
    
    print(f"Using {len(tokens_and_cookies)} tokens for Modal Labs Community")
    print(f"Workspace: {WORKSPACE_URL}")
    
    messages_per_channel = int(input("Messages per channel (default 1000): ") or "1000")
    
    # Create token pool and downloader
    token_pool = SlackTokenPool(tokens_and_cookies)
    downloader = SlackDownloader(WORKSPACE_URL, token_pool)
    
    try:
        workspace_data = await downloader.download_workspace_data(
            messages_per_channel=messages_per_channel
        )
        print("\n✅ Download complete!")
        
    except KeyboardInterrupt:
        print("\n❌ Download interrupted by user")
    except Exception as e:
        print(f"\n❌ Download failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())
