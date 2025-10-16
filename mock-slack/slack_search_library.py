#!/usr/bin/env python3
"""
Slack search library with round-robin tokens, rate limiting, and caching
"""
import json
import asyncio
import time
import os
from typing import List, Dict, Any
from dataclasses import asdict

# Import Slack search components
import sys
sys.path.append('..')
from slack_search import SlackSearch

# Global cache and rate limiter for Slack searches
slack_search_cache = {}
last_slack_request_time = 0
MIN_SLACK_INTERVAL = 4.0  # 4 seconds between requests for rate limiting

# Load tokens and cookies for Slack API
TOKENS_AND_COOKIES = [
    {
        "token": "xoxc-3052645262231-9689129827878-9704176242193-ede9e23190f6b136aceaab8e42bd414808ddf9032c0d33aa33bcb9ae4410a5d8",
        "cookies": 'b=.693510bbd0d5677b628fff03f268acf4; d-s=1759520448; utm=%7B%7D; x=693510bbd0d5677b628fff03f268acf4.1760409378; shown_ssb_redirect_page=1; shown_download_ssb_modal=1; show_download_ssb_banner=1; no_download_ssb_banner=1; tz=-420; web_cache_last_updated5f1c2806a2541abd794aa08422f95de2=1760409445559; lc=1760410519; d=xoxd-HhQFLWY%2B0vp3R1qj1eDXWlR3Jj4kuvHRdPdUHUs%2FYKbKTAKhPdpUM%2BuR5EKcQtJIw7nmeL57HKM%2F3aY6FwNEf%2Fb6hcGBjKcHrDrooNTZhEJ0GI92aIlwJktXkV9amDk02Y1HsfBBYD6vp7UPpxHwCa%2Fq6zyfupwaHEFOn0Y5Th6CkrM1GL4aGy1GjWol6auPBfkOiBe5OQM%2B5EsIJZyVOdnWR11I; web_cache_last_updated34a7ff4d0b036d3d72ee8717822ef770=1760411388730'
    },
    {
        "token": "xoxc-3052645262231-9641512460897-9626513329798-19e797687a5e0bb7539701cd740f4a9b3c98f040ebd6213e7f33577468f85c6d",
        "cookies": 'utm=%7B%7D; d=xoxd-9jnd5xe9oeEUyLp%2BRKca5gj8q52vJn6HmzamGg6lmEe6lt2qvUO9qlhpnpwxYOL%2BNXsgi02JupH%2F0rv2ZSWMFhXcpokUbyyruy3%2FzQuAZGcU5naZQmwOyzshjHIp9%2B7hHId567haJOfjL63ak6Gln7ui6sZG413neXIOiz%2FPs6J5OI9aMJanpXQDW7szEUQ0TdcU8ZBcUrdcoYyI0rFMuD65; x=f3db5096c114fdcea90c10e9316228dc.1760473163; shown_ssb_redirect_page=1; OptanonConsent=isGpcEnabled=0&datestamp=Sun+Oct+12+2025+12%3A14%3A38+GMT-0700+(Pacific+Daylight+Time)&version=202402.1.0&browserGpcFlag=0&isIABGlobal=false&hosts=&consentId=1ff3be3e-e588-4932-9ac3-2630dd7c33aa&interactionCount=1&isAnonUser=1&landingPath=NotLandingPage&groups=1%3A1%2C3%3A1%2C2%3A1%2C4%3A1&AwaitingReconsent=false; _ga=GA1.1.221978663.1757447861; _ga_QTJQME5M5D=GS2.1.s1760296466$o9$g0$t1760296466$j60$l0$h0; _cs_cvars=%7B%7D; _cs_id=65bbed2f-e942-a0d8-ff58-7364edf3ae6f.1757447860.14.1760296466.1760296466.1.1791611860287.1.x; _lc2_fpi_js=e00b11ac9c9b--01k4r0wbxafwq3ab8a66d6tj9e; _li_dcdm_c=.slack.com; _li_ss=ClkKBgj5ARD1GwoFCAoQ9RsKBgikARD5GwoGCN0BEPUbCgYI4QEQ9RsKBgiBARD1GwoGCKIBEPUbCgkI_____wcQ-RsKBQh-EPUbCgYIiQEQ-RsKBgilARD5Gw; cjConsent=MHxOfDB8Tnww; cjUser=7ca4c4b8-116d-45aa-ad4c-88a5d183fc3d; PageCount=1; ssb_instance_id=b9822ad1-6df9-40d2-8374-d0b286d41559; d-s=1760296437; no_download_ssb_banner=1; show_download_ssb_banner=1; shown_download_ssb_modal=1; _fbp=fb.1.1759438148806.71444673446120306; lc=1759536553; optimizelySession=0; _gcl_au=1.1.536689637.1757447861.707819349.1759440091.1759440092; _cs_c=0; _lc2_fpi=e00b11ac9c9b--01k4r0wbxafwq3ab8a66d6tj9e; tz=-420; b=.f3db5096c114fdcea90c10e9316228dc'
    },
    {
        "token": "xoxc-3052645262231-9697857150834-9691497212867-37f1525a905e8505827d929693c18f405ddd3fd12167cbcda985ad33ad8f9dc3",
        "cookies": 'utm=%7B%7D; b=.a5bc76f488ac86ccc37120cf98c842d6; x=a5bc76f488ac86ccc37120cf98c842d6.1760476907; OptanonConsent=isGpcEnabled=0&datestamp=Tue+Oct+14+2025+14%3A29%3A22+GMT-0700+(Pacific+Daylight+Time)&version=202402.1.0&browserGpcFlag=0&isIABGlobal=false&hosts=&consentId=5304d73c-f71d-4e85-a496-29147eab897a&interactionCount=1&isAnonUser=1&landingPath=NotLandingPage&groups=1%3A1%2C3%3A1%2C2%3A1%2C4%3A1&AwaitingReconsent=false; d=xoxd-%2BYOrnqBVqe8psoS5%2BHYoJaX%2F8RbXjLnNNqxp98d9B0TubF90rrxMMASHnUnAdjl2NUz%2BAkH%2BPlXnm%2BXsPMpVPmsUHTWkTC8MEzlxRc5gJ0igjdF%2BFr3oPLzvDG2esxuYev5aBL85ZuzxrcnECKSAM%2BNlW92V0zagayns6wLWjrGu3b18ZL0dqv71C60MOTIL0xXp3Jk%3D; lc=1760477361; d-s=1760477361; shown_ssb_redirect_page=1; shown_download_ssb_modal=1; show_download_ssb_banner=1; no_download_ssb_banner=1; tz=-420'
    }
]

# Initialize Slack clients (round-robin)
slack_clients = []
current_client_index = 0

def initialize_slack_clients():
    """Initialize Slack search clients"""
    global slack_clients
    if not slack_clients:
        for i, creds in enumerate(TOKENS_AND_COOKIES):
            client = SlackSearch(
                token=creds["token"],
                auth_mode='browser',
                cookies=creds["cookies"],
                workspace_url='https://modallabscommunity.slack.com'
            )
            slack_clients.append(client)

def get_next_slack_client():
    """Get next Slack client in round-robin fashion"""
    global current_client_index
    if not slack_clients:
        initialize_slack_clients()
    
    client = slack_clients[current_client_index]
    current_client_index = (current_client_index + 1) % len(slack_clients)
    return client

async def rate_limited_slack_search(query: str):
    """Perform rate-limited Slack search with caching"""
    global last_slack_request_time
    
    # Check cache first
    if query in slack_search_cache:
        return slack_search_cache[query]
    
    # Rate limiting
    current_time = time.time()
    time_since_last = current_time - last_slack_request_time
    if time_since_last < MIN_SLACK_INTERVAL:
        wait_time = MIN_SLACK_INTERVAL - time_since_last
        await asyncio.sleep(wait_time)
    
    # Get client and perform search
    client = get_next_slack_client()
    try:
        result = await client.search_async(query, search_type="messages", count=100)
        result_dict = asdict(result)
        
        # Cache the result
        slack_search_cache[query] = result_dict
        last_slack_request_time = time.time()
        
        return result_dict
    except Exception as e:
        print(f"Error in Slack search for '{query}': {e}")
        return {"matches": [], "total": 0}

async def batch_slack_search_with_ranks(queries: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """
    Perform batch Slack search and return rank of target document for each query
    
    Args:
        queries: List of dicts with "search_query" and "target_document_id" keys
    
    Returns:
        List of dicts with "search_query", "target_document_id", and "slack_target_rank" keys
        slack_target_rank is null if document not found in results, otherwise 1-based rank
    """
    # Load timestamp and message mappings
    with open("../timestamp_to_message_id.json") as f:
        timestamp_to_message_id = json.load(f)
    
    message_id_to_document_id = {}
    with open("../synthetic_data/Modal_Community_T031JJZ7Q6T/beir_format_individual_message/documents.jsonl") as f:
        for line in f:
            if "{" not in line:
                continue
            doc = json.loads(line)
            message_id = doc["metadata"]["message_id"]
            if message_id not in message_id_to_document_id:
                message_id_to_document_id[message_id] = set()
            message_id_to_document_id[message_id].add(doc["id"])
    
    results = []
    
    for query_info in queries:
        search_query = query_info["search_query"]
        target_doc_id = query_info["target_document_id"]
        
        # Perform Slack search
        slack_result = await rate_limited_slack_search(search_query)
        
        # Find rank of target document
        slack_target_rank = None
        matches = slack_result.get("matches", [])
        
        for i, match in enumerate(matches, 1):
            ts = match.get("ts", "")
            message_id = timestamp_to_message_id.get(ts, "")
            if message_id:
                doc_ids = message_id_to_document_id.get(message_id, set())
                if target_doc_id in doc_ids:
                    slack_target_rank = i
                    break
        
        # Copy input object and add slack_target_rank
        result = query_info.copy()
        result["slack_target_rank"] = slack_target_rank
        results.append(result)
    
    return results

def commit_trace(filename: str = "slack_search_cache.json"):
    """Save the Slack search cache to disk"""
    try:
        with open(filename, "w") as f:
            json.dump(slack_search_cache, f, indent=2)
        print(f"Slack search cache saved to {filename} ({len(slack_search_cache)} entries)")
    except Exception as e:
        print(f"Error saving cache: {e}")

def load_cache(filename: str = "slack_search_cache.json"):
    """Load the Slack search cache from disk"""
    global slack_search_cache
    try:
        with open(filename, "r") as f:
            slack_search_cache = json.load(f)
        print(f"Loaded Slack search cache from {filename} ({len(slack_search_cache)} entries)")
    except FileNotFoundError:
        print(f"No cache file found at {filename}, starting with empty cache")
    except Exception as e:
        print(f"Error loading cache: {e}")

# Load cache on module import
load_cache()