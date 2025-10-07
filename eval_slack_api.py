import argparse
import json
import sys
from slack_search import SlackSearch, search_messages

# Replace with your actual tokens and workspace info
# Browser mode (recommended)
XOXC_TOKEN = "xoxc-3052645262231-9641679200657-9613480439127-31e794a9afc781c3fc3cdcd019732e0fe4fd25af2b58e86ba83ec0285c3c6283"
FULL_COOKIES = "d-s=1759437760; b=.a61952af5dcace9baebd31847eaf197f; shown_ssb_redirect_page=1; tz=-420; ssb_instance_id=332ff020-d3b1-46d7-8be7-973d48eda65a; optimizelySession=1759438330883; utm=%7B%22utm_source%22%3A%22thehiveindex.com%22%7D; shown_download_ssb_modal=1; show_download_ssb_banner=1; no_download_ssb_banner=1; web_cache_last_updated652e8224e5adbb12232da00d72e1b32f=1759444360895; web_cache_last_updated0ad2d18e8d30e6d95e1c1d8604385e48=1759449180999; web_cache_last_updated58208ac66e071bd79a824469a9c06679=1759449373157; web_cache_last_updated04735373bcef794525852d1ec5c5c79a=1759449506346; lc=1759449634; ec=enQtOTYyOTE5NzI3NDUzMS1hYjcxMGIyMGU1MDMwZTFjYjIzMDc2Y2QzODhmMGY3ODEzMzIwOTQ0ZmQ3MDFmMWE0NjUyMjhiMzFhZDUyODg1; OptanonConsent=isGpcEnabled=0&datestamp=Fri+Oct+03+2025+16%3A06%3A03+GMT-0700+(Pacific+Daylight+Time)&version=202402.1.0&browserGpcFlag=0&isIABGlobal=false&hosts=&consentId=07597623-af2c-4821-bd16-7dcdeea0d673&interactionCount=1&isAnonUser=1&landingPath=NotLandingPage&groups=1%3A1%2C3%3A1%2C2%3A1%2C4%3A1&AwaitingReconsent=false; d=xoxd-NVk82qusHfrcV%2BNk15wbhMBu6Hk6AYucoL6oUxB8o8q2rMXCrN7qPLbHD37GA2fi2oyPPdI3fMDMXFLE%2BTs7WUcJ%2FhM95RGC%2FxpoybCys68QMqmbMMiRNh5nETmtnFGIhe4ByJAR6xQQL87Q8T5H1M7pNR30%2B9TIu9FQnarKqigOtTWzGeeydKbxNrnMoz84eQxc8tVtHcOpf12c0Jid3ZfDbUCk; x=a61952af5dcace9baebd31847eaf197f.1759538401"
WORKSPACE_URL = "https://modallabscommunity.slack.com"


def load_jsonl_file(filepath):
    """
    Load and parse a JSONL file.
    
    Args:
        filepath: Path to the JSONL file
        
    Returns:
        List of dictionaries, one per line in the JSONL file
    """
    data = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:  # Skip empty lines
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        print(f"Warning: Failed to parse line {line_num}: {e}", file=sys.stderr)
        print(f"Successfully loaded {len(data)} records from {filepath}")
        return data
    except FileNotFoundError:
        print(f"Error: File not found: {filepath}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description='Evaluate Slack API with data from a JSONL file')
    parser.add_argument('jsonl_file', help='Path to the JSONL file to process')
    
    args = parser.parse_args()
    
    # Create the SlackSearch object
    search = SlackSearch(
        token=XOXC_TOKEN,
        auth_mode='browser',
        cookies=FULL_COOKIES,
        workspace_url=WORKSPACE_URL
    )

    # Load the JSONL file
    data = load_jsonl_file(args.jsonl_file)
    
    # Process the data (add your processing logic here)
    for query in data[:2]:
        query_text = query.get('query', '')

        keyword = query_text.split()[0] if query_text else 'N/A'
        results = search.search(keyword, search_type="messages", count=5)

        print(f"Results: {results}")


if __name__ == "__main__":
    main()
