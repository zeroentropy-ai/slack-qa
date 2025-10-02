import requests
from requests_toolbelt.multipart.encoder import MultipartEncoder

XOXC_TOKEN = "xoxc-3052645262231-9641512460897-9626513329798-19e797687a5e0bb7539701cd740f4a9b3c98f040ebd6213e7f33577468f85c6d"
XOXD_TOKEN = "xoxd-RWApM1zwIRrr+2XACnx2j85mNnq7jXRq08nLu92OuSVSD/Oq/JrB6huw+0E7SYpflbEhm7gKHZsqNX3mypdDoskW+s9x+DXCS74une9GFhai5C8C5pxXVy60iJT/U685IypCMgGzGnIiGmv03u+QU/CuEBtrLAVOjxqeLlunHjtjotJQAgBX2fJuQVq59uDyGb7SNSH5YWJEFrqdJFA6i0nxDrY2"

# All the cookies from your browser
FULL_COOKIES = "utm=%7B%7D; x=f3db5096c114fdcea90c10e9316228dc.1759443932; _cs_s=34.5.U.9.1759446530433; _fbp=fb.1.1759438148806.71444673446120306; _ga_QTJQME5M5D=GS2.1.s1759444670$o3$g1$t1759444703$j60$l0$h0; _ga=GA1.1.221978663.1757447861; optimizelySession=0; OptanonConsent=isGpcEnabled=0&datestamp=Thu+Oct+02+2025+15%3A37%3A50+GMT-0700+(Pacific+Daylight+Time)&version=202402.1.0&browserGpcFlag=0&isIABGlobal=false&hosts=&consentId=1ff3be3e-e588-4932-9ac3-2630dd7c33aa&interactionCount=1&isAnonUser=1&landingPath=NotLandingPage&groups=1%3A1%2C3%3A1%2C2%3A1%2C4%3A1&AwaitingReconsent=false; PageCount=34; _cs_id=65bbed2f-e942-a0d8-ff58-7364edf3ae6f.1757447860.2.1759444670.1759437756.1.1791611860287.1.x; _li_ss=ClIKBgj5ARDvGwoFCAoQ7xsKBgikARDvGwoGCN0BEO8bCgYI4QEQ7xsKBgiBARDvGwoGCKIBEO8bCgkI_____wcQ-RsKBgiJARDvGwoGCKUBEO8b; agentforce_chatID=; cjConsent=MHxOfDB8Tnww; cjUser=7ca4c4b8-116d-45aa-ad4c-88a5d183fc3d; shown_ssb_redirect_page=1; d=xoxd-RWApM1zwIRrr%2B2XACnx2j85mNnq7jXRq08nLu92OuSVSD%2FOq%2FJrB6huw%2B0E7SYpflbEhm7gKHZsqNX3mypdDoskW%2Bs9x%2BDXCS74une9GFhai5C8C5pxXVy60iJT%2FU685IypCMgGzGnIiGmv03u%2BQU%2FCuEBtrLAVOjxqeLlunHjtjotJQAgBX2fJuQVq59uDyGb7SNSH5YWJEFrqdJFA6i0nxDrY2; lc=1759444060; ec=enQtOTY1Njg2MjM5NDcyMC02MWE0MjMyMGNmMjlhNzJjZWI5ZTVhZjdjMzhlYTY1NTc5ODE5YTNmYjA1M2QzMjdkMDc2MjJhNWQ2NzY4ODA0; _gcl_au=1.1.536689637.1757447861.707819349.1759440091.1759440092; ssb_instance_id=b9822ad1-6df9-40d2-8374-d0b286d41559; _cs_cvars=%7B%7D; _lc2_fpi_js=e00b11ac9c9b--01k4r0wbxafwq3ab8a66d6tj9e; _li_dcdm_c=.slack.com; d-s=1758655485; _cs_c=0; _lc2_fpi=e00b11ac9c9b--01k4r0wbxafwq3ab8a66d6tj9e; tz=-420; no_download_ssb_banner=1; show_download_ssb_banner=1; shown_download_ssb_modal=1; b=.f3db5096c114fdcea90c10e9316228dc"

# Prepare multipart form data with token
multipart_data = MultipartEncoder(
    fields={'token': XOXC_TOKEN}
)

headers = {
    "Accept": "*/*",
    "Accept-Encoding": "gzip, deflate, br",
    "Accept-Language": "en-US,en;q=0.9",
    "Cookie": FULL_COOKIES,
    "Content-Type": multipart_data.content_type,
    "Origin": "https://app.slack.com",
    "Referer": "https://app.slack.com/",
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.5 Safari/605.1.15",
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "same-site",
}

# Query parameters from the browser
params = {
    "_x_id": "test-1759444960.566",
    "slack_route": "T031JJZ7Q6T",
    "_x_version_ts": "1759439266",
    "_x_frontend_build_type": "current",
    "_x_desktop_ia": "4",
    "_x_gantry": "true",
    "fp": "a2"
}

response = requests.post(
    "https://modallabscommunity.slack.com/api/conversations.list",
    headers=headers,
    params=params,
    data=multipart_data
)

print(response.json())