"""
One-time local setup script — generates Google Drive OAuth credentials
for use in Kaggle Save & Run sessions.

Run this ONCE on your local machine. It opens a browser, you log in
with your Google account, and it prints the credentials JSON.
Paste that JSON into Kaggle Secrets as 'GDRIVE_OAUTH_CREDS'.

SETUP STEPS (do this before running this script):
─────────────────────────────────────────────────
1. Go to: https://console.cloud.google.com
2. Select your project (e.g. flashmind-ai-rohith)
3. APIs & Services → Enable APIs → enable "Google Drive API"
4. APIs & Services → Credentials → Create Credentials → OAuth 2.0 Client ID
   - Application type: Desktop app
   - Name: kaggle-drive-oauth (any name)
   → Download the client_secrets JSON file
5. Save that file as 'client_secrets.json' in the SAME folder as this script
6. Run:  python setup_gdrive_oauth.py

REQUIREMENTS:
    pip install google-auth-oauthlib google-auth-httplib2 google-api-python-client
"""

import json
import sys
from pathlib import Path

CLIENT_SECRETS_FILE = Path(__file__).parent / "client_secrets.json"
SCOPES = ["https://www.googleapis.com/auth/drive"]

def main():
    if not CLIENT_SECRETS_FILE.exists():
        print("❌  client_secrets.json not found in this directory.")
        print("    Download it from Google Cloud Console:")
        print("    APIs & Services → Credentials → your OAuth 2.0 Client ID → Download JSON")
        print(f"    Save it as: {CLIENT_SECRETS_FILE}")
        sys.exit(1)

    try:
        from google_auth_oauthlib.flow import InstalledAppFlow
    except ImportError:
        print("❌  google-auth-oauthlib not installed.")
        print("    Run: pip install google-auth-oauthlib google-auth-httplib2 google-api-python-client")
        sys.exit(1)

    print("🌐  Opening browser for Google login...")
    print("    Log in with the Google account that OWNS the Drive folder.")
    print()

    flow = InstalledAppFlow.from_client_secrets_file(
        str(CLIENT_SECRETS_FILE),
        scopes=SCOPES,
    )
    # run_local_server opens a browser tab for OAuth consent
    creds = flow.run_local_server(port=0, prompt="consent", access_type="offline")

    # Build the credentials dict to store as Kaggle Secret
    creds_dict = {
        "token":         creds.token,
        "refresh_token": creds.refresh_token,
        "token_uri":     creds.token_uri,
        "client_id":     creds.client_id,
        "client_secret": creds.client_secret,
        "scopes":        list(creds.scopes),
    }

    if not creds_dict.get("refresh_token"):
        print("⚠️  No refresh_token returned.")
        print("    This usually means you already authorised this app before.")
        print("    Fix: go to https://myaccount.google.com/permissions")
        print("    Remove 'kaggle-drive-oauth' → re-run this script.")
        sys.exit(1)

    creds_json = json.dumps(creds_dict, indent=2)

    print("=" * 60)
    print("✅  SUCCESS! Copy the JSON below into Kaggle Secrets.")
    print()
    print("   Kaggle → Account → Settings → Secrets → Add New Secret")
    print("   Name : GDRIVE_OAUTH_CREDS")
    print("   Value: (paste the entire block below, including braces)")
    print()
    print(creds_json)
    print("=" * 60)

    # Also save locally for reference
    out_path = Path(__file__).parent / "gdrive_oauth_creds.json"
    out_path.write_text(creds_json)
    print(f"\n📄  Also saved locally to: {out_path}")
    print("    ⚠️  Keep this file private — it gives full Drive access!")
    print("    Add it to .gitignore if it isn't already.")

if __name__ == "__main__":
    main()
