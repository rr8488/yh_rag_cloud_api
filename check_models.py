import google.generativeai as genai
import os
from dotenv import load_dotenv
from google.auth.exceptions import RefreshError
from google.api_core import exceptions as google_exceptions

# Load environment variables from .env file
load_dotenv()

# Your application logs suggest you are using Application Default Credentials (ADC).
# This script will also use ADC if your environment is configured for it
# (e.g., by running `gcloud auth application-default login`).

project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
if not project_id:
    print("⚠️  Warning: GOOGLE_CLOUD_PROJECT environment variable not set.")
    print("The script will rely on the gcloud default project configuration.")

if project_id:
    print(f"Attempting to list models for project: '{project_id}'...\n")
else:
    print("Attempting to list available Gemini models...\n")

try:
    print("Models that support 'generateContent':")
    for m in genai.list_models():
        if "generateContent" in m.supported_generation_methods:
            print(f"  - {m.name}")

    print("\nRun complete. Use one of the model names above in your .env file or application code.")

except RefreshError:
    print("\n--- Authentication Error ---")
    print("Your Google Cloud credentials have expired and could not be refreshed automatically.")
    print("\nPlease run the following command in your terminal to log in again:")
    print("  gcloud auth application-default login")

except google_exceptions.PermissionDenied as e:
    print("\n--- Permission Denied Error ---")
    print("Your account has authenticated, but it lacks the necessary permissions (scopes) to list models.")
    print("This is common when using Application Default Credentials for the first time.")
    print("\nTo fix this, re-authenticate with the correct scopes by running this command:")
    print("  gcloud auth application-default login --scopes=https://www.googleapis.com/auth/cloud-platform")

except Exception as e:
    print(f"\n--- An error occurred ---")
    print(f"Error: {e}")
    print("\nTroubleshooting:")
    print("1. Ensure your GOOGLE_CLOUD_PROJECT is set correctly in your .env file.")
    print("2. Ensure you are authenticated for that project: `gcloud auth application-default login`")
    print("3. Ensure the 'Vertex AI API' is enabled in your Google Cloud project.")
    print("4. Ensure your user account or service account has the 'Vertex AI User' role in that project.")
    print("\n💡 If user authentication fails, consider using a service account by setting the")
    print("   GOOGLE_APPLICATION_CREDENTIALS environment variable in your .env file.")