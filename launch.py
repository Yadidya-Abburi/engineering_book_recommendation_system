import os
import subprocess
import sys

def launch():
    print("🚀 Initializing Engineering Book Recommendation System...")
    
    # Path to your main app file
    # Ensure this matches the file name inside your /app folder
    app_path = os.path.join("app", "app.py") 

    if not os.path.exists(app_path):
        print(f"❌ Error: Could not find {app_path}")
        return

    try:
        # This command starts the Streamlit server properly
        subprocess.run(["streamlit", "run", app_path])
    except KeyboardInterrupt:
        print("\n👋 System stopped.")
    except Exception as e:
        print(f"❌ An error occurred: {e}")

if __name__ == "__main__":
    launch()
