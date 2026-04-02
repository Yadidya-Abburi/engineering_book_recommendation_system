import os
import sys
import subprocess

if __name__ == "__main__":
    # Point to the new filename
    app_file = os.path.join("app", "app.py")

    if not os.path.exists(app_file):
        print(f"❌ Error: Cannot find {app_file}")
        sys.exit(1)

    print("🚀 Launching Bookify System...")
    print("💡 Tip: The app can fall back to data/final_books.csv, but to load recommendations run scripts/train_model.py first!")
    
    # Execute the Flask app
    subprocess.run([sys.executable, app_file])