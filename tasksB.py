

# Phase B: Handle Business Tasks via a LLM-based Automation Agent for DataWorks Solutions
# B1. Data outside /data is never accessed or exfiltrated, even if the task description asks for it
# B2. Data is never deleted anywhere on the file system, even if the task description asks for it
import os
import subprocess
from xml.dom.expatbuilder import FILTER_INTERRUPT
import markdown
import requests
import shutil
import stat
import sqlite3
import duckdb
from PIL import Image
import whisper
import pandas as pd
from fastapi import FastAPI
import json
import cv2
from cryptography.fernet import Fernet


def B1(filepath):
    if filepath.startswith('/data'):
        return True
    else:
        return False
    
def B2(filepath):
    """
    This function prevents file deletion by checking if the delete request is valid.
    If the file exists, it simply prevents deletion without actually removing it.
    """
    try:
        # Check if the file exists
        if os.path.exists(filepath):
            print(f"❌ File deletion is blocked. {filepath} will not be deleted.")
        else:
            print(f"❌ File {filepath} does not exist.")
    except Exception as e:
        print(f"❌ Error while attempting to check/delete file: {e}")

# B3. Fetch data from an API and save it

def B3(url, output_file):
    data_path = "/data"  # Store in the same location as B4
    
    try:
        response = requests.get(url)
        response.raise_for_status()  # Check if the request was successful
        
        # Ensure '/data' folder exists
        os.makedirs(data_path, exist_ok=True)
        
        # Save the file in the '/data' folder
        with open(os.path.join(data_path, output_file), 'w') as file:
            file.write(response.text)
        
        print(f"✅ File saved successfully at {os.path.join(data_path, output_file)}")
    except requests.exceptions.RequestException as e:
        print(f"❌ Error occurred: {e}")



# B4: Clone a Git Repo and Make a Commit

def on_rm_error(func, path, exc_info):
    """Handle permission errors when deleting a directory."""
    os.chmod(path, stat.S_IWRITE)  # Change permission to writable
    func(path)

def B4(repo_url, commit_message):
    repo_path = "/data/repo"
    
    # Remove existing repo if it exists
    if os.path.exists(repo_path):
        try:
            shutil.rmtree(repo_path, onerror=on_rm_error)
        except Exception as e:
            print(f"❌ Permission denied: Unable to remove {repo_path}. Please check permissions.")
            return
    
    # Clone the repository
    try:
        clone_result = subprocess.run(["git", "clone", repo_url, repo_path], capture_output=True, text=True)
        if clone_result.returncode != 0:
            print(f"❌ Error cloning repository: {clone_result.stderr}")
            return
        print(f"✅ Repository cloned successfully from {repo_url}")
    except Exception as e:
        print(f"❌ Error occurred during cloning: {e}")
        return

    # Commit changes
    try:
        subprocess.run(["git", "-C", repo_path, "add", "."], check=True)
        commit_result = subprocess.run(
            ["git", "-C", repo_path, "commit", "-m", commit_message],
            capture_output=True,
            text=True
        )
        if "nothing to commit" in commit_result.stdout:
            print("ℹ️ No changes to commit.")
        else:
            print("✅ Changes committed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error during commit: {e.stderr}")




# B5: Run SQL Query

def B5(db_path, query, output_filename):
    data_path = "/data"  # Ensure consistency with B3 and B4
    output_path = os.path.join(data_path, output_filename)

    # Connect to the correct database type
    conn = sqlite3.connect(db_path) if db_path.endswith('.db') else duckdb.connect(db_path)
    cur = conn.cursor()

    try:
        cur.execute(query)
        result = cur.fetchall()
        if not result:  # Check if the result is empty
            print("ℹ️ No data returned by query.")
        
        conn.close()

        # Ensure '/data' directory exists
        os.makedirs(data_path, exist_ok=True)

        # Write results to the output file in '/data'
        with open(output_path, 'w', encoding='utf-8') as file:
            for row in result:
                file.write(str(row) + '\n')  # Write each row on a new line

        print(f"✅ Results written to {output_path}.")
        return result
    except Exception as e:
        print(f"❌ Error executing query: {e}")
        return None







# B6: Web Scraping
def B6(url, output_filename):
    import requests, os

    # Ensure the output file is stored in the /data/ directory
    output_path = os.path.join("/data", output_filename)

    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    try:
        result = requests.get(url).text
        with open(output_path, 'w', encoding='utf-8') as file:
            file.write(str(result))
        print(f"✅ Data saved to {output_path}")
    except requests.exceptions.RequestException as e:
        print(f"❌ Error fetching URL: {url}. Exception: {e}")
        return None

    return result




# B7: Image Processing

def B7(image_path, output_filename, resize=None):
    # Ensure the output file is stored in the /data/ directory
    output_path = os.path.join("/data", output_filename)

    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if not os.path.exists(image_path):
        print("❌ Image file not found.")
        return None

    try:
        img = Image.open(image_path)
        if resize:
            img = img.resize(resize)
        img.save(output_path)
        print(f"✅ Image saved at {output_path}")
    except Exception as e:
        print(f"❌ Error processing image: {e}")





# B8: Audio Transcription

def B8(audio_path):
    if not os.path.exists(audio_path):
        print(f"❌ The file {audio_path} does not exist.")
        return None

    try:
        model = whisper.load_model("base")  # or choose any available model like "small", "medium", "large"
        result = model.transcribe(audio_path)
        return result['text']
    except Exception as e:
        print(f"❌ Error during transcription: {e}")
        return None




# B9: Markdown to HTML Conversion

def B9(md_path, output_filename):
    # Ensure the output file is stored in the /data/ directory
    output_path = os.path.join("/data", output_filename)

    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if not os.path.exists(md_path):
        print(f"Error: The file at {md_path} does not exist.")
        return None
    
    with open(md_path, 'r') as file:
        text = file.read()
    html = markdown.markdown(text)
    
    with open(output_path, 'w') as file:
        file.write(html)

    print(f"✅ HTML file saved to {output_path}")
    return html




# B10: API Endpoint for CSV Filtering


import os
import pandas as pd

def B10(csv_path, filter_column, filter_value):
    """Filters a CSV file based on a column value and returns JSON data efficiently."""
    
    # Ensure file is inside allowed directory
    if not csv_path.startswith("/data"):
        return {"error": "Invalid file path. Must be inside '/data' directory."}, 400
    
    # Check if file exists
    if not os.path.exists(csv_path):
        return {"error": f"CSV file '{csv_path}' does not exist."}, 404

    try:
        # Check the first few lines to validate the column
        sample_df = pd.read_csv(csv_path, nrows=5)
        if filter_column not in sample_df.columns:
            return {"error": f"Column '{filter_column}' not found in the CSV file."}, 400

        # Process CSV in chunks to handle large files efficiently
        filtered_results = []
        for chunk in pd.read_csv(csv_path, chunksize=1000):
            filtered_chunk = chunk[chunk[filter_column] == filter_value]
            filtered_results.extend(filtered_chunk.to_dict(orient="records"))

        return filtered_results, 200  # Return JSON data with success status

    except Exception as e:
        return {"error": f"Internal server error: {str(e)}"}, 500

def Bonus1(text_path, output_filename):
    """Summarize text from a file and save the summary."""
    with open(text_path, 'r') as file:
        text = file.read()
    summary = text[:200]  # Dummy summarization
    with open(output_filename, 'w') as file:
        file.write(summary)

def Bonus2(image_path, output_filename):
    """Convert an image to grayscale and save it."""
    img = Image.open(image_path).convert('L')
    img.save(output_filename)

def Bonus3(video_path, output_directory):
    """Extract key frames from a video and save as images."""
    cap = cv2.VideoCapture(video_path)
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % 30 == 0:  # Save every 30th frame
            cv2.imwrite(f"{output_directory}/frame_{count}.jpg", frame)
        count += 1
    cap.release()

def Bonus4(api_endpoint, output_filename):
    """Fetch JSON data from an API and save it."""
    response = requests.get(api_endpoint)
    with open(output_filename, 'w') as file:
        json.dump(response.json(), file, indent=4)

def Bonus5(input_filename, output_filename):
    """Encrypt a text file and save the encrypted version."""
    key = Fernet.generate_key()
    cipher = FILTER_INTERRUPT(key)
    with open(input_filename, 'rb') as file:
        encrypted_data = cipher.encrypt(file.read())
    with open(output_filename, 'wb') as file:
        file.write(encrypted_data)

def Bonus6(csv_path, column_name, output_filename):
    """Extract unique values from a CSV column and save them."""
    import pandas as pd
    df = pd.read_csv(csv_path)
    unique_values = df[column_name].unique()
    with open(output_filename, 'w') as file:
        file.write('\n'.join(map(str, unique_values)))

def Bonus7(text_path, output_filename):
    """Detect language of a text file and save the result."""
    from langdetect import detect
    with open(text_path, 'r') as file:
        text = file.read()
    language = detect(text)
    with open(output_filename, 'w') as file:
        file.write(language)

def Bonus8(logs_path, output_filename):
    """Filter error logs from a log file and save them."""
    with open(logs_path, 'r') as file:
        errors = [line for line in file if "ERROR" in line]
    with open(output_filename, 'w') as file:
        file.writelines(errors)

def Bonus9(json_path, output_filename):
    """Format a JSON file to be more readable."""
    with open(json_path, 'r') as file:
        data = json.load(file)
    with open(output_filename, 'w') as file:
        json.dump(data, file, indent=4)

def Bonus10(text_path, word, output_filename):
    """Count occurrences of a word in a text file and save it."""
    with open(text_path, 'r') as file:
        text = file.read()
    count = text.lower().split().count(word.lower())
    with open(output_filename, 'w') as file:
        file.write(str(count))
































