import requests
import os
import time

# Define the base URL
base_url = "https://waterkantfestival2023.sched.com/"

# Specify the file path for the .txt file
file_path = "data/2023/links.txt"

# Create a directory to store the downloaded HTML files
os.makedirs('data/2023/html_pages', exist_ok=True)

# Headers to mimic a browser request
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'
}

# Function to fetch and save the webpage with retry logic
def fetch_and_save(url_extension):
    # Construct the full URL
    full_url = base_url + url_extension.strip()
    # Determine the filename from the last part of the URL extension
    file_name = url_extension.strip().split('/')[-1]
    # Attempt to fetch the webpage up to 3 times if a 503 error is encountered
    attempts = 3
    for attempt in range(attempts):
        try:
            response = requests.get(full_url, headers=headers)
            response.raise_for_status()  # Raises an HTTPError for bad responses
            # Save the webpage to an HTML file
            with open(os.path.join('data/2023/html_pages', file_name + '.html'), 'w', encoding='utf-8') as f:
                f.write(response.text)
            print(f"Saved {file_name}.html successfully.")
            break  # Exit the loop if successful
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 503 and attempt < attempts - 1:
                print(f"Attempt {attempt + 1} failed with 503 error, retrying after 5 seconds...")
                time.sleep(5)  # Wait for 5 seconds before retrying
            else:
                print(f"Failed to retrieve {url_extension.strip()}: {e}")
        except requests.RequestException as e:
            print(f"Failed to retrieve {url_extension.strip()}: {e}")
            break  # Exit loop if there's a non-retriable error

# Read the .txt file and process each line
with open(file_path, 'r', encoding='utf-8') as file:
    for line in file:
        fetch_and_save(line)