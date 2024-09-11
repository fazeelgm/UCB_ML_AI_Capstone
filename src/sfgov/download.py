import requests

url = "https://data.sfgov.org/resource/wg3w-h783.json"
output_file = "sfgov_dataset.json"

try:
    # Make a GET request to the URL
    response = requests.get(url, stream=True)
    response.raise_for_status()  # Raise an HTTPError for bad responses (4xx and 5xx)

    # Open a file to write the response content incrementally
    with open(output_file, 'wb') as file:
        # Iterate over the response in chunks
        for chunk in response.iter_content(chunk_size=8192):
            # Write each chunk to the file
            file.write(chunk)

    print(f"Dataset downloaded and saved to {output_file}")

except requests.exceptions.RequestException as e:
    print(f"An error occurred: {e}")

