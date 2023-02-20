import urllib.request
import os
from .utils import reporthook,unzip_file


def download_HMDB(url="https://hmdb.ca/system/downloads/current/saliva_metabolites.zip", file_path="data/saliva_metabolites.zip"):
    """Downloads the HMDB data from the given URL and extracts it to a specified directory.

    Args:
        url (str): The URL from which to download the data.
        file_path (str): The path where the data should be downloaded and extracted.

    Returns:
        str: The path to the directory where the data has been extracted.
    """
    # Create directory if it doesn't exist
    if not os.path.exists("data"):
        os.makedirs("data")

    try:
        urllib.request.urlretrieve(url, file_path, reporthook=reporthook)
        print(f"\nSuccessfully downloaded HMDB data from {url} to {file_path}")
    except urllib.error.HTTPError as e:
        print(f"HTTPError: {e.code} {e.reason}")
    except urllib.error.URLError as e:
        print(f"URLError: {e.reason}")
    except Exception as e:
        print(f"Error: {e}")

    # Extract downloaded zip file
    extract_path = os.path.splitext(file_path)[0] + ".xml"
    unzip_file(file_path=file_path, extract_path="data/")
    os.remove(file_path)
    return extract_path
