import json
import zipfile

def load_json(input_path):
    """
    Load and parse a JSON file from a specified input file path.

    Args:
        input_path (str): The file path of the input JSON file.

    Returns:
        dict: A dictionary containing the parsed JSON data.
    """
    with open(input_path, "r") as f:
        content = json.load(f)
    return content


def reporthook(blocknum, blocksize, totalsize):
    """Report hook function to show progress of file download."""
    readsofar = blocknum * blocksize
    if totalsize > 0:
        percent = readsofar * 1e2 / totalsize
        s = "\r%5.1f%% %*d / %d" % (
            percent, len(str(totalsize)), readsofar, totalsize)
        print(s, end='')
        if readsofar >= totalsize:  # near the end
            print("\n")
    else:  # total size is unknown
        print("read %d\n" % (readsofar,))


def unzip_file(file_path, extract_path):
    """Extracts a ZIP file to the specified directory."""
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
        
def clean_quote(sentence):
    """
    Removes all instances of the character "'" from the input sentence.

    Args:
        sentence (str): The sentence to be cleaned.

    Returns:
        str: The cleaned sentence.
    """
    while "'" in sentence:
        sentence = sentence.replace("'", "")
    return sentence


def drop_duplicate(li):
    """
    Removes all duplicate elements from the input list.

    Args:
        li (list): The list to be deduplicated.

    Returns:
        list: The deduplicated list.
    """
    if isinstance(li[0], str):
        return list(set(li))
    else:
        temp_list = list(set([str(i) for i in li]))
        li = [eval(i) for i in temp_list]
    return li

