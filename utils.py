import os

from document_loaders import DOCUMENT_LOADERS

# bard
def read_txt_files(directory):
    """Reads the contents of the /*.txt/ files in the specified directory and returns a dict where the key is the base name of the file without the .txt and the value is the text contents of the file."""
    dict = {}
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            base_name = os.path.basename(filename).split(".txt")[0]
            with open(os.path.join(directory, filename), "r") as f:
                text = f.read().strip()
            dict[base_name] = text
    return dict


def default_dict(uDict,defaultz):
    for k,v in defaultz.items():
        uDict.setdefault(k, v) 
    return uDict



def load_document(filename):
    suffix = os.path.splitext(filename)[1][1:]  # Get the file suffix
    if suffix in DOCUMENT_LOADERS:
        return DOCUMENT_LOADERS[suffix](filename).load_and_split()
    print( f'no loader found found for {filename} ({suffix}), bro' )
    return None


