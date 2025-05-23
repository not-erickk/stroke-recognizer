import os

# Define the tests directory as a constant
ROOT_DIR = "inkSightTesting"

def build_path(file_name: str, local_path: str) -> str:
    return os.path.join(get_root_dir(), local_path, file_name)

def get_root_dir() -> str:
    current_path = os.path.abspath(os.curdir)
    path_parts = current_path.split(os.sep)

    while path_parts and path_parts[-1] != ROOT_DIR:
        path_parts = path_parts[:-1]
        if not path_parts:
            raise Exception(f"Root directory not find: {current_path}")

    return str(os.path.join(os.sep.join(path_parts)))

def from_inputs(file_name: str) -> str:
    """
    Get the path to the input image.
    Args:
        file_name: Name of the input image file
    Returns:
        Full path to the input image
    """
    return build_path(file_name, "tests/inputs")

def from_models(file_name: str) -> str:
    return build_path(file_name, "models")