import os


def print_directory_tree(startpath, indent=0):
    # Print the current directory name
    print(" " * indent + "|-- " + os.path.basename(startpath))
    indent += 4  # Increase indentation for subdirectories

    # List all files and directories in the current directory
    for item in os.listdir(startpath):
        path = os.path.join(startpath, item)
        # Skip specified directories
        if item in [".git", ".github", "hooks", "refs"]:
            continue
        if os.path.isdir(path):
            print_directory_tree(path, indent)
        else:
            print(" " * indent + "|-- " + item)


if __name__ == "__main__":
    # Automatically set the root directory to the parent directory of the script's location
    script_directory = os.path.dirname(os.path.realpath(__file__))
    parent_directory = os.path.dirname(script_directory)

    if os.path.exists(parent_directory):
        print_directory_tree(parent_directory)
    else:
        print("The directory does not exist.")
