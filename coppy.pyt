import os
import pyperclip

def copy_py_files_to_clipboard(directory='.', max_depth=2):
    all_contents = ''

    # Get the absolute path of the root directory
    root_dir = os.path.abspath(directory)

    for root, dirs, files in os.walk(root_dir):
        # Calculate the current depth
        rel_path = os.path.relpath(root, root_dir)
        if rel_path == '.':
            depth = 1
        else:
            depth = rel_path.count(os.sep) + 1

        # If the current depth exceeds max_depth, don't traverse further
        if depth > max_depth:
            dirs[:] = []  # Clear dirs to prevent os.walk from going deeper
            continue

        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    # Format: Relative path, Name of file, and its contents
                    relative_file_path = os.path.relpath(file_path, root_dir)
                    all_contents += f"Path: {relative_file_path}\nContents:\n{content}\n\n"
                except (IOError, UnicodeDecodeError) as e:
                    print(f"Failed to read {file_path}: {e}")

    if all_contents:
        pyperclip.copy(all_contents)
        print("Python files have been copied to clipboard.")
    else:
        print("No Python files found within the specified depth.")

# Example usage:
copy_py_files_to_clipboard()
