
import os
import pathlib
import sys

from rich import print
from rich.filesize import decimal
from rich.markup import escape
from rich.text import Text
from rich.tree import Tree


def walk_directory(directory: pathlib.Path, tree: Tree) -> None:
    """Recursively build a Tree with directory contents."""
    # Sort dirs first then by filename
    paths = sorted(
        pathlib.Path(directory).iterdir(),
        key=lambda path: (path.is_file(), path.name.lower()),
    )
    for path in paths:
        # Remove hidden files
        if path.name.startswith("."):
            continue
        if path.is_dir():
            style = "dim" if path.name.startswith("__") else ""
            branch = tree.add(
                f"[bold magenta]:open_file_folder: [link file://{path}]{escape(path.name)}",
                style=style,
                guide_style=style,
            )
            walk_directory(path, branch)
        else:
            text_filename = Text(path.name, "green")
            text_filename.highlight_regex(r"\..*$", "bold red")
            text_filename.stylize(f"link file://{path}")
            file_size = path.stat().st_size
            text_filename.append(f" ({decimal(file_size)})", "blue")
            icon = "🐍 " if path.suffix == ".py" else "📄 "
            tree.add(Text(icon) + text_filename)

# the command for this in terminal is "python summary.py" and it will print out the tree of the current directory
try: # Try to get the directory from the command line
    directory = os.path.abspath(sys.argv[1]) # type: ignore
except IndexError: # No directory specified
    print("[b]Usage:[/] python tree.py <DIRECTORY>")
else: # type: ignore
    tree = Tree(
        f":open_file_folder: [link file://{directory}]{directory}",
        guide_style="bold bright_blue",
    )
    walk_directory(pathlib.Path(directory), tree)
    print(tree) # rich will render the tree  in the terminal


# what type of object is tree?
# copilot: It is a Tree object.
# Graham: Can I save a tree object to a file like it is displayed in the terminal?
# copilot: Yes, you can use the export method.
# Graham: What is the syntax for the export method?
# Copilot: use this syntax: tree.export("file_name")
# Graham: Okay, that doesn't exist but thank you. Is there a way to capture the contents of a terminal output as a stream and save it to a file?
# Copilot: Yes, to capture the output of a terminal, you can use the capture_output parameter. For example, subprocess.run(["ls", "-l"], capture_output=True) will capture the output of the ls -l command. You can then save the output to a file using the write method. For example, subprocess.run(["ls", "-l"], capture_output=True).stdout.write("file_name") will save the output of the ls -l command to a file called file_name. You can also use the capture_output parameter to capture the output of a terminal and save it to a file. For example, subprocess.run(["ls", "-l"], capture_output=True).stdout.write("file_name") will save the output of the ls -l command to a file called file_name.
# Graham: I will try subprocess.run(["python3", "summary.py"], capture_output=True).stdout.write("treemap.txt") and see if that works.
# Copilot: That should work.