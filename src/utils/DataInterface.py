from pathlib import Path
import os

import numpy as np
import pandas as pd

class DataInterface:
    """
    A class to interface with the project's data directory, typically located
    in the project root. It helps in locating the data directory and listing
    its contents.
    """

    def __init__(self):
        """
        Initializes the DataInterface by locating the project root and
        the data directory within it.
        
        Raises:
            FileNotFoundError: If the project root or the data directory cannot be found.
        """
        self.project_root = self.find_project_root()
        self.data_dir = self.project_root / "data"

        if not self.data_dir.exists() or not self.data_dir.is_dir():
            raise FileNotFoundError(
                f"Data directory '{self.data_dir}' not found or is not a directory. "
                "Please ensure it exists in the project root."
            )
        
        self.initialization_routine()

    @classmethod
    def find_project_root(self) -> Path:
        """
        Finds the project root directory by searching upwards from this file's
        location for a marker file/directory (e.g., '.git' or 'CMakeLists.txt').

        Returns:
            Path: The absolute path to the project root.
        
        Raises:
            FileNotFoundError: If a project root marker cannot be found.
        """
        current_path = Path(__file__).resolve()  # Path to this data_interface.py file
        # Traverse upwards until a marker is found or filesystem root is reached
        while not (current_path / ".git").exists() and \
              not (current_path / "CMakeLists.txt").exists() and \
              not (current_path / ".project_root_marker").exists(): # Optional custom marker
            if current_path.parent == current_path:
                raise FileNotFoundError(
                    "Could not find project root. Searched for '.git' or 'CMakeLists.txt'."
                )
            current_path = current_path.parent
        return current_path

    def get_data_directory_path(self) -> Path:
        """
        Returns the absolute path to the data directory.
        """
        return self.data_dir

    def list_data_subfolders(self) -> list[Path]:
        """
        Lists all subdirectories directly within the main data directory.

        Returns:
            list[Path]: A list of Path objects representing the subdirectories.
        """
        if not self.data_dir.exists():
            print(f"Warning: Data directory '{self.data_dir}' does not exist.")
            return []
        
        subfolders = [item for item in self.data_dir.iterdir() if item.is_dir()]
        return subfolders

    def list_folder_contents(self, relative_folder_path: str = "") -> list[Path]:
        """
        Lists all files and directories within a specified folder relative to the
        main data directory. If no relative_folder_path is provided, lists
        contents of the main data directory itself.

        Args:
            relative_folder_path (str, optional): The relative path from the
                                                 main data directory. Defaults to "".

        Returns:
            list[Path]: A list of Path objects for items in the folder.
                        Returns an empty list if the folder doesn't exist.
        """
        target_folder = self.data_dir / relative_folder_path
        
        if not target_folder.exists() or not target_folder.is_dir():
            print(f"Warning: Target folder '{target_folder}' does not exist or is not a directory.")
            return []
            
        contents = list(target_folder.iterdir())
        return contents

    def get_file_path(self, relative_file_path: str) -> Path | None:
        """
        Constructs and returns the absolute path to a file within the data directory.

        Args:
            relative_file_path (str): The relative path to the file from the
                                      main data directory (e.g., "subfolder/file.csv").
        
        Returns:
            Path | None: The absolute Path object if the file exists, otherwise None.
        """
        file_path = self.data_dir / relative_file_path
        if file_path.exists() and file_path.is_file():
            return file_path
        print(f"Warning: File '{file_path}' does not exist.")
        return None

    def initialization_routine(self) -> None:
        """Prints main information when initializing the class."""
        print(f"Project Root: {self.project_root}")
        print(f"Data Directory: {self.data_dir}")
        self.print_contents_of_folder()

    def print_contents_of_folder(self, relative_folder_path: str = "") -> None:
        """Prints all the 

        Args:
            relative_folder_path (str, optional): The relative path from the
                                                 main data directory. Defaults to "".
        """
        if relative_folder_path == "":
            print("\nContents of the main data directory:")
        else:
            print(f"\nContents of the {relative_folder_path} directory:")

        _contents = self.list_folder_contents(relative_folder_path)
        if _contents:
            for item in _contents:
                item_type = "Dir" if item.is_dir() else "File"
                print(f"- {item.name} ({item_type})")
        else:
            print("Data directory is empty or does not exist.")

    def read_df(self, file_relative_path, columns_to_numeric = None):
        file_path = self.get_file_path(file_relative_path)
        df = pd.read_csv(file_path, header=0,index_col=False,)

        df = df.replace('-nan(ind)', np.nan)
        df = df.replace('nan(ind)', np.nan)

        if columns_to_numeric:
            try:
                for column in columns_to_numeric:
                    df[column] = pd.to_numeric(df[column])
            except:
                pass
                
        return df

    

if __name__ == "__main__":
    # Example usage (this will only run if you execute data_interface.py directly)
    try:
        di = DataInterface()
        print(f"Project Root: {di.project_root}")
        print(f"Data Directory: {di.data_dir}")

        print("\nSubfolders in data directory:")
        subfolders = di.list_data_subfolders()
        if subfolders:
            for folder in subfolders:
                print(f"- {folder.name} (Path: {folder})")
        else:
            print("No subfolders found.")

        print("\nContents of the main data directory:")
        main_data_contents = di.list_folder_contents()
        if main_data_contents:
            for item in main_data_contents:
                item_type = "Dir" if item.is_dir() else "File"
                print(f"- {item.name} ({item_type})")
        else:
            print("Data directory is empty or does not exist.")
            
        # Example: List contents of a specific subfolder (if it exists)
        # First, create a dummy subfolder and file for testing this part
        # (Path(di.data_dir, "test_subfolder")).mkdir(exist_ok=True)
        # (Path(di.data_dir, "test_subfolder", "sample.txt")).touch(exist_ok=True)
        
        test_subfolder_name = "supersonic_le" # Change to a subfolder you expect
        print(f"\nContents of data/{test_subfolder_name}:")
        test_contents = di.list_folder_contents(test_subfolder_name)
        if test_contents:
            for item in test_contents:
                item_type = "Dir" if item.is_dir() else "File"
                print(f"- {item.name} ({item_type})")
        else:
            print(f"Subfolder 'data/{test_subfolder_name}' is empty or does not exist.")

        # Example: Get a specific file path
        # (Path(di.data_dir, "main_output.csv")).touch(exist_ok=True) # Create dummy file for testing
        # target_csv = "main_output.csv" # Your C++ output file name
        target_csv = "data\\supersonic_le\\supersonic_le_cpp.csv" # Assuming this is in your project root for now
                                                       # If it moves to data/, change this to "supersonic_leading_edge_cpp.csv"
                                                       # or "subfolder/file.csv"
        
        # To find the CSV in project root (current C++ output location)
        csv_in_root = di.project_root / target_csv
        if csv_in_root.exists():
            print(f"\nFound CSV in project root: {csv_in_root}")
        else:
            print(f"\nCSV '{target_csv}' not found in project root.")

        # Example for a file within the 'data' directory
        # (Path(di.data_dir, "actual_data_file.csv")).touch(exist_ok=True) # Create dummy file
        file_in_data_dir_path = di.get_file_path("actual_data_file.csv") # replace with an actual file
        if file_in_data_dir_path:
             print(f"\nFound file in data dir: {file_in_data_dir_path}")
        else:
             print("\nFile 'actual_data_file.csv' not found in data dir.")


    except FileNotFoundError as e:
        print(f"Error initializing DataInterface: {e}")