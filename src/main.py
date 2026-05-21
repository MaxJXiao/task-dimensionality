"""
Main entry point for the project.
"""

import numpy as np
from src.utils import greet


def main():
    """Main function to run the application."""
    print(greet("World"))
    
    # Example: Use numpy for quick testing
    data = np.array([1, 2, 3, 4, 5])
    print(f"Data mean: {np.mean(data)}")
    print(f"Data std: {np.std(data)}")


if __name__ == "__main__":
    main()
