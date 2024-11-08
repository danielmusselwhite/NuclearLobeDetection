from rich.console import Console
from rich.text import Text

class Color:
    # Initialize the Console instance
    def __init__(self):
        self.console = Console()

    def print(self, text, color=None, bold=False, underline=False):
        """Method to print styled text using rich"""
        styled_text = Text(text)
        
        # Apply color if specified
        if color:
            styled_text.stylize(color)
        
        # Apply bold if specified
        if bold:
            styled_text.stylize("bold")
        
        # Apply underline if specified
        if underline:
            styled_text.stylize("underline")

        # Print the final styled text
        self.console.print(styled_text)