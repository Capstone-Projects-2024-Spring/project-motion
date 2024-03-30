from rich.live import Live
from rich.table import Table
from rich.layout import Layout
from rich.console import Console
from rich.panel import Panel
import os

class GestureConsole:
    # make this class a singleton
    _initialized = False

    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super(GestureConsole, cls).__new__(cls)
        return cls.instance

    def __init__(self) -> None:
        if not self._initialized:
            self._initialized = True
            self.console = ConsolePanel()
            self.layout = Layout()
            self.layout.split_column(Layout(name="upper"), Layout(name="lower"))
            self.live = Live(self.layout, auto_refresh=False)
            self.live.start()

    def table(self, headers, rows):     
        table = Table()
        for header in headers:
            table.add_column(header)

        for row in rows:
            table_row = []
            for item in row:
                table_row.append(str(f"{float(item):.3f}"))
            table.add_row(*table_row) 

        self.layout["upper"].update(Panel(table))
        self.update() 

    def print(self, string: str):
        self.console.print(string)
        self.layout["lower"].update(Panel(self.console))
        self.update()
   
    def update(self):
        self.live.update(self.layout, refresh=True)

# https://stackoverflow.com/questions/71077706/redirect-print-and-or-logging-to-panel
class ConsolePanel(Console):
    def __init__(self, *args, **kwargs):
        console_file = open(os.devnull, "w")
        super().__init__(record=True, file=console_file, *args, **kwargs)

    def __rich_console__(self, console, options):
        texts = self.export_text(clear=False).split("\n")
        for line in texts[-options.height :]:
            yield line