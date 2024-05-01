from rich.live import Live
from rich.table import Table
from rich.layout import Layout
from rich.console import Console
from rich.panel import Panel
from rich.columns import Columns
from os import devnull
from typing import Callable, Union
import threading

# https://stackoverflow.com/questions/71077706/redirect-print-and-or-logging-to-panel
class ConsolePanel(Console):
    def __init__(self, *args, **kwargs):
        console_file = open(devnull, "w")
        super().__init__(record=True, file=console_file, *args, **kwargs)

    def __rich_console__(self, console, options):
        texts = self.export_text(clear=False).split("\n")
        for line in texts[-options.height :]:
            yield line

lock = threading.Lock()

printing = True
console = ConsolePanel()
layout = Layout()
layout.split_column(Layout(name="upper"), Layout(name="lower"))
live = Live(layout, auto_refresh=False)
live.start()
tables = []
max_tables = 5

for i in range(max_tables):
    tables.append(Table())

def console_flag(func: Callable) -> Callable:
    def print_function(self, *args, **kwargs) -> Union[Callable, bool]:
        if not printing:
            return lambda *args, **kwargs: None
        return func(self, *args, **kwargs)

    return print_function

@console_flag
def table(headers, rows, table_number=0):
    with lock:
        table = tables[table_number]
        table.columns.clear()  # Clear existing columns
        table.rows.clear()  # Clear existing rows

        for header in headers:
            table.add_column(header)

        for row in rows:
            table_row = []
            for item in row:
                if isinstance(item, float):
                    table_row.append(str(f"{float(item):.3f}"))
                else:
                    table_row.append(str(item))
            table.add_row(*table_row)

        layout["upper"].update(
            Panel.fit(
                Columns(tables),
            )
        )

@console_flag
def print(string: str):
    with lock:
        console.print(string)
        layout["lower"].update(Panel(console))

def update():
    with lock:
        live.update(layout, refresh=True)


