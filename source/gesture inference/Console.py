from rich.live import Live
from rich.table import Table
from rich.layout import Layout
from rich.console import Console
from rich.panel import Panel
from rich.columns import Columns
import os
from typing import Callable, Union

class GestureConsole:
    # make this class a singleton
    _initialized = False

    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super(GestureConsole, cls).__new__(cls)
        return cls.instance

    def __init__(self, flags=None, max_tables=5) -> None:
        if not self._initialized:
            self._initialized = True
            self.printing = True
            self.console = ConsolePanel()
            self.layout = Layout()
            self.layout.split_column(Layout(name="upper"), Layout(name="lower"))
            self.live = Live(self.layout, auto_refresh=False)
            self.live.start()
            self.tables = []
            for i in range(max_tables):
                self.tables.append(Table())
                
    def console_flag(self, func: Callable) -> Callable:
        def print_function(*args, **kwargs) -> Union[Callable, bool]:
            if not self.printing:
                
                return
            return func(*args, **kwargs)
        return print_function

    def table(self, headers, rows, table_number=0):
        table = self.tables[table_number]
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

        self.layout["upper"].update(
            Panel.fit(
                Columns(self.tables),
            )
        )
        #self.update()

    def print(self, string: str):
        self.console.print(string)
        self.layout["lower"].update(Panel(self.console))
        #self.update()

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
