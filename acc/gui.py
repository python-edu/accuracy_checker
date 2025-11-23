from nicegui import ui
from types import SimpleNamespace
from acc.main import main  # używamy twojego main(args)

def start_script(args):

    # przechwycenie stdout
    import io, sys
    buf = io.StringIO()
    old = sys.stdout
    sys.stdout = buf

    try:
        main(args)
    finally:
        sys.stdout = old

    output_box.set_content(buf.getvalue())


def main_gui():
    ui.label('Accuracy GUI')

    pth1 = ui.input('Input file1')
    pth2 = ui.input('Input file1')
    pth3 = ui.input('Input file1')
    
    paths = [pth.value.strip() if pth for pth in (pth1, pth2, pth3)]

    args = SimpleNamespace(path = paths)

    output_box = ui.code('')  # miejsce na wyniki

    ui.button(
        'Run',
        on_click=lambda: start_script(args)
    )

    ui.run()

