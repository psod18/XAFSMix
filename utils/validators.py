"""
    Custom validators for Entry fields
"""


def check_r_factor_value(event):
    val = event.widget.get()
    try:
        val = float(val)
        if not 1 > val > 0:
            val = 0.02
    except ValueError:
        val = 0.02
    finally:
        event.widget.delete(0, 'end')
        event.widget.insert(0, val)
        event.widget.config(state='disabled')


def enable_entry_editing(event):
    event.widget.config(state='normal')
    event.widget.focus_set()
