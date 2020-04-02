from pynput.mouse import Listener

def on_click(x, y, button, pressed):
    if pressed:
        print('Mouse clicked at (x={0}, y={1})'.format(int(x), int(y)))


with Listener(on_click=on_click) as listener:
    listener.join()