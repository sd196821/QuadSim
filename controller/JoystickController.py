from evdev import InputDevice, categorize, ecodes, KeyEvent 

gamepad = InputDevice('/dev/input/event28')

last = {
        "ABS_RZ": 128,
        "ABS_Z": 128
    }

for event in gamepad.read_loop():
    if event.type == ecodes.EV_ABS:
        absevent = categorize(event)
        if ecodes.bytype[absevent.event.type][absevent.event.code] == 'ABS_RZ':
            print("RZ: ", absevent.event.value)

        if ecodes.bytype[absevent.event.type][absevent.event.code] == 'ABS_Z':
            print("Z: ", absevent.event.value)