from evdev import InputDevice, categorize, ecodes, KeyEvent
# from PIDController import controller
import numpy as np
# import evdev
import threading

class RCInput(threading.Thread):
    def __init__(self, dev_str=None):
        threading.Thread.__init__(self)
        self.rc_in = np.array([1037, 1024, 1018, 1100])
        if dev_str is None:
            self.dev = '/dev/input/event27'
        else:
            self.dev = dev_str
        self.gamepad = InputDevice(self.dev)
    # devices = [evdev.InputDevice(path) for path in evdev.list_devices()]

    # for device in devices:
    #     print(device.path, device.name, device.phys)

    # rc_in = np.array([1037, 1024, 1018, 1100])  # [throttle:Z, roll_des:X, pitch_des:Y, yaw_des:RX]
    def get_rc(self):
        for event in self.gamepad.read_loop():
            if event.type == ecodes.EV_ABS:
                absevent = categorize(event)
                if ecodes.bytype[absevent.event.type][absevent.event.code] == 'ABS_X':
                    self.rc_in[1] = absevent.event.value

                if ecodes.bytype[absevent.event.type][absevent.event.code] == 'ABS_Y':
                    self.rc_in[2] = absevent.event.value

                if ecodes.bytype[absevent.event.type][absevent.event.code] == 'ABS_Z':
                    self.rc_in[0] = absevent.event.value

                if ecodes.bytype[absevent.event.type][absevent.event.code] == 'ABS_RX':
                    self.rc_in[3] = absevent.event.value
        return self.rc_in

    def run(self):
        self.get_rc()
